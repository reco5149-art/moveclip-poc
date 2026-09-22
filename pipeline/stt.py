"""
stt.py — 영상의 발화를 타임스탬프와 함께 텍스트로 만든다.

문제정의서 3-2 (E) 단계. 자막을 직접 타이핑하던 작업을 대체하고,
동시에 (F) 카드뉴스의 원재료인 '내가 실제로 한 말'을 확보한다.

VAD(음성 구간 검출) → 구간별 Whisper 디코딩 순서로 처리한다.
Whisper는 30초 창 단위라 긴 영상을 통째로 넣을 수 없기 때문.
"""

from __future__ import annotations

import os
import re
import subprocess
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
MAX_CHUNK_SEC = 25.0

# 운동·통증케어 도메인 용어 교정 사전 (문제정의서 7. 리스크 대응)
TERM_FIXES = [
    (r"하이 ?록스|하이 ?록시|하이록쓰", "하이록스"),
    (r"크로스 ?핏|크로스 ?피트", "크로스핏"),
    (r"스 ?쿼 ?트|스콰트", "스쿼트"),
    (r"데드 ?리 ?프트", "데드리프트"),
    (r"코 ?어", "코어"),
    (r"힙 ?힌지|힙 ?힌쥐", "힙힌지"),
    (r"가동 ?범위", "가동범위"),
    (r"버 ?피", "버피"),
    (r"런 ?지", "런지"),
    (r"골 ?반", "골반"),
]


@dataclass
class Utterance:
    start: float
    end: float
    text: str

    def to_dict(self):
        return {"start": round(self.start, 2), "end": round(self.end, 2), "text": self.text}


def has_audio(video_path: str) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return "audio" in out.stdout


def extract_audio(video_path: str, out_wav: str) -> str | None:
    if not has_audio(video_path):
        return None
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video_path,
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-vn", out_wav],
        check=True,
    )
    return out_wav


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path) as wf:
        n = wf.getnframes()
        data = np.frombuffer(wf.readframes(n), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def _fix_terms(text: str) -> str:
    for pat, rep in TERM_FIXES:
        text = re.sub(pat, rep, text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_model(size: str | None = None) -> tuple[str, str]:
    """(모델 폴더, 파일 접두사) 반환.

    한국어 정확도는 small이 base보다 확실히 낫다(실측 문자정확도 61% → 71%).
    그래서 small이 받아져 있으면 자동으로 그쪽을 쓴다.
    """
    order = [size] if size else ["small", "base"]
    for s in order:
        d = f"models/sherpa-onnx-whisper-{s}"
        if os.path.isdir(d):
            return d, s
    raise FileNotFoundError("Whisper 모델이 없습니다. bash setup_models.sh 를 먼저 실행하세요.")


def transcribe(
    video_path: str,
    work_dir: str,
    model_size: str | None = None,
    vad_model: str = "models/silero_vad.onnx",
    language: str = "ko",
    progress=None,
) -> list[Utterance]:
    """영상에서 발화 구간별 텍스트를 뽑는다. 실패하면 빈 리스트."""
    os.makedirs(work_dir, exist_ok=True)
    wav_path = os.path.join(work_dir, "audio.wav")
    try:
        if extract_audio(video_path, wav_path) is None:
            return []
    except Exception:
        return []

    try:
        import sherpa_onnx
    except ImportError:
        return []

    samples = _read_wav(wav_path)
    if len(samples) < SAMPLE_RATE * 0.5:
        return []

    # 1) VAD로 말한 구간만 골라낸다
    vad_cfg = sherpa_onnx.VadModelConfig()
    vad_cfg.silero_vad.model = vad_model
    vad_cfg.silero_vad.threshold = 0.5
    vad_cfg.silero_vad.min_silence_duration = 0.35
    vad_cfg.silero_vad.min_speech_duration = 0.3
    vad_cfg.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=60)

    window = 512
    spans = []
    for i in range(0, len(samples), window):
        chunk = samples[i:i + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        vad.accept_waveform(chunk)
        while not vad.empty():
            seg = vad.front
            spans.append((seg.start / SAMPLE_RATE, len(seg.samples) / SAMPLE_RATE,
                          np.array(seg.samples, dtype=np.float32)))
            vad.pop()
    vad.flush()
    while not vad.empty():
        seg = vad.front
        spans.append((seg.start / SAMPLE_RATE, len(seg.samples) / SAMPLE_RATE,
                      np.array(seg.samples, dtype=np.float32)))
        vad.pop()

    if not spans:
        return []

    # 30초 넘는 구간은 쪼갠다 (Whisper 입력 한계)
    chunks = []
    for start, dur, data in spans:
        if dur <= MAX_CHUNK_SEC:
            chunks.append((start, dur, data))
        else:
            n = int(np.ceil(dur / MAX_CHUNK_SEC))
            size = len(data) // n
            for j in range(n):
                piece = data[j * size:(j + 1) * size]
                chunks.append((start + j * size / SAMPLE_RATE, len(piece) / SAMPLE_RATE, piece))

    # 2) 구간마다 Whisper 디코딩
    model_dir, pre = resolve_model(model_size)
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=os.path.join(model_dir, f"{pre}-encoder.int8.onnx"),
        decoder=os.path.join(model_dir, f"{pre}-decoder.int8.onnx"),
        tokens=os.path.join(model_dir, f"{pre}-tokens.txt"),
        language=language,
        task="transcribe",
        num_threads=max(2, (os.cpu_count() or 4) // 2),
    )

    out = []
    for i, (start, dur, data) in enumerate(chunks):
        s = rec.create_stream()
        s.accept_waveform(SAMPLE_RATE, data)
        rec.decode_stream(s)
        text = _fix_terms(s.result.text)
        if text and not re.fullmatch(r"[\s.,!?·…]*", text):
            out.append(Utterance(start=start, end=start + dur, text=text))
        if progress:
            progress((i + 1) / len(chunks))

    return out


def to_srt(utts: list[Utterance], offset: float = 0.0) -> str:
    def ts(t):
        t = max(0.0, t - offset)
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"

    lines = []
    for i, u in enumerate(utts, 1):
        lines += [str(i), f"{ts(u.start)} --> {ts(u.end)}", u.text, ""]
    return "\n".join(lines)


def speech_density(utts: list[Utterance], start: float, end: float) -> float:
    """구간 내 발화 밀도 (초당 글자 수). 하이라이트 점수에 쓰인다."""
    if end <= start:
        return 0.0
    chars = 0
    for u in utts:
        ov = min(u.end, end) - max(u.start, start)
        if ov > 0:
            chars += len(u.text) * (ov / max(u.end - u.start, 1e-6))
    return chars / (end - start)
