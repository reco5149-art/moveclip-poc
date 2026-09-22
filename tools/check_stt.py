"""
check_stt.py — 음성인식 단계 단독 점검.

합성 테스트 영상에는 음성이 없어서 STT를 채점할 수 없다.
대신 모델 배포본에 들어있는 참조 음성으로 '이 단계가 실제로 도는지'를 확인한다.
본인 촬영본(한국어 코칭 음성)이 준비되면 그 파일 경로를 인자로 넘겨 재검증한다.

    python tools/check_stt.py                 # 참조 음성으로 동작 확인
    python tools/check_stt.py 내영상.mp4 ko   # 실제 영상으로 검증
"""

import glob
import os
import sys
import time
import wave

import numpy as np

import sys as _s, os as _o
_s.path.insert(0, _o.getcwd())
from pipeline.stt import resolve_model
MODEL, PRE = resolve_model()


def recognizer(language):
    import sherpa_onnx
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=f"{MODEL}/{PRE}-encoder.int8.onnx",
        decoder=f"{MODEL}/{PRE}-decoder.int8.onnx",
        tokens=f"{MODEL}/{PRE}-tokens.txt",
        language=language, task="transcribe",
        num_threads=max(2, (os.cpu_count() or 4) // 2),
    )


def from_wavs():
    rec = recognizer("en")
    total_audio = total_time = 0.0
    for p in sorted(glob.glob(f"{MODEL}/test_wavs/*.wav"))[:2]:
        with wave.open(p) as wf:
            sr, n = wf.getframerate(), wf.getnframes()
            data = np.frombuffer(wf.readframes(n), np.int16).astype(np.float32) / 32768
        s = rec.create_stream()
        s.accept_waveform(sr, data)
        t = time.time()
        rec.decode_stream(s)
        el = time.time() - t
        total_audio += n / sr
        total_time += el
        print(f"\n[{os.path.basename(p)}] 음성 {n / sr:.1f}초 → 디코딩 {el:.1f}초")
        print(f"  {s.result.text.strip()}")
    print(f"\n처리 속도: 실시간 대비 {total_time / total_audio:.2f}x "
          f"(음성 {total_audio:.0f}초를 {total_time:.0f}초에 처리)")


def from_video(path, language):
    sys.path.insert(0, os.getcwd())
    from pipeline.stt import transcribe
    t = time.time()
    utts = transcribe(path, "outputs/_stt_check", language=language)
    print(f"\n{len(utts)}개 발화 구간 / {time.time() - t:.1f}초 소요\n")
    for u in utts[:25]:
        print(f"  [{u.start:6.1f}s] {u.text}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from_video(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ko")
    else:
        from_wavs()
