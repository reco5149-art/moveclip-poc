"""
run.py — 전체 파이프라인 오케스트레이터.

입력(영상) → AI 처리 → 출력(숏츠 + 자막 + 카드뉴스 + 캡션)
각 단계 소요 시간을 재서 result.json에 남긴다. (개선 효과 검증용)
"""

from __future__ import annotations

import json
import os
import time

from . import cards as cards_mod
from .highlight import rank_clips
from .pose import extract_pose
from .render import grab_frame, render_vertical
from .reps import detect_sets
from .stt import to_srt, transcribe


def process(
    video_path: str,
    out_dir: str,
    top_k: int = 3,
    sample_fps: float = 5.0,
    language: str = "ko",
    handle: str = "@my_movement",
    do_stt: bool = True,
    stt_model: str | None = None,
    progress=None,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    timing = {}
    t_all = time.time()

    def say(stage, pct):
        if progress:
            progress(stage, pct)

    # 1) 자세 추출 (YOLO11-pose)
    say("포즈 추출", 0.0)
    t = time.time()
    track = extract_pose(video_path, sample_fps=sample_fps,
                         progress=lambda p: say("포즈 추출", p * 0.45))
    timing["pose"] = time.time() - t

    # 2) 반복·세트 검출 (신호처리)
    say("반복 구간 검출", 0.50)
    t = time.time()
    sets, _ = detect_sets(track)
    timing["reps"] = time.time() - t

    # 3) 음성인식 (Silero VAD + Whisper)
    utts = []
    t = time.time()
    if do_stt:
        say("음성 인식", 0.55)
        utts = transcribe(video_path, out_dir, language=language, model_size=stt_model,
                          progress=lambda p: say("음성 인식", 0.55 + p * 0.15))
    timing["stt"] = time.time() - t

    # 4) 하이라이트 순위
    say("하이라이트 선별", 0.72)
    t = time.time()
    clips = rank_clips(sets, utts, track.duration, top_k=top_k)
    timing["rank"] = time.time() - t

    if utts:
        with open(os.path.join(out_dir, "transcript.srt"), "w") as f:
            f.write(to_srt(utts))
        with open(os.path.join(out_dir, "transcript.json"), "w") as f:
            json.dump([u.to_dict() for u in utts], f, ensure_ascii=False, indent=2)

    # 5) 숏츠 렌더 + 카드뉴스
    t = time.time()
    outputs = []
    for i, clip in enumerate(clips):
        base = 0.75 + (i / max(len(clips), 1)) * 0.25
        say(f"숏츠 {i + 1}/{len(clips)} 렌더링", base)
        mp4 = os.path.join(out_dir, f"short_{clip.rank}.mp4")
        render_vertical(
            video_path, mp4, clip.start, clip.end, track, utts,
            label=f"{clip.exercise.split(' ')[0]} · {clip.reps}회",
            progress=lambda p, b=base: say("숏츠 렌더링", b + p * 0.08),
        )

        cue_pool = [u for u in utts if u.end > clip.start and u.start < clip.end] or utts
        cues = cards_mod.pick_cues(cue_pool, n=4)
        card_dir = os.path.join(out_dir, f"cards_{clip.rank}")
        frame = grab_frame(video_path, (clip.start + clip.end) / 2)
        card_paths = cards_mod.build_cards(card_dir, clip, cues, frame, handle=handle)
        caption = cards_mod.build_caption(clip, cues)
        with open(os.path.join(card_dir, "caption.txt"), "w") as f:
            f.write(caption)

        outputs.append({
            **clip.to_dict(),
            "video": os.path.relpath(mp4, out_dir),
            "cards": [os.path.relpath(p, out_dir) for p in card_paths],
            "cues": cues,
            "caption": caption,
        })
    timing["render"] = time.time() - t
    timing["total"] = time.time() - t_all

    result = {
        "source": os.path.basename(video_path),
        "video": {
            "duration_sec": round(track.duration, 2),
            "resolution": f"{track.width}x{track.height}",
            "src_fps": round(track.src_fps, 2),
            "analyzed_fps": round(track.sample_fps, 2),
            "frames_analyzed": track.meta["frames_analyzed"],
            "person_detect_ratio": round(track.detect_ratio, 3),
        },
        "sets_detected": [s.to_dict() for s in sets],
        "utterances": len(utts),
        "clips": outputs,
        "timing_sec": {k: round(v, 2) for k, v in timing.items()},
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    say("완료", 1.0)
    return result
