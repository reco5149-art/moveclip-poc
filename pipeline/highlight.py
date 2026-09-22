"""
highlight.py — 검출된 세트 중 '숏츠로 쓸 만한 구간'을 점수로 고른다.

문제정의서 3-2 (C) 단계.
사람이 20~30분 스크러빙하며 하던 '탐색'을 점수화로 대체한다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .reps import SetSegment, guess_exercise
from .stt import Utterance, speech_density

PRE_ROLL = 1.2      # 동작 시작 전 여유
POST_ROLL = 1.2     # 동작 끝난 뒤 여유
MIN_CLIP = 8.0
MAX_CLIP = 45.0

WEIGHTS = {
    "reps": 0.25,         # 반복이 많을수록 보여줄 게 많다
    "energy": 0.20,       # 움직임이 큰 쪽이 눈길을 끈다
    "periodicity": 0.20,  # 리듬이 깨끗해야 보기 좋다
    "visibility": 0.15,   # 사람이 계속 보여야 한다
    "speech": 0.20,       # 코칭 멘트가 있으면 콘텐츠가 된다
}


@dataclass
class Clip:
    rank: int
    start: float
    end: float
    score: float
    reps: int
    exercise: str
    signal: str
    parts: dict
    reason: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self):
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        for k in ("start", "end", "score"):
            d[k] = round(float(d[k]), 2)
        d["parts"] = {k: round(float(v), 3) for k, v in d["parts"].items()}
        return d


def _norm(vals):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return v
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return np.ones_like(v) * 0.5
    return (v - lo) / (hi - lo)


def _trim(seg: SetSegment, duration: float) -> tuple[float, float]:
    """세트가 너무 길면 반복이 가장 촘촘한 구간만 잘라낸다."""
    start = max(0.0, seg.start - PRE_ROLL)
    end = min(duration, seg.end + POST_ROLL)
    if end - start <= MAX_CLIP:
        return start, end

    rt = np.array(seg.rep_times)
    if len(rt) < 2:
        return start, start + MAX_CLIP

    # MAX_CLIP 길이 창을 밀면서 반복이 가장 많이 들어가는 위치를 찾는다
    best, best_n = start, -1
    for s in np.arange(start, end - MAX_CLIP, 1.0):
        n = int(((rt >= s) & (rt <= s + MAX_CLIP)).sum())
        if n > best_n:
            best, best_n = s, n
    return best, best + MAX_CLIP


def rank_clips(
    sets: list[SetSegment],
    utts: list[Utterance],
    duration: float,
    top_k: int = 3,
) -> list[Clip]:
    if not sets:
        return []

    reps = _norm([min(s.reps, 20) for s in sets])
    energy = _norm([s.energy for s in sets])
    period = _norm([s.periodicity for s in sets])
    vis = np.array([s.visibility for s in sets])
    speech = _norm([speech_density(utts, s.start, s.end) for s in sets]) if utts \
        else np.zeros(len(sets))

    scored = []
    for i, s in enumerate(sets):
        parts = {
            "reps": float(reps[i]),
            "energy": float(energy[i]),
            "periodicity": float(period[i]),
            "visibility": float(vis[i]),
            "speech": float(speech[i]),
        }
        total = sum(WEIGHTS[k] * v for k, v in parts.items())
        scored.append((total, parts, s))

    scored.sort(key=lambda x: -x[0])

    clips = []
    used = []
    for total, parts, s in scored:
        st, en = _trim(s, duration)
        if en - st < MIN_CLIP:
            continue
        # 이미 뽑은 구간과 절반 이상 겹치면 건너뛴다
        if any(min(en, e) - max(st, b) > 0.5 * (en - st) for b, e in used):
            continue
        used.append((st, en))

        # 이 클립이 뽑힌 '가장 큰 이유' = 가중 기여가 제일 큰 항목
        top = max(parts.items(), key=lambda kv: kv[1] * WEIGHTS[kv[0]])
        reason_map = {
            "reps": f"반복 {s.reps}회로 분량이 충분",
            "energy": "움직임이 커서 시선을 끎",
            "periodicity": "동작 리듬이 일정해 보기 좋음",
            "visibility": "인물이 계속 프레임에 있음",
            "speech": "코칭 멘트가 많아 자막·카드 소재가 됨",
        }
        clips.append(
            Clip(
                rank=len(clips) + 1,
                start=st,
                end=en,
                score=float(total),
                reps=s.reps,
                exercise=guess_exercise(s),
                signal=s.signal,
                parts=parts,
                reason=reason_map[top[0]],
            )
        )
        if len(clips) >= top_k:
            break

    return clips
