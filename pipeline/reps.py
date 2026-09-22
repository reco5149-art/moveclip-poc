"""
reps.py — 키포인트 시계열에서 '반복(rep)'과 '세트 구간'을 찾아낸다.

문제정의서 3-2 (B) 단계. 이 프로젝트의 핵심 가설이 여기 있다:
  운동 영상은 동작이 주기적으로 반복되므로,
  사람이 눈으로 찾던 '세트의 시작과 끝'을 신호에서 기계적으로 뽑을 수 있다.

절차
  1. 관절 각도·높이 등 후보 신호 생성
  2. 움직임 에너지로 '활동 구간' 분할  (설명·휴식 구간 제거)
  3. 구간마다 가장 주기적인 신호를 자동 선택 (자기상관)
  4. 그 신호의 극값에서 반복 횟수를 카운트
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .pose import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
    PoseTrack,
)

MIN_SET_SEC = 4.0      # 이보다 짧은 활동은 세트로 보지 않음
MAX_GAP_SEC = 2.0      # 이 정도 끊김은 같은 세트로 이어붙임
MIN_REPS = 2           # 최소 반복 수
KP_CONF = 0.3

# 신호 우선순위 — 실제 코칭 영상으로 튜닝한 값 (docs/03_validation.md 참조)
#
# 코치는 설명하는 내내 팔을 쓴다. 플레이트를 들어 보이고, 손으로 방향을 가리키고,
# 시범 자세를 손으로 설명한다. 그래서 상체 신호(팔꿈치·손목)는 '운동이 아닌 구간'에서도
# 큰 진폭으로 흔들리는 대표적 오검출원이다. 반면 무릎·고관절은 실제로 운동할 때만 움직인다.
# → 하체 신호를 우선하고, 상체 신호는 다른 근거가 없을 때만 쓴다.
SIGNAL_PRIORITY = {
    "무릎각": 1.00,
    "고관절각": 0.95,
    "골반높이": 0.80,
    "팔꿈치각": 0.55,
    "손목높이": 0.50,
}

# 신호별 최소 가동범위 — 이보다 작게 움직였으면 '운동'으로 보지 않는다
# (각도는 도, 높이는 인물 크기 대비 %)
MIN_ROM = {
    "무릎각": 25.0, "고관절각": 25.0, "팔꿈치각": 30.0,
    "골반높이": 15.0, "손목높이": 20.0,
}


@dataclass
class SetSegment:
    """검출된 한 세트."""

    start: float
    end: float
    reps: int
    signal: str            # 반복 판정에 사용한 신호 이름
    periodicity: float     # 주기성 점수 0~1 (자기상관 최대값)
    period: float          # 반복 1회 주기(초)
    rom: float             # 가동범위 (신호 진폭, 각도는 도 단위)
    energy: float          # 평균 움직임 에너지
    visibility: float      # 인물 검출 비율
    rep_times: list        # 각 반복의 시각

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        for k in ("start", "end", "periodicity", "period", "rom", "energy", "visibility"):
            d[k] = round(float(d[k]), 3)
        d["rep_times"] = [round(float(t), 2) for t in self.rep_times]
        return d


# ---------------------------------------------------------------- 신호 생성

def _angle(a, b, c):
    """b를 꼭짓점으로 하는 각도(도). a,b,c: (N,2)"""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.sum(v1 * v2, axis=-1) / (n1 * n2 + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _masked(track: PoseTrack, idxs) -> np.ndarray:
    """해당 키포인트들이 모두 신뢰할 만한 프레임만 True."""
    return (track.conf[:, idxs] > KP_CONF).all(axis=1) & track.present


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """NaN 구간을 선형 보간. 전부 NaN이면 0으로."""
    x = x.astype(float).copy()
    nan = np.isnan(x)
    if nan.all():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[nan] = np.interp(idx[nan], idx[~nan], x[~nan])
    return x


def _smooth(x: np.ndarray, fps: float, win_sec: float = 0.5) -> np.ndarray:
    w = int(round(win_sec * fps))
    w = max(5, w | 1)  # 홀수
    if len(x) <= w:
        return x
    return savgol_filter(x, w, 2)


def build_signals(track: PoseTrack) -> dict:
    """반복 검출 후보 신호들을 만든다."""
    fps = track.sample_fps
    k = track.kpts
    scale = _fill_nan(track.scale())
    scale = np.maximum(scale, 1.0)

    sig = {}

    # 무릎 각도 (스쿼트·런지 계열)
    m = _masked(track, [L_HIP, L_KNEE, L_ANKLE])
    lk = np.where(m, _angle(k[:, L_HIP], k[:, L_KNEE], k[:, L_ANKLE]), np.nan)
    m = _masked(track, [R_HIP, R_KNEE, R_ANKLE])
    rk = np.where(m, _angle(k[:, R_HIP], k[:, R_KNEE], k[:, R_ANKLE]), np.nan)
    knee = np.nanmean(np.vstack([lk, rk]), axis=0)
    sig["무릎각"] = knee

    # 고관절 각도 (힌지·데드리프트 계열)
    m = _masked(track, [L_SHOULDER, L_HIP, L_KNEE])
    lh = np.where(m, _angle(k[:, L_SHOULDER], k[:, L_HIP], k[:, L_KNEE]), np.nan)
    m = _masked(track, [R_SHOULDER, R_HIP, R_KNEE])
    rh = np.where(m, _angle(k[:, R_SHOULDER], k[:, R_HIP], k[:, R_KNEE]), np.nan)
    sig["고관절각"] = np.nanmean(np.vstack([lh, rh]), axis=0)

    # 팔꿈치 각도 (푸시·프레스 계열)
    m = _masked(track, [L_SHOULDER, L_ELBOW, L_WRIST])
    le = np.where(m, _angle(k[:, L_SHOULDER], k[:, L_ELBOW], k[:, L_WRIST]), np.nan)
    m = _masked(track, [R_SHOULDER, R_ELBOW, R_WRIST])
    re = np.where(m, _angle(k[:, R_SHOULDER], k[:, R_ELBOW], k[:, R_WRIST]), np.nan)
    sig["팔꿈치각"] = np.nanmean(np.vstack([le, re]), axis=0)

    # 골반 높이 (점프·버피 등 전신 상하 운동) — 인물 크기로 정규화, 위로 갈수록 +
    m = _masked(track, [L_HIP, R_HIP])
    hip_y = np.where(m, (k[:, L_HIP, 1] + k[:, R_HIP, 1]) / 2, np.nan)
    sig["골반높이"] = -(hip_y / scale) * 100.0  # 백분율 스케일

    # 손목 높이 (상체 주도 반복)
    m = _masked(track, [L_WRIST, R_WRIST])
    wr_y = np.where(m, (k[:, L_WRIST, 1] + k[:, R_WRIST, 1]) / 2, np.nan)
    sig["손목높이"] = -(wr_y / scale) * 100.0

    out = {}
    for name, s in sig.items():
        valid = np.isfinite(s).mean()
        if valid < 0.4:  # 절반 이상 못 본 신호는 버린다
            continue
        out[name] = _smooth(_fill_nan(s), fps)
    return out


def motion_energy(track: PoseTrack) -> np.ndarray:
    """프레임 간 키포인트 이동량 = 움직임 에너지 (인물 크기로 정규화)."""
    k = track.kpts.copy()
    conf_ok = track.conf > KP_CONF
    k[~conf_ok] = np.nan
    d = np.linalg.norm(np.diff(k, axis=0), axis=-1)  # (N-1, 17)
    with np.errstate(invalid="ignore"):
        e = np.nanmean(d, axis=1)
    e = np.concatenate([[0.0], e])
    scale = np.maximum(_fill_nan(track.scale()), 1.0)
    e = _fill_nan(e) / scale * 100.0
    e[~track.present] = 0.0
    return _smooth(e, track.sample_fps, 0.6)


# ------------------------------------------------------------ 구간 / 반복

def _active_segments(energy, times, fps, present):
    """움직임이 살아있는 구간을 찾는다 (설명·휴식 제거)."""
    live = energy[present]
    if len(live) < 5:
        return []
    thr = max(np.percentile(live, 60) * 0.55, np.percentile(live, 90) * 0.18)
    active = (energy > thr) & present

    segs, start = [], None
    gap_frames = int(MAX_GAP_SEC * fps)
    gap = 0
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > gap_frames:
                segs.append((start, i - gap))
                start = None
    if start is not None:
        segs.append((start, len(active) - 1))

    return [(s, e) for s, e in segs if times[e] - times[s] >= MIN_SET_SEC]


def _periodicity(x, fps, min_period=0.7, max_period=10.0):
    """자기상관으로 주기성 점수와 주기를 구한다.

    주의: 자기상관의 '구간 내 최대값'을 그냥 쓰면 안 된다.
    부드러운 신호는 lag이 작을수록 자기상관이 높아서, 항상 최소 lag이 뽑힌다
    (실제로 첫 구현이 이 버그로 모든 구간의 주기를 0.4초로 보고했다).
    진짜 주기는 자기상관의 **국소 최대(peak)** 위치다.
    """
    x = x - x.mean()
    if np.allclose(x, 0) or len(x) < int(fps * 3):
        return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] + 1e-9
    lo = max(2, int(min_period * fps))
    hi = min(len(ac) - 1, int(max_period * fps))
    if hi <= lo + 1:
        return 0.0, 0.0

    seg = ac[lo:hi]
    peaks, _ = find_peaks(seg)
    if len(peaks) == 0:
        return 0.0, 0.0
    j = int(peaks[int(np.argmax(seg[peaks]))])
    return float(max(0.0, seg[j])), (lo + j) / fps


def _count_reps(x, times, fps, period):
    """극값 간격으로 반복을 센다."""
    rng = np.percentile(x, 95) - np.percentile(x, 5)
    if rng <= 1e-6:
        return 0, []
    dist = max(3, int(0.6 * period * fps)) if period > 0 else max(3, int(0.4 * fps))
    peaks, _ = find_peaks(x, distance=dist, prominence=rng * 0.35)
    valleys, _ = find_peaks(-x, distance=dist, prominence=rng * 0.35)
    # 반복 1회 = 내려갔다 올라오기. 더 안정적인 쪽(개수가 많은 쪽)을 기준으로.
    use = valleys if len(valleys) >= len(peaks) else peaks
    return len(use), [float(times[i]) for i in use]


def detect_sets(track: PoseTrack) -> tuple[list[SetSegment], dict]:
    """활동 구간을 찾고, 구간마다 반복 수를 센다."""
    fps = track.sample_fps
    times = track.times
    energy = motion_energy(track)
    signals = build_signals(track)
    if not signals:
        return [], {"energy": energy, "signals": {}}

    results = []
    for s, e in _active_segments(energy, times, fps, track.present):
        best = None
        for name, sig in signals.items():
            seg = sig[s:e + 1]
            if len(seg) < int(fps * 2):
                continue
            rng = float(np.percentile(seg, 95) - np.percentile(seg, 5))
            if rng < MIN_ROM.get(name, 20.0):  # 충분히 움직이지 않은 신호는 제외
                continue
            score, period = _periodicity(seg, fps)

            # 가동범위 × 관절 우선순위를 주로 보고, 주기성은 보조 가점으로만 쓴다.
            # 코칭 영상은 설명하느라 반복 간격이 불규칙해서(실측 4.8~10.6초)
            # 주기성만으로 신호를 고르면 정작 스쿼트를 놓친다.
            pick = (SIGNAL_PRIORITY.get(name, 0.5)
                    * min(1.0, rng / 60.0)
                    * (0.6 + 0.4 * score))
            if best is None or pick > best[0]:
                best = (pick, name, seg, score, period, rng)

        if best is None:
            continue
        _, name, seg, score, period, rng = best
        reps, rep_times = _count_reps(seg, times[s:e + 1], fps, period)
        if reps < MIN_REPS:
            continue

        results.append(
            SetSegment(
                start=float(times[s]),
                end=float(times[e]),
                reps=int(reps),
                signal=name,
                periodicity=float(score),
                period=float(period),
                rom=float(rng),
                energy=float(np.mean(energy[s:e + 1])),
                visibility=float(track.present[s:e + 1].mean()),
                rep_times=rep_times,
            )
        )

    debug = {"energy": energy, "signals": signals}
    return results, debug


def guess_exercise(seg: SetSegment) -> str:
    """어떤 계열 동작인지 추정 (확정이 아니라 힌트)."""
    table = {
        "무릎각": "하체 반복 동작 (스쿼트·런지 계열)",
        "고관절각": "힌지 계열 (데드리프트·굿모닝 계열)",
        "팔꿈치각": "상체 밀기·당기기 계열",
        "골반높이": "전신 상하 동작 (점프·버피 계열)",
        "손목높이": "상체 주도 반복 동작",
    }
    return table.get(seg.signal, "반복 동작")
