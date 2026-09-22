"""
pose.py — 영상에서 사람의 관절 키포인트 시계열을 뽑아낸다.

문제정의서 3-2 (A) 단계에 해당.
YOLO11-pose로 프레임마다 17개 관절 좌표를 추출하고,
프레임 간에 '같은 사람'이 유지되도록 주 피사체를 추적한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

os.environ.setdefault("YOLO_VERBOSE", "0")

# COCO-17 키포인트 인덱스
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class PoseTrack:
    """주 피사체 1명의 키포인트 시계열."""

    times: np.ndarray  # (N,) 초 단위 타임스탬프
    kpts: np.ndarray  # (N, 17, 2) 픽셀 좌표
    conf: np.ndarray  # (N, 17) 키포인트 신뢰도
    boxes: np.ndarray  # (N, 4) x1,y1,x2,y2
    present: np.ndarray  # (N,) bool — 해당 프레임에서 사람이 검출됐는지

    width: int = 0
    height: int = 0
    src_fps: float = 0.0
    duration: float = 0.0
    sample_fps: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def detect_ratio(self) -> float:
        return float(self.present.mean()) if len(self.present) else 0.0

    def center_x(self) -> np.ndarray:
        """프레임별 인물 중심 x (미검출 구간은 NaN)."""
        cx = np.full(len(self.times), np.nan)
        ok = self.present
        if ok.any():
            # 어깨·골반 중심을 우선 사용하고, 없으면 박스 중심
            core = self.kpts[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :]
            core_c = self.conf[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
            good = core_c > 0.3
            for i in np.where(ok)[0]:
                if good[i].sum() >= 2:
                    cx[i] = core[i][good[i], 0].mean()
                else:
                    cx[i] = (self.boxes[i, 0] + self.boxes[i, 2]) / 2
        return cx

    def scale(self) -> np.ndarray:
        """프레임별 인물 크기(박스 높이). 정규화 기준."""
        h = self.boxes[:, 3] - self.boxes[:, 1]
        h[~self.present] = np.nan
        return h


def _pick_person(result, prev_center):
    """한 프레임의 검출 결과에서 주 피사체 1명을 고른다.

    면적 × 신뢰도를 기본 점수로 하되, 직전 프레임의 중심과 가까운 쪽에
    가산점을 줘서 프레임 간 인물이 튀지 않게 한다.
    """
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    box_conf = result.boxes.conf.cpu().numpy()
    kxy = result.keypoints.xy.cpu().numpy()
    kcf = (
        result.keypoints.conf.cpu().numpy()
        if result.keypoints.conf is not None
        else np.ones(kxy.shape[:2])
    )

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if areas.max() <= 0:
        return None
    score = (areas / areas.max()) * box_conf

    if prev_center is not None:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        dist = np.hypot(cx - prev_center[0], cy - prev_center[1])
        diag = np.hypot(*prev_center) + 1e-6
        score = score * (1.0 + 0.6 * np.exp(-dist / (0.25 * diag + 1e-6)))

    i = int(score.argmax())
    if box_conf[i] < 0.35:
        return None
    return boxes[i], kxy[i], kcf[i]


def extract_pose(
    video_path: str,
    sample_fps: float = 5.0,
    imgsz: int = 640,
    model_path: str = "models/yolo11n-pose.pt",
    progress=None,
) -> PoseTrack:
    """영상을 sample_fps로 샘플링하며 포즈를 추출한다."""
    from ultralytics import YOLO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / src_fps if n_frames else 0.0

    step = max(1, int(round(src_fps / sample_fps)))
    model = YOLO(model_path)

    times, kpts, confs, boxes, present = [], [], [], [], []
    prev_center = None
    idx = 0
    total_samples = max(1, n_frames // step) if n_frames else 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            res = model.predict(frame, verbose=False, imgsz=imgsz)[0]
            picked = _pick_person(res, prev_center)
            times.append(idx / src_fps)
            if picked is None:
                boxes.append(np.zeros(4))
                kpts.append(np.zeros((17, 2)))
                confs.append(np.zeros(17))
                present.append(False)
            else:
                b, k, c = picked
                boxes.append(b)
                kpts.append(k)
                confs.append(c)
                present.append(True)
                prev_center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

            if progress and total_samples:
                done = len(times)
                if done % 10 == 0 or done == total_samples:
                    progress(done / total_samples)
        idx += 1

    cap.release()

    if not times:
        raise RuntimeError("프레임을 하나도 읽지 못했습니다.")

    return PoseTrack(
        times=np.array(times),
        kpts=np.array(kpts),
        conf=np.array(confs),
        boxes=np.array(boxes),
        present=np.array(present, dtype=bool),
        width=width,
        height=height,
        src_fps=float(src_fps),
        duration=float(duration or (times[-1] + 1 / src_fps)),
        sample_fps=float(src_fps / step),
        meta={"frames_analyzed": len(times), "step": step, "imgsz": imgsz},
    )
