"""
make_sample.py — 검증용 합성 영상 생성기 (정답 라벨 포함).

실제 회원이 나온 촬영본은 공개 저장소에 올릴 수 없다. 그렇다고 영상이 없으면
"반복을 몇 개 맞췄나"를 채점할 수 없다. 그래서 **정답을 내가 정한 영상**을 만든다.

방법:
  1) 실제 인물 사진(YOLO 기본 배포 이미지)에서 사람을 분할해 잘라낸다.
  2) 골반~무릎 구간을 프레임마다 눌러서 '스쿼트로 앉았다 일어나는' 움직임을 만든다.
     → 픽셀은 진짜 사람이라 YOLO가 정상적으로 검출한다.
  3) 세트 구성(반복 횟수·휴식 길이)은 내가 지정하므로 그대로 정답 라벨이 된다.

주의: 이건 파이프라인 채점용 테스트 자산이지 실제 운동 영상이 아니다.
      최종 검증은 반드시 본인 촬영본으로 한 번 더 한다.

사용법: python tools/make_sample.py data/sample_workout.mp4
"""

import json
import os
import sys

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30
LEAD_IN = 5.0

# (반복 횟수, 반복 1회 주기(초), 세트 뒤 휴식(초)) — 이게 정답 라벨이 된다
PLAN = [
    {"reps": 8, "period": 2.0, "rest_after": 6.0},
    {"reps": 6, "period": 1.6, "rest_after": 5.0},
    {"reps": 10, "period": 1.4, "rest_after": 4.0},
]


def cut_person():
    """YOLO 기본 배포 이미지에서 전신이 보이는 사람 1명을 분할해 잘라낸다."""
    import ultralytics
    from ultralytics import YOLO

    src = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
    img = cv2.imread(src)

    kp = YOLO("models/yolo11n-pose.pt").predict(img, verbose=False, imgsz=960)[0]
    kc = kp.keypoints.conf.cpu().numpy()
    best = int(np.argmax((kc > 0.5).sum(axis=1) * kp.boxes.conf.cpu().numpy()))
    box = kp.boxes.xyxy.cpu().numpy()[best]
    key = kp.keypoints.xy.cpu().numpy()[best]

    seg = YOLO("models/yolo11n-seg.pt").predict(img, verbose=False, imgsz=960)[0]
    # pose에서 고른 사람과 가장 많이 겹치는 마스크를 찾는다
    masks = seg.masks.data.cpu().numpy() if seg.masks is not None else []
    mask = None
    bx = np.zeros(img.shape[:2], np.uint8)
    bx[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
    best_iou = 0
    for m in masks:
        m = cv2.resize(m, (img.shape[1], img.shape[0])) > 0.5
        iou = (m & bx.astype(bool)).sum() / max(m.sum(), 1)
        if iou > best_iou:
            best_iou, mask = iou, m.astype(np.uint8)
    if mask is None:
        mask = bx

    x1, y1, x2, y2 = [int(v) for v in box]
    pad = 10
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)

    rgb = img[y1:y2, x1:x2].copy()
    alpha = (mask[y1:y2, x1:x2] * 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    joints = {
        "hip": float(np.mean([key[11][1], key[12][1]]) - y1),
        "knee": float(np.mean([key[13][1], key[14][1]]) - y1),
        "ankle": float(np.mean([key[15][1], key[16][1]]) - y1),
    }
    return rgb, alpha, joints


def squat_warp(rgb, alpha, joints, s):
    """s=0 서있음, s=1 최대 하강. 골반~무릎 구간을 눌러 앉는 동작을 만든다."""
    h = rgb.shape[0]
    hip, knee = joints["hip"], joints["knee"]
    drop = (knee - hip) * 0.55 * s  # 골반이 내려가는 양

    src_pts = np.array([0.0, hip, knee, h - 1.0])
    dst_pts = np.array([drop, hip + drop, knee, h - 1.0])
    rows = np.arange(h, dtype=np.float32)
    y_src = np.interp(rows, dst_pts, src_pts)

    map_y = np.repeat(y_src[:, None], rgb.shape[1], axis=1).astype(np.float32)
    map_x = np.repeat(np.arange(rgb.shape[1], dtype=np.float32)[None, :], h, axis=0)

    warped = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wa = cv2.remap(alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wa[: int(drop)] = 0  # 내려간 만큼 머리 위는 비운다
    return warped, wa


def background():
    bg = np.full((H, W, 3), 42, np.uint8)
    cv2.rectangle(bg, (0, 0), (W, int(H * 0.74)), (62, 60, 57), -1)
    for x in range(0, W, 170):
        cv2.line(bg, (x, 0), (x, int(H * 0.74)), (52, 50, 47), 2)
    cv2.rectangle(bg, (0, int(H * 0.74)), (W, H), (36, 42, 50), -1)
    cv2.rectangle(bg, (70, int(H * 0.36)), (210, int(H * 0.74)), (44, 48, 64), -1)
    cv2.putText(bg, "TEST FOOTAGE (synthetic)", (W - 430, H - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 115), 2)
    return bg


def composite(bg, person, alpha, cx, bottom_y, target_h):
    scale = target_h / person.shape[0]
    p = cv2.resize(person, None, fx=scale, fy=scale)
    a = cv2.resize(alpha, None, fx=scale, fy=scale).astype(np.float32) / 255.0
    ph, pw = p.shape[:2]
    x0, y0 = int(cx - pw / 2), int(bottom_y - ph)
    x1, y1 = min(W, x0 + pw), min(H, y0 + ph)
    x0, y0 = max(0, x0), max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return bg
    ps, as_ = p[: y1 - y0, : x1 - x0], a[: y1 - y0, : x1 - x0, None]
    bg[y0:y1, x0:x1] = (bg[y0:y1, x0:x1] * (1 - as_) + ps * as_).astype(np.uint8)
    return bg


def main(out_path="data/sample_workout.mp4", distractor=False):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rgb, alpha, joints = cut_person()
    bg = background()
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    target_h, bottom_y = int(H * 0.80), int(H * 0.94)
    truth, t = [], 0.0

    def emit(dur, fn):
        nonlocal t
        for i in range(int(dur * FPS)):
            s = float(np.clip(fn(i / FPS), 0, 1))
            p, a = squat_warp(rgb, alpha, joints, s)
            frame = composite(bg.copy(), p, a, W * 0.45 + np.sin(t * 0.25) * 55,
                              bottom_y, target_h)
            if distractor:
                # 실패 케이스용: 옆에서 다른 사람이 서성인다 (그룹 수업 상황 모사)
                d, da = squat_warp(rgb, alpha, joints, 0.05)
                frame = composite(frame, d, da,
                                  W * 0.80 + np.sin(t * 0.55) * 160,
                                  bottom_y - 6, int(target_h * 0.96))
            vw.write(frame)
            t += 1 / FPS

    emit(LEAD_IN, lambda s: 0.02)
    for seg in PLAN:
        start = t
        emit(seg["reps"] * seg["period"],
             lambda s, p=seg["period"]: (1 - np.cos(2 * np.pi * s / p)) / 2 * 0.95)
        truth.append({"start": round(start, 2), "end": round(t, 2), "reps": seg["reps"]})
        emit(seg["rest_after"], lambda s: 0.03)

    vw.release()
    meta = {"video": out_path, "fps": FPS, "duration": round(t, 2),
            "note": "합성 테스트 영상 — 세트 구성이 곧 정답 라벨", "sets": truth}
    with open(out_path.rsplit(".", 1)[0] + "_truth.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "data/sample_workout.mp4",
         distractor="--distractor" in sys.argv)
