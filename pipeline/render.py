"""
render.py — 하이라이트 구간을 9:16 세로 숏츠로 렌더링한다.

문제정의서 3-2 (D)+(E) 단계.
  - 인물 중심을 따라가며 자동 크롭 (수동 키프레임 대체)
  - 자막 번인 (직접 타이핑 대체)

크롭 위치는 키포인트 중심을 강하게 스무딩해서 쓴다.
프레임마다 그대로 따라가면 화면이 떨리기 때문에,
'데드존 + 지수이동평균'으로 카메라맨이 천천히 패닝하는 느낌을 만든다.
"""

from __future__ import annotations

import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .fonts import korean_font
from .pose import PoseTrack
from .stt import Utterance

TARGET_W, TARGET_H = 1080, 1920
DEADZONE_RATIO = 0.06   # 인물이 이 정도 움직여야 카메라가 따라감
EMA_ALPHA = 0.06        # 작을수록 더 천천히 따라감


def _smooth_center(track: PoseTrack, times: np.ndarray, crop_w: int) -> np.ndarray:
    """출력 프레임 시각마다 크롭 중심 x를 계산한다."""
    cx = track.center_x()
    ok = np.isfinite(cx)
    if ok.sum() < 2:
        return np.full(len(times), track.width / 2)

    raw = np.interp(times, track.times[ok], cx[ok])

    dead = crop_w * DEADZONE_RATIO
    out = np.empty_like(raw)
    cur = raw[0]
    for i, target in enumerate(raw):
        if abs(target - cur) > dead:
            goal = target - np.sign(target - cur) * dead
            cur += EMA_ALPHA * (goal - cur)
        out[i] = cur

    half = crop_w / 2
    return np.clip(out, half, max(half, track.width - half))


def _active_subs(utts: list[Utterance], t: float) -> str:
    for u in utts:
        if u.start <= t <= u.end:
            return u.text
    return ""


def _wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for ch in text:
        if draw.textbbox((0, 0), cur + ch, font=font)[2] > max_w and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines[-3:]


def _draw_overlay(frame_bgr, text, label, font_sub, font_label):
    """자막과 좌상단 라벨을 그린다."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    W, H = img.size

    if label:
        pad = 16
        bbox = d.textbbox((0, 0), label, font=font_label)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.rounded_rectangle(
            [40, 48, 40 + w + pad * 2, 48 + h + pad * 2], radius=14, fill=(17, 17, 17, 255)
        )
        d.text((40 + pad, 48 + pad - bbox[1]), label, font=font_label, fill=(255, 255, 255))

    if text:
        lines = _wrap(d, text, font_sub, W - 120)
        line_h = font_sub.size + 16
        total = line_h * len(lines)
        y = H - 320 - total
        for ln in lines:
            bbox = d.textbbox((0, 0), ln, font=font_sub)
            w = bbox[2] - bbox[0]
            x = (W - w) / 2
            # 외곽선으로 어떤 배경에서도 읽히게
            for ox, oy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)):
                d.text((x + ox, y + oy - bbox[1]), ln, font=font_sub, fill=(0, 0, 0))
            d.text((x, y - bbox[1]), ln, font=font_sub, fill=(255, 255, 255))
            y += line_h

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_vertical(
    video_path: str,
    out_path: str,
    start: float,
    end: float,
    track: PoseTrack,
    utts: list[Utterance] | None = None,
    label: str = "",
    fps_out: float | None = None,
    progress=None,
) -> str:
    """[start, end] 구간을 9:16 세로 영상으로 자르고 자막을 입혀 저장한다."""
    utts = utts or []
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = fps_out or min(30.0, src_fps)

    crop_w = int(min(W, round(H * 9 / 16)))
    crop_w -= crop_w % 2
    n_out = max(1, int(round((end - start) * fps_out)))
    out_times = start + np.arange(n_out) / fps_out
    centers = _smooth_center(track, out_times, crop_w)

    font_sub = korean_font(58)
    font_label = korean_font(38)

    has_audio = (
        subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=codec_type", "-of", "csv=p=0", video_path],
            capture_output=True, text=True,
        ).stdout.strip() != ""
    )

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{TARGET_W}x{TARGET_H}", "-r", f"{fps_out}", "-i", "-",
    ]
    if has_audio:
        cmd += ["-ss", f"{start}", "-t", f"{end - start}", "-i", video_path,
                "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-shortest", out_path]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    last = None
    try:
        for i, t in enumerate(out_times):
            # 출력 시각에 해당하는 원본 프레임까지 읽어온다
            while True:
                pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if last is not None and pos > t:
                    break
                ok, frame = cap.read()
                if not ok:
                    frame = last
                    break
                last = frame
            frame = last
            if frame is None:
                break

            cx = int(centers[i])
            x1 = max(0, min(W - crop_w, cx - crop_w // 2))
            crop = frame[:, x1:x1 + crop_w]
            crop = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            crop = _draw_overlay(crop, _active_subs(utts, t), label, font_sub, font_label)
            proc.stdin.write(crop.tobytes())

            if progress and i % 20 == 0:
                progress(i / n_out)
    finally:
        cap.release()
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 인코딩 실패: {err[:400]}")
    return out_path


def grab_frame(video_path: str, t: float) -> np.ndarray | None:
    """썸네일·카드 배경용 프레임 1장."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
