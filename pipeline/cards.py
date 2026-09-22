"""
cards.py — 영상에서 카드뉴스를 만든다.

문제정의서 3-2 (F) 단계. '재활용 병목'을 푸는 부분.
같은 촬영본이 릴스 1개로 끝나지 않도록, 내가 실제로 한 말(STT 결과)에서
코칭 큐를 뽑아 인스타 1:1 카드 이미지로 렌더링한다.

요약은 규칙 기반 추출 요약이다. 생성형 모델로 새 문장을 지어내면
'내가 하지 않은 말'이 콘텐츠에 섞이는데, 전문가 계정에서 그건 치명적이다.
그래서 원문 문장을 그대로 고르고 다듬기만 한다.
"""

from __future__ import annotations

import os
import re
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .fonts import korean_font
from .highlight import Clip
from .stt import Utterance

SIZE = 1080

# 코칭 큐에 잘 등장하는 도메인 키워드
CUE_WORDS = [
    "무릎", "발목", "발", "고관절", "골반", "허리", "척추", "코어", "복압", "호흡",
    "어깨", "견갑", "팔꿈치", "손목", "목", "턱", "중심", "체중", "정렬", "각도",
    "가동범위", "스쿼트", "런지", "힌지", "데드리프트", "버피", "템포", "속도",
    "버티", "내려", "올라", "밀어", "당겨", "펴", "접어", "힘", "긴장", "이완",
    "통증", "뭉치", "풀어", "스트레칭", "마사지", "압박", "자세",
]

# 지시형 어미 — 코칭 멘트의 특징
IMPERATIVE = re.compile(r"(하세요|세요|보세요|합니다|해요|주세요|할게요|입니다|죠|니다)$")

STOP = re.compile(r"^(자|그럼|네|어|음|아|오케이|좋아요|자자)[\s,]*$")


def _sentences(utts: list[Utterance]) -> list[tuple[float, str]]:
    out = []
    for u in utts:
        for part in re.split(r"(?<=[.!?])\s+|\n", u.text):
            s = part.strip(" .,·")
            if len(s) >= 6 and not STOP.match(s):
                out.append((u.start, s))
    return out


def _score(sentence: str) -> float:
    s = 0.0
    for w in CUE_WORDS:
        if w in sentence:
            s += 1.0
    if IMPERATIVE.search(sentence):
        s += 1.5
    n = len(sentence)
    if 12 <= n <= 45:       # 카드에 얹기 좋은 길이
        s += 1.0
    elif n > 70:
        s -= 1.0
    return s


def _dedupe(cands):
    kept = []
    for sc, t, s in cands:
        if any(len(set(s) & set(k[2])) / max(len(set(s)), 1) > 0.65 for k in kept):
            continue
        kept.append((sc, t, s))
    return kept


def pick_cues(utts: list[Utterance], n: int = 3) -> list[dict]:
    """카드에 쓸 코칭 큐 문장을 고른다."""
    cands = [(_score(s), t, s) for t, s in _sentences(utts)]
    cands = [c for c in cands if c[0] > 0]
    cands.sort(key=lambda x: -x[0])
    picked = _dedupe(cands)[:n]
    picked.sort(key=lambda x: x[1])  # 영상 시간 순서대로
    return [{"time": round(t, 2), "text": s, "score": sc} for sc, t, s in picked]


# ------------------------------------------------------------------ 렌더링

BG = (18, 18, 20)
ACCENT = (222, 255, 94)
WHITE = (255, 255, 255)
GRAY = (150, 150, 155)


def _bg_from_frame(frame) -> Image.Image:
    """영상 프레임을 어둡게 깔아 표지 배경으로."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))
    img = img.resize((SIZE, SIZE), Image.LANCZOS).filter(ImageFilter.GaussianBlur(6))
    dark = Image.new("RGB", (SIZE, SIZE), BG)
    return Image.blend(img, dark, 0.62)


def _draw_wrapped(d, text, font, x, y, max_w, fill, line_gap=18):
    words, line, lines = list(text), "", []
    for ch in words:
        if d.textbbox((0, 0), line + ch, font=font)[2] > max_w and line:
            lines.append(line)
            line = ch
        else:
            line += ch
    if line:
        lines.append(line)
    for ln in lines:
        d.text((x, y), ln, font=font, fill=fill)
        y += font.size + line_gap
    return y


def _base_card(bg_frame=None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = _bg_from_frame(bg_frame) if bg_frame is not None else Image.new("RGB", (SIZE, SIZE), BG)
    return img, ImageDraw.Draw(img)


def make_cover(title: str, subtitle: str, frame, index_text: str) -> Image.Image:
    img, d = _base_card(frame)
    d.rectangle([0, SIZE - 14, SIZE, SIZE], fill=ACCENT)
    f_tag = korean_font(34)
    f_title = korean_font(86, "black")
    f_sub = korean_font(40)

    d.rounded_rectangle([72, 78, 72 + d.textbbox((0, 0), index_text, font=f_tag)[2] + 48, 148],
                        radius=999, fill=ACCENT)
    d.text((96, 96), index_text, font=f_tag, fill=(18, 18, 20))

    y = 360
    y = _draw_wrapped(d, title, f_title, 72, y, SIZE - 144, WHITE, 14)
    _draw_wrapped(d, subtitle, f_sub, 72, y + 36, SIZE - 144, GRAY, 10)
    return img


def make_cue_card(no: int, total: int, text: str, caption: str) -> Image.Image:
    img, d = _base_card()
    f_no = korean_font(150, "black")
    f_text = korean_font(62)
    f_cap = korean_font(34)

    d.text((72, 96), f"{no:02d}", font=f_no, fill=ACCENT)
    d.text((72 + d.textbbox((0, 0), f'{no:02d}', font=f_no)[2] + 24, 190),
           f"/ {total:02d}", font=f_cap, fill=GRAY)

    d.line([72, 330, SIZE - 72, 330], fill=(60, 60, 64), width=3)
    y = _draw_wrapped(d, text, f_text, 72, 400, SIZE - 144, WHITE, 20)
    if caption:
        _draw_wrapped(d, caption, f_cap, 72, max(y + 40, SIZE - 220), SIZE - 144, GRAY, 8)
    return img


def make_outro(handle: str, lines: list[str]) -> Image.Image:
    img, d = _base_card()
    f_h = korean_font(64, "black")
    f_l = korean_font(38)
    d.text((72, 300), "저장해두고", font=f_h, fill=GRAY)
    d.text((72, 300 + 90), "운동할 때 꺼내 보세요", font=f_h, fill=WHITE)
    y = 560
    for ln in lines:
        d.text((72, y), f"· {ln}", font=f_l, fill=GRAY)
        y += 62
    d.rounded_rectangle([72, SIZE - 190, SIZE - 72, SIZE - 90], radius=20, fill=ACCENT)
    d.text((104, SIZE - 165), handle, font=korean_font(44, "black"), fill=(18, 18, 20))
    return img


def build_cards(
    out_dir: str,
    clip: Clip,
    cues: list[dict],
    frame,
    handle: str = "@my_movement",
) -> list[str]:
    """카드뉴스 세트를 png로 저장하고 경로 리스트를 반환."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    if cues:
        title = cues[0]["text"][:28]
        subtitle = f"{clip.exercise} · 영상에서 자동 추출한 코칭 포인트"
    else:
        title = f"{clip.exercise} {clip.reps}회"
        subtitle = "동작 구간을 자동으로 찾아 정리했습니다"

    cards = [make_cover(title, subtitle, frame, "오늘의 운동 노트")]
    body = cues[1:] if len(cues) > 1 else cues
    for i, cue in enumerate(body, 1):
        cards.append(make_cue_card(i, max(len(body), 1), cue["text"],
                                   f"영상 {int(cue['time'] // 60)}분 {int(cue['time'] % 60)}초 지점"))
    if not body:
        cards.append(make_cue_card(1, 1, f"{clip.exercise} · 반복 {clip.reps}회",
                                   "발화가 없어 동작 분석 결과로 대체했습니다"))
    cards.append(make_outro(handle, [
        f"{clip.exercise}",
        f"반복 {clip.reps}회 / {clip.duration:.0f}초",
        "자세한 설명은 릴스에서",
    ]))

    for i, img in enumerate(cards):
        p = os.path.join(out_dir, f"card_{i + 1:02d}.png")
        img.save(p, quality=95)
        paths.append(p)
    return paths


def build_caption(clip: Clip, cues: list[dict]) -> str:
    """인스타 캡션 초안 + 해시태그."""
    head = f"{clip.exercise} — 반복 {clip.reps}회 구간"
    body = "\n".join(f"· {c['text']}" for c in cues) if cues else "· 동작 구간 자동 추출"
    tags = ["#크로스핏", "#하이록스", "#기능성운동", "#골프피트니스", "#통증케어",
            "#마사지", "#운동루틴", "#홈트", "#personaltraining", "#movement"]
    return f"{head}\n\n{body}\n\n{' '.join(tags)}"
