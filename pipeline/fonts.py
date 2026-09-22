"""fonts.py — 한글이 실제로 그려지는 폰트를 찾아준다."""

from __future__ import annotations

import functools
import glob

from PIL import ImageFont

CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
    "C:/Windows/Fonts/malgunbd.ttf",  # Windows
]


@functools.lru_cache(maxsize=4)
def _resolve(weight: str = "bold") -> tuple[str, int]:
    """(폰트경로, ttc 인덱스) 반환. 한글 글리프가 있는 것만 고른다."""
    paths = [p for p in CANDIDATES if glob.glob(p)]
    paths += sorted(glob.glob("/usr/share/fonts/**/*Nanum*.ttf", recursive=True))
    if weight == "black":
        paths.sort(key=lambda p: 0 if "Black" in p else 1)

    for path in paths:
        for idx in range(0, 8):
            try:
                f = ImageFont.truetype(path, 40, index=idx)
            except Exception:
                break
            name = " ".join(f.getname())
            if not path.endswith(".ttc") or "KR" in name or "Korean" in name:
                # 한글이 실제로 그려지는지 확인 (폭이 0이면 글리프 없음)
                if f.getbbox("한글")[2] > 0:
                    return path, idx
    raise RuntimeError("한글 폰트를 찾지 못했습니다. NanumGothic 등을 설치하세요.")


def korean_font(size: int, weight: str = "bold") -> ImageFont.FreeTypeFont:
    path, idx = _resolve(weight)
    return ImageFont.truetype(path, size, index=idx)
