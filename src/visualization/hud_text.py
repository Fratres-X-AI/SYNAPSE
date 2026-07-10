"""Antialiased HUD typography (PIL) with OpenCV fallback."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/Consolas.ttf"),
    Path("C:/Windows/Fonts/segoeuib.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/cour.ttf"),
)

_LABEL_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/segoeuib.ttf"),
    Path("C:/Windows/Fonts/Consolas.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
)


@lru_cache(maxsize=32)
def _load_font(size: int, *, label: bool = False) -> object | None:
    try:
        from PIL import ImageFont
    except ImportError:
        return None

    for path in (_LABEL_FONT_CANDIDATES if label else _FONT_CANDIDATES):
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return color[2], color[1], color[0]


HUD_INK = (252, 253, 255)
HUD_SHADOW = (16, 18, 22)
HUD_DIM = (188, 198, 208)
HUD_LABEL = (255, 224, 150)
HUD_ACCENT = (255, 200, 56)


def draw_hud_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    *,
    size: int = 12,
    color: tuple[int, int, int] = HUD_INK,
    label: bool = False,
) -> tuple[int, int]:
    draw_text(frame, text, (pos[0] + 1, pos[1] + 1), size=size, color=HUD_SHADOW, label=label)
    return draw_text(frame, text, pos, size=size, color=color, label=label)


def draw_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    *,
    size: int = 13,
    color: tuple[int, int, int] = (28, 32, 38),
    label: bool = False,
) -> tuple[int, int]:
    """Draw crisp text; returns (width, height) of rendered string."""
    if not text:
        return 0, 0

    font = _load_font(size, label=label)
    if font is None:
        scale = max(size / 24.0, 0.35)
        thickness = 1
        cv2.putText(
            frame,
            text,
            pos,
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        metrics = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)[0]
        return metrics[0], metrics[1]

    from PIL import Image, ImageDraw

    temp = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(temp)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 2
    label_img = Image.new("RGBA", (text_w + pad * 2, text_h + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(label_img).text(
        (pad - bbox[0], pad - bbox[1]),
        text,
        font=font,
        fill=(*_bgr_to_rgb(color), 255),
    )

    x, y = pos
    h, w = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + label_img.width)
    y2 = min(h, y + label_img.height)
    if x2 <= x1 or y2 <= y1:
        return text_w, text_h

    roi = frame[y1:y2, x1:x2]
    src = label_img.crop((x1 - x, y1 - y, x2 - x, y2 - y))
    src_rgb = np.array(src)
    alpha = src_rgb[:, :, 3:4].astype(np.float32) / 255.0
    rgb = src_rgb[:, :, :3][:, :, ::-1].astype(np.float32)
    blended = alpha * rgb + (1.0 - alpha) * roi.astype(np.float32)
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return text_w, text_h


def text_width(text: str, *, size: int = 13, label: bool = False) -> int:
    font = _load_font(size, label=label)
    if font is None:
        scale = max(size / 24.0, 0.35)
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)[0][0]
    from PIL import Image, ImageDraw

    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]
