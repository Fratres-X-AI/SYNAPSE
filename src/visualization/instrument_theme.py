"""Glass-cockpit HUD primitives — no panels, wire instruments only."""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.cognition.cognitive_state import State
from src.visualization.hud_text import HUD_ACCENT, HUD_DIM, HUD_INK, HUD_LABEL, draw_hud_text

ACCENT = HUD_ACCENT
TEXT = HUD_INK
TEXT_MUTED = HUD_DIM
TEXT_LABEL = HUD_LABEL
GRID = HUD_DIM
FACE = HUD_INK
FACE_DIM = HUD_DIM
BEZEL = HUD_DIM
NEEDLE = HUD_ACCENT
TRACE = HUD_ACCENT
SAFE = (92, 188, 76)
CAUTION = (64, 176, 255)
WARN = (72, 80, 228)

STATE_COLORS = {
    State.HIGH_ATTENTION: SAFE,
    State.MODERATE: HUD_ACCENT,
    State.FATIGUED: (255, 126, 48),
    State.DISTRACTED: WARN,
}

STATE_LABELS = {
    State.HIGH_ATTENTION: "FOCUS",
    State.MODERATE: "NOMINAL",
    State.FATIGUED: "FATIGUE",
    State.DISTRACTED: "DRIFT",
}


def draw_panel(*args, **kwargs):
    """Panels disabled — layout draws directly on the feed."""
    if args:
        return args[1][0], args[1][1], args[2][0], args[2][1]
    return 0, 0, 0, 0


def draw_tick_marks(*args, **kwargs):
    return None


def draw_vertical_tape(
    frame: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    value_pct: int,
    *,
    label: str = "",
    embedded: bool = False,
    compact: bool = False,
) -> None:
    del embedded
    x, y = origin
    width, height = size
    if label:
        draw_hud_text(frame, label, (x, y), size=10, color=HUD_LABEL, label=True)
    track_top = y + 12
    track_bottom = y + height - 4
    cx = x + width // 2
    cv2.line(frame, (cx, track_top), (cx, track_bottom), HUD_DIM, 1, cv2.LINE_AA)
    for tick in range(5):
        ty = int(track_bottom - (track_bottom - track_top) * tick / 4)
        cv2.line(frame, (cx - 4, ty), (cx + 4, ty), HUD_DIM, 1, cv2.LINE_AA)

    clamped = int(np.clip(value_pct, 0, 100))
    fill_top = int(track_bottom - (track_bottom - track_top) * clamped / 100)
    color = SAFE if clamped < 45 else CAUTION if clamped < 70 else WARN
    cv2.line(frame, (cx, fill_top), (cx, track_bottom), color, 3, cv2.LINE_AA)
    if not compact:
        draw_hud_text(frame, str(clamped), (x, track_bottom + 2), size=10)


def draw_gaze_compass(
    frame: np.ndarray,
    gaze_x: float,
    gaze_y: float,
    origin: tuple[int, int],
    *,
    radius: int = 44,
) -> None:
    x, y = origin
    center = (x + radius, y + radius)
    cv2.circle(frame, center, radius, HUD_DIM, 1, cv2.LINE_AA)
    cv2.line(
        frame,
        (center[0] - radius + 6, center[1]),
        (center[0] + radius - 6, center[1]),
        HUD_DIM,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (center[0], center[1] - radius + 6),
        (center[0], center[1] + radius - 6),
        HUD_DIM,
        1,
        cv2.LINE_AA,
    )
    for angle_deg in (0, 90, 180, 270):
        angle = math.radians(angle_deg)
        outer = (
            int(center[0] + math.sin(angle) * (radius - 2)),
            int(center[1] - math.cos(angle) * (radius - 2)),
        )
        inner = (
            int(center[0] + math.sin(angle) * (radius - 8)),
            int(center[1] - math.cos(angle) * (radius - 8)),
        )
        cv2.line(frame, inner, outer, HUD_DIM, 1, cv2.LINE_AA)

    dot_x = int(center[0] + np.clip(gaze_x, -0.5, 0.5) * (radius - 10) * 2)
    dot_y = int(center[1] + np.clip(gaze_y, -0.5, 0.5) * (radius - 10) * 2)
    cv2.circle(frame, (dot_x, dot_y), 3, HUD_ACCENT, -1, cv2.LINE_AA)


def draw_annunciator_strip(
    frame: np.ndarray,
    state: State | None,
    *,
    flash: bool = False,
) -> None:
    width = frame.shape[1]
    labels = list(State)
    segment_w = 86
    gap = 18
    total_w = len(labels) * segment_w + (len(labels) - 1) * gap
    origin_x = max(12, (width - total_w) // 2)
    y = 8

    for index, candidate in enumerate(labels):
        x = origin_x + index * (segment_w + gap)
        active = state == candidate
        accent = STATE_COLORS[candidate]
        text_color = HUD_INK if active else accent
        draw_hud_text(
            frame,
            STATE_LABELS[candidate],
            (x + 8, y),
            size=11,
            color=text_color,
            label=active,
        )
        underline_y = y + 18
        thickness = 3 if active else 2
        if active and flash:
            thickness = 4
        cv2.line(
            frame,
            (x, underline_y),
            (x + segment_w, underline_y),
            accent,
            thickness,
            cv2.LINE_AA,
        )


def draw_waiting_state(frame: np.ndarray) -> None:
    draw_annunciator_strip(frame, None)
    draw_hud_text(frame, "ACQUIRING FACE", (16, 40), size=13, color=HUD_LABEL, label=True)
    draw_hud_text(frame, "Center in frame", (16, 58), size=12, color=HUD_DIM)


def gauge_color_for_value(value: float, *, invert: bool = False) -> tuple[int, int, int]:
    score = 1.0 - value if invert else value
    if score < 0.45:
        return SAFE
    if score < 0.7:
        return CAUTION
    return WARN
