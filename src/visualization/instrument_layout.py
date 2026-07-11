"""Panel-free HUD layout — floating instruments over the live feed."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2

from src.cognition.cognitive_state import CognitiveState, State
from src.cognition.fusion_state import FusionState
from src.perception.state_estimator import StateEstimator
from src.visualization.hud_text import HUD_ACCENT, HUD_DIM, HUD_LABEL, draw_hud_text, text_width
from src.visualization.instrument_theme import (
    CAUTION,
    SAFE,
    WARN,
    draw_annunciator_strip,
    draw_gaze_compass,
    draw_vertical_tape,
    draw_waiting_state,
    gauge_color_for_value,
)


def _draw_glass_backplate(
    frame,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    alpha: float = 0.34,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (4, 12, 18), -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x, y), (x + width, y + height), HUD_DIM, 1, cv2.LINE_AA)
    roi = frame[y : y + height, x : x + width]
    source = overlay[y : y + height, x : x + width]
    cv2.addWeighted(source, alpha, roi, 1.0 - alpha, 0, roi)


@dataclass(frozen=True)
class LayoutMetrics:
    width: int
    height: int
    top: int = 34
    margin: int = 12
    left_w: int = 158
    right_w: int = 84

    @property
    def left_x(self) -> int:
        return self.margin

    @property
    def right_x(self) -> int:
        return self.width - self.right_w - self.margin

    @property
    def content_top(self) -> int:
        return self.top + 4

    @property
    def content_bottom(self) -> int:
        return self.height - self.margin


def metrics_for_frame(frame) -> LayoutMetrics:
    height, width = frame.shape[:2]
    return LayoutMetrics(width=width, height=height)


def draw_annunciator(
    frame,
    state: State | None,
    *,
    flash: bool = False,
    subtitle: str = "",
) -> None:
    draw_annunciator_strip(frame, state, flash=flash)
    if subtitle:
        tw = text_width(subtitle, size=11)
        draw_hud_text(frame, subtitle, (frame.shape[1] - tw - 12, 10), size=11, color=HUD_DIM)


def _draw_mini_gauge(
    frame,
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
    *,
    invert: bool = False,
) -> int:
    draw_hud_text(frame, label, (x, y), size=11, color=HUD_LABEL, label=True)
    track_left = x + 40
    track_right = x + width - 32
    track_y = y + 12
    cv2.line(frame, (track_left, track_y), (track_right, track_y), HUD_DIM, 1, cv2.LINE_AA)
    clamped = max(0.0, min(1.0, value))
    needle_x = int(track_left + (track_right - track_left) * clamped)
    accent = gauge_color_for_value(value, invert=invert)
    cv2.line(frame, (needle_x, track_y - 4), (needle_x, track_y + 4), accent, 2, cv2.LINE_AA)
    draw_hud_text(frame, f"{clamped:.0%}", (x + width - 30, y), size=11)
    return y + 20


def _draw_sparkline(
    frame,
    ear_history: deque[float],
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    _draw_glass_backplate(frame, x - 6, y - 4, width + 12, height + 10)
    draw_hud_text(frame, "EYE", (x, y), size=10, color=HUD_LABEL, label=True)
    plot_left = x
    plot_right = x + width
    plot_top = y + 14
    plot_bottom = y + height
    if len(ear_history) < 2:
        return
    min_value = min(ear_history)
    max_value = max(ear_history)
    span = max(max_value - min_value, 0.05)
    plot_width = plot_right - plot_left
    points = []
    for index, value in enumerate(ear_history):
        px = plot_left + int(index * plot_width / max(len(ear_history) - 1, 1))
        normalized = (value - min_value) / span
        py = plot_bottom - int(normalized * (plot_bottom - plot_top))
        points.append((px, py))
    for start, end in zip(points, points[1:]):
        cv2.line(frame, start, end, HUD_ACCENT, 2, cv2.LINE_AA)


def draw_left_stack(
    frame,
    layout: LayoutMetrics,
    cognitive: CognitiveState,
    distraction: int,
    ear_history: deque[float],
    fusion: FusionState | None = None,
) -> None:
    signals = cognitive.signals
    x = layout.left_x
    y = layout.content_top
    w = layout.left_w

    rows = [
        f"EAR {signals['ear']:.3f}  BLK {signals['blink_counter']:03d}",
        f"HDG Y{signals['head_yaw']:+.0f} P{signals['head_pitch']:+.0f}",
        f"GAZE {signals['gaze_direction'][:8].upper()}  DRIFT {distraction:02d}%",
    ]
    for row in rows:
        draw_hud_text(frame, row, (x, y), size=12)
        y += 16

    if fusion is not None:
        for label, value, invert in (
            ("ENG", fusion.soft.engagement, False),
            ("FAT", fusion.soft.fatigue, False),
            ("TEN", fusion.soft.tension, False),
            ("POS", fusion.soft.positivity, True),
        ):
            y = _draw_mini_gauge(frame, x, y + 2, w, label, value, invert=invert)

    spark_h = 36
    _draw_sparkline(frame, ear_history, x, layout.content_bottom - spark_h, w - 8, spark_h)


def draw_right_stack(
    frame,
    layout: LayoutMetrics,
    cognitive: CognitiveState,
    distraction: int,
    fusion: FusionState | None = None,
    *,
    fps: float | None = None,
) -> None:
    x = layout.right_x
    y = layout.content_top
    compass_size = 72
    tape_h = 72
    stack_h = compass_size + tape_h + 34
    _draw_glass_backplate(frame, x - 6, y - 4, layout.right_w + 10, stack_h + 8)
    draw_gaze_compass(
        frame,
        cognitive.signals["gaze_x"],
        cognitive.signals["gaze_y"],
        (x, y),
        radius=compass_size // 2,
    )
    draw_vertical_tape(frame, (x + 18, y + compass_size + 8), (44, tape_h), distraction, label="DRF")
    stat_y = y + compass_size + tape_h + 18
    stab = int(max(0.0, min(1.0, cognitive.confidence)) * 100)
    draw_hud_text(frame, f"STAB {stab:03d}", (x, stat_y), size=11)
    if fps is not None and fps > 0:
        draw_hud_text(frame, f"FPS {fps:4.0f}", (x, stat_y + 14), size=11, color=HUD_DIM)


def render_instrument_hud(
    frame,
    cognitive: CognitiveState | None,
    ear_history: deque[float],
    estimator: StateEstimator,
    *,
    fusion: FusionState | None = None,
    flash: bool = False,
    alert_message: str = "",
    subtitle: str = "",
    fps: float | None = None,
) -> None:
    if cognitive is None:
        draw_waiting_state(frame)
        return

    layout = metrics_for_frame(frame)
    distraction = estimator.distraction_score(cognitive.signals)
    draw_annunciator(frame, cognitive.state, flash=flash, subtitle=subtitle)
    draw_left_stack(frame, layout, cognitive, distraction, ear_history, fusion)
    draw_right_stack(frame, layout, cognitive, distraction, fusion, fps=fps)

    if alert_message:
        from src.visualization.alerts import draw_alert_banner

        draw_alert_banner(frame, alert_message, cognitive.state)
