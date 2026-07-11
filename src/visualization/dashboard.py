from collections import deque

import cv2

from src.cognition.cognitive_state import CognitiveState, State
from src.cognition.fusion_state import FusionState
from src.perception.state_estimator import StateEstimator
from src.visualization.instrument_layout import render_instrument_hud
from src.visualization.instrument_theme import (
    STATE_COLORS,
    draw_vertical_tape,
    draw_waiting_state,
)

# Backward-compatible exports used by alerts and legacy callers.
STATE_BORDER_COLORS = STATE_COLORS


def distraction_score(signals: dict, estimator: StateEstimator) -> int:
    return estimator.distraction_score(signals)


def draw_state_border(frame, state: State | None, flash: bool = False, thickness: int = 10):
    del thickness
    from src.visualization.instrument_theme import draw_annunciator_strip

    draw_annunciator_strip(frame, state, flash=flash)
    if state is not None and flash:
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (1, 1), (width - 2, height - 2), STATE_COLORS[state], 2)
    return frame


def draw_sparkline(frame, values, origin, size, label, color=(0, 255, 180)):
    del color, values, origin, size, label
    return frame


def draw_distraction_meter(frame, score, origin, size):
    draw_vertical_tape(frame, origin, size, score, label="DRIFT")
    return frame


def draw_status_panel(frame, cognitive_state: CognitiveState, distraction: int):
    return frame


def render_dashboard(
    frame,
    cognitive_state: CognitiveState | None,
    ear_history: deque[float],
    estimator: StateEstimator,
    flash: bool = False,
    alert_message: str = "",
):
    render_instrument_hud(
        frame,
        cognitive_state,
        ear_history,
        estimator,
        flash=flash,
        alert_message=alert_message,
    )
    return frame


def draw_soft_score_bars(frame, fusion: FusionState, origin: tuple[int, int]):
    return frame


def draw_profile_match_bars(frame, fusion: FusionState, origin: tuple[int, int]):
    from src.visualization.hud_text import HUD_DIM, HUD_LABEL, draw_hud_text
    from src.visualization.timeline import PROFILE_COLORS

    x, y = origin
    width = 148
    draw_hud_text(frame, "MATCH", (x, y), size=10, color=HUD_LABEL, label=True)
    bar_top = y + 14
    bar_height = 10
    labels = ("N", "H", "S", "M")
    keys = ("neutral", "happy", "sad", "mad")
    slot_w = (width - 8) // len(labels)
    active = (fusion.profile_phase or "").lower()
    for index, (label, key) in enumerate(zip(labels, keys)):
        score = float(fusion.profile_scores.get(key, 0.0))
        slot_x = x + index * slot_w
        cv2.rectangle(
            frame,
            (slot_x, bar_top),
            (slot_x + slot_w - 4, bar_top + bar_height),
            HUD_DIM,
            1,
            cv2.LINE_AA,
        )
        fill_w = max(1, int((slot_w - 6) * max(0.0, min(1.0, score))))
        color = PROFILE_COLORS.get(key, (170, 165, 158))
        cv2.rectangle(
            frame,
            (slot_x + 1, bar_top + 1),
            (slot_x + 1 + fill_w, bar_top + bar_height - 1),
            color,
            -1,
            cv2.LINE_AA,
        )
        if active.startswith(key):
            cv2.rectangle(
                frame,
                (slot_x, bar_top - 2),
                (slot_x + slot_w - 4, bar_top + bar_height + 2),
                color,
                1,
                cv2.LINE_AA,
            )
        draw_hud_text(frame, label, (slot_x + 2, bar_top + bar_height + 4), size=9, color=HUD_LABEL)
    return frame


def render_fusion_dashboard(
    frame,
    fusion: FusionState | None,
    ear_history: deque[float],
    estimator: StateEstimator,
    flash: bool = False,
    alert_message: str = "",
    *,
    subtitle: str = "",
    fps: float | None = None,
):
    if fusion is None:
        draw_waiting_state(frame)
        return frame

    render_instrument_hud(
        frame,
        fusion.cognitive,
        ear_history,
        estimator,
        fusion=fusion,
        flash=flash,
        alert_message=alert_message,
        subtitle=subtitle,
        fps=fps,
    )
    return frame
