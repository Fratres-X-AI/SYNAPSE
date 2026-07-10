from collections import deque

import cv2
import numpy as np

from src.cognition.cognitive_state import CognitiveState, State
from src.perception.state_estimator import StateEstimator

STATE_COLORS = {
    State.HIGH_ATTENTION: (80, 220, 100),
    State.MODERATE: (0, 220, 255),
    State.FATIGUED: (0, 140, 255),
    State.DISTRACTED: (60, 60, 255),
}

STATE_BORDER_COLORS = {
    State.HIGH_ATTENTION: (100, 255, 120),
    State.MODERATE: (0, 255, 255),
    State.FATIGUED: (0, 165, 255),
    State.DISTRACTED: (80, 80, 255),
}


def distraction_score(signals: dict, estimator: StateEstimator) -> int:
    return estimator.distraction_score(signals)


def draw_state_border(
    frame,
    state: State | None,
    flash: bool = False,
    thickness: int = 10,
):
    if state is None:
        color = (90, 90, 90)
    else:
        color = STATE_BORDER_COLORS[state]
        if flash:
            color = tuple(min(255, channel + 80) for channel in color)

    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, thickness)
    return frame


def draw_sparkline(
    frame,
    values: deque[float],
    origin: tuple[int, int],
    size: tuple[int, int],
    label: str,
    color: tuple[int, int, int] = (0, 255, 180),
):
    x, y = origin
    width, height = size
    panel = frame[y : y + height, x : x + width].copy()
    panel[:] = (20, 20, 20)

    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (60, 60, 60), 1)
    cv2.putText(panel, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    if len(values) >= 2:
        min_value = min(values)
        max_value = max(values)
        span = max(max_value - min_value, 0.05)
        plot_left = 8
        plot_right = width - 8
        plot_top = 24
        plot_bottom = height - 8
        plot_width = plot_right - plot_left

        points = []
        for index, value in enumerate(values):
            px = plot_left + int(index * plot_width / max(len(values) - 1, 1))
            normalized = (value - min_value) / span
            py = plot_bottom - int(normalized * (plot_bottom - plot_top))
            points.append((px, py))

        for start, end in zip(points, points[1:]):
            cv2.line(panel, start, end, color, 2)

    frame[y : y + height, x : x + width] = panel
    return frame


def draw_gaze_compass(
    frame,
    gaze_x: float,
    gaze_y: float,
    origin: tuple[int, int],
    radius: int = 48,
):
    x, y = origin
    center = (x + radius, y + radius)
    cv2.circle(frame, center, radius, (30, 30, 30), -1)
    cv2.circle(frame, center, radius, (80, 80, 80), 2)
    cv2.line(frame, (center[0] - radius + 8, center[1]), (center[0] + radius - 8, center[1]), (60, 60, 60), 1)
    cv2.line(frame, (center[0], center[1] - radius + 8), (center[0], center[1] + radius - 8), (60, 60, 60), 1)
    cv2.putText(frame, "GAZE", (center[0] - 22, center[1] - radius - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    dot_x = int(center[0] + np.clip(gaze_x, -0.5, 0.5) * (radius - 12) * 2)
    dot_y = int(center[1] + np.clip(gaze_y, -0.5, 0.5) * (radius - 12) * 2)
    cv2.circle(frame, (dot_x, dot_y), 6, (0, 0, 255), -1)
    return frame


def draw_distraction_meter(
    frame,
    score: int,
    origin: tuple[int, int],
    size: tuple[int, int],
):
    x, y = origin
    width, height = size
    panel = frame[y : y + height, x : x + width].copy()
    panel[:] = (20, 20, 20)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (60, 60, 60), 1)
    cv2.putText(panel, "DISTRACTION", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    bar_left = 12
    bar_right = width - 12
    bar_top = 30
    bar_bottom = height - 12
    cv2.rectangle(panel, (bar_left, bar_top), (bar_right, bar_bottom), (50, 50, 50), -1)

    fill_width = int((bar_right - bar_left) * score / 100)
    if fill_width > 0:
        color = (0, int(255 * (1 - score / 100)), int(255 * score / 100))
        cv2.rectangle(panel, (bar_left, bar_top), (bar_left + fill_width, bar_bottom), color, -1)

    cv2.putText(panel, f"{score}%", (bar_left, bar_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    frame[y : y + height, x : x + width] = panel
    return frame


def draw_status_panel(frame, cognitive_state: CognitiveState, distraction: int):
    signals = cognitive_state.signals
    color = STATE_COLORS[cognitive_state.state]
    lines = [
        "SYNAPSE DASHBOARD",
        f"State: {cognitive_state.state.value}",
        f"EAR: {signals['ear']:.3f} | Blinks: {signals['blink_counter']}",
        f"Yaw: {signals['head_yaw']:+.1f} Pitch: {signals['head_pitch']:+.1f}",
        f"Gaze: {signals['gaze_direction']}",
    ]

    y = 28
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        y += 26

    cv2.putText(
        frame,
        f"Distraction: {distraction}%",
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Distraction: {distraction}%",
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def render_dashboard(
    frame,
    cognitive_state: CognitiveState | None,
    ear_history: deque[float],
    estimator: StateEstimator,
    flash: bool = False,
    alert_message: str = "",
):
    if cognitive_state is None:
        draw_state_border(frame, None)
        cv2.putText(frame, "Waiting for face...", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        return frame

    distraction = distraction_score(cognitive_state.signals, estimator)
    height, width = frame.shape[:2]

    draw_state_border(frame, cognitive_state.state, flash=flash)
    draw_status_panel(frame, cognitive_state, distraction)
    draw_sparkline(frame, ear_history, (16, height - 110), (220, 90), "EAR (30s)")
    draw_gaze_compass(
        frame,
        cognitive_state.signals["gaze_x"],
        cognitive_state.signals["gaze_y"],
        (width - 130, 16),
    )
    draw_distraction_meter(frame, distraction, (width - 130, height - 150), (110, 130))

    if alert_message:
        from src.visualization.alerts import draw_alert_banner

        draw_alert_banner(frame, alert_message, cognitive_state.state)

    return frame
