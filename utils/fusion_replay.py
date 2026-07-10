import time
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.cognition.cognitive_state import CognitiveState, State
from src.cognition.emotion_state import Emotion, EmotionState
from src.cognition.fusion_state import FusionState
from src.cognition.profile_matcher import emotion_from_phase
from src.cognition.soft_scores import SoftScores
from src.perception.state_estimator import StateEstimator
from src.visualization.dashboard import render_fusion_dashboard
from src.visualization.display_adapter import create_display_adapter
from src.visualization.timeline import draw_timeline_bar, load_alert_log, load_session_rows
from utils.config import Config

FRAME_SIZE = (960, 540)
PLAYBACK_FPS = 24
TIMELINE_HEIGHT = 72
TIMELINE_MODES = ["state", "profile", "engagement"]

PHASE_COLORS = {
    "neutral": (200, 200, 200),
    "happy": (0, 220, 0),
    "sad": (255, 120, 0),
    "mad": (0, 0, 255),
}


def alert_log_path_for(session_path: Path) -> Path:
    return session_path.parent / f"{session_path.stem}.alerts.csv"


def row_to_fusion(row: dict, estimator: StateEstimator) -> FusionState:
    state = State(row["state"])
    signals = {
        "ear": float(row["ear"]),
        "mean_ear": float(row["ear"]),
        "is_blinking": False,
        "blink_rate": float(row["blink_rate"]),
        "blink_counter": int(row["blink_count"]),
        "head_yaw": float(row["head_yaw"]),
        "head_pitch": float(row["head_pitch"]),
        "gaze_direction": row["gaze_direction"],
        "gaze_x": float(row["gaze_x"]),
        "gaze_y": float(row["gaze_y"]),
    }
    cognitive = CognitiveState(state=state, confidence=float(row["confidence"]), signals=signals)
    emotion = EmotionState(
        emotion=Emotion(row["emotion"]),
        confidence=float(row["emotion_confidence"]),
        signals={
            "smile_score": float(row["smile_score"]),
            "smile_delta": float(row["smile_delta"]),
            "cheek_raise": float(row["cheek_raise"]),
            "cheek_delta": float(row["cheek_delta"]),
            "brow_raise": float(row["brow_raise"]),
            "brow_furrow": float(row["brow_furrow"]),
            "brow_inner_pinch": float(row["brow_inner_pinch"]),
            "mouth_open": float(row["mouth_open"]),
            "ear": float(row["ear"]),
        },
    )
    soft = SoftScores(
        engagement=float(row["engagement"]),
        fatigue=float(row["fatigue"]),
        tension=float(row["tension"]),
        positivity=float(row["positivity"]),
    )
    profile_scores = {
        "neutral": float(row.get("profile_neutral") or 0.0),
        "happy": float(row.get("profile_happy") or 0.0),
        "sad": float(row.get("profile_sad") or 0.0),
        "mad": float(row.get("profile_mad") or 0.0),
    }
    return FusionState.build(
        cognitive,
        emotion,
        soft,
        estimator,
        labeled_phase=row.get("labeled_phase", ""),
        profile_phase=row.get("profile_phase", ""),
        profile_scores=profile_scores,
        profile_confidence=float(row.get("profile_confidence") or 0.0),
    )


def row_to_fusion_monitor(row: dict, estimator: StateEstimator) -> FusionState:
    state = State(row["state"])
    signals = {
        "ear": float(row["ear"]),
        "mean_ear": float(row["ear"]),
        "is_blinking": False,
        "blink_rate": float(row["blink_rate"]),
        "blink_counter": int(row["blink_count"]),
        "head_yaw": float(row["head_yaw"]),
        "head_pitch": float(row["head_pitch"]),
        "gaze_direction": row["gaze_direction"],
        "gaze_x": float(row["gaze_x"]),
        "gaze_y": float(row["gaze_y"]),
    }
    cognitive = CognitiveState(state=state, confidence=float(row["confidence"]), signals=signals)
    profile_phase = row.get("profile_phase", "")
    mapped = emotion_from_phase(profile_phase) if profile_phase else None
    emotion = EmotionState(
        emotion=mapped or Emotion.NEUTRAL,
        confidence=float(row.get("profile_confidence") or 0.0),
        signals={
            "smile_score": 0.0,
            "smile_delta": 0.0,
            "cheek_raise": 0.0,
            "cheek_delta": 0.0,
            "brow_raise": 0.0,
            "brow_furrow": 0.0,
            "brow_inner_pinch": 0.0,
            "mouth_open": 0.0,
            "ear": float(row["ear"]),
        },
    )
    soft = SoftScores(
        engagement=float(row["engagement"]),
        fatigue=float(row["fatigue"]),
        tension=float(row["tension"]),
        positivity=float(row["positivity"]),
    )
    profile_scores = {
        "neutral": float(row.get("profile_neutral") or 0.0),
        "happy": float(row.get("profile_happy") or 0.0),
        "sad": float(row.get("profile_sad") or 0.0),
        "mad": float(row.get("profile_mad") or 0.0),
    }
    return FusionState.build(
        cognitive,
        emotion,
        soft,
        estimator,
        profile_phase=profile_phase,
        profile_scores=profile_scores,
        profile_confidence=float(row.get("profile_confidence") or 0.0),
    )


def _overlay_lines(
    row: dict,
    title: str,
    index: int,
    total: int,
    paused: bool,
    *,
    has_labeled_phase: bool = False,
    has_active_alerts: bool = False,
) -> list[str]:
    status = "PAUSED" if paused else "PLAYING"
    lines = [f"{title} {index + 1}/{total} | {status}"]
    if has_labeled_phase:
        lines.append(
            f"Label: {row.get('labeled_phase') or '—'} | Match: {row.get('profile_phase') or '—'}"
        )
    elif has_active_alerts:
        lines.append(
            f"Profile: {row.get('profile_phase') or '—'} | Alerts: {row.get('active_alerts') or '—'}"
        )
    else:
        lines.append(f"Profile: {row.get('profile_phase') or '—'}")
    lines.append(
        f"Eng {float(row['engagement']):.0%} | Ten {float(row['tension']):.0%}"
    )
    return lines


def draw_replay_overlay(
    frame,
    row: dict,
    index: int,
    total: int,
    paused: bool,
    *,
    title: str,
    timeline_height: int = TIMELINE_HEIGHT,
    has_labeled_phase: bool = False,
    has_active_alerts: bool = False,
):
    y = frame.shape[0] - timeline_height - 90
    for line in _overlay_lines(
        row,
        title,
        index,
        total,
        paused,
        has_labeled_phase=has_labeled_phase,
        has_active_alerts=has_active_alerts,
    ):
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)
        y += 24
    return frame


def draw_replay_timeline(
    frame,
    rows: list[dict],
    index: int,
    mode: str,
    alert_rows: list[dict] | None = None,
    timeline_height: int = TIMELINE_HEIGHT,
):
    return draw_timeline_bar(
        frame,
        rows,
        index,
        (0, frame.shape[0] - timeline_height),
        (frame.shape[1], timeline_height),
        mode=mode,
        alert_rows=alert_rows,
    )


def replay_fusion_rows(
    rows: list[dict],
    row_builder: Callable[[dict, StateEstimator], FusionState],
    *,
    title: str,
    window_title: str,
    session_path: Path | None = None,
    fullscreen: bool = False,
) -> None:
    if not rows:
        return

    config = Config()
    estimator = StateEstimator(**config.estimator_kwargs())
    display = create_display_adapter(config.display_mode, window_title, fullscreen=fullscreen)
    ear_history: deque[float] = deque(maxlen=180)
    index = 0
    paused = False
    timeline_mode = TIMELINE_MODES[0]
    timeline_mode_index = 0
    last_step = time.perf_counter()

    has_labeled_phase = "labeled_phase" in rows[0]
    has_active_alerts = "active_alerts" in rows[0]

    alert_rows: list[dict] = []
    alert_path: Path | None = None
    if session_path is not None:
        alert_path = alert_log_path_for(session_path)
        alert_rows = load_alert_log(alert_path)

    print("\nReplay controls: SPACE pause/resume | LEFT/RIGHT step | t timeline mode | f fullscreen | q quit")
    if alert_rows and alert_path is not None:
        print(f"Loaded {len(alert_rows)} alert marker(s) from {alert_path.name}")

    while True:
        frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        row = rows[index]
        fusion = row_builder(row, estimator)
        ear_history.append(float(row["ear"]))

        border_label = row.get("labeled_phase") or row.get("profile_phase", "")
        color = PHASE_COLORS.get(border_label, (90, 90, 90))
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 8)

        frame = render_fusion_dashboard(frame, fusion, ear_history, estimator)
        frame = draw_replay_overlay(
            frame,
            row,
            index,
            len(rows),
            paused,
            title=title,
            has_labeled_phase=has_labeled_phase,
            has_active_alerts=has_active_alerts,
        )
        frame = draw_replay_timeline(frame, rows, index, timeline_mode, alert_rows)
        display.show(frame)

        key = display.read_key(1 if not paused else 30)
        if display.try_handle_display_key(key):
            break
        if key == ord(" "):
            paused = not paused
        if key == 81 and index > 0:
            index -= 1
        if key == 83 and index < len(rows) - 1:
            index += 1
        if key in (ord("t"), ord("T")):
            timeline_mode_index = (timeline_mode_index + 1) % len(TIMELINE_MODES)
            timeline_mode = TIMELINE_MODES[timeline_mode_index]

        now = time.perf_counter()
        if not paused and now - last_step >= 1.0 / PLAYBACK_FPS:
            index = min(index + 1, len(rows) - 1)
            last_step = now

    display.close()


def load_rows(path: Path) -> list[dict]:
    return load_session_rows(path)
