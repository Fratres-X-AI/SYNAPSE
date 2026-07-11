from collections import deque

import numpy as np

from src.cognition.cognitive_state import CognitiveState, State
from src.cognition.emotion_state import Emotion, EmotionState
from src.cognition.fusion_state import FusionState
from src.cognition.soft_scores import SoftScores
from src.perception.state_estimator import StateEstimator
from src.visualization import instrument_layout


def _signals() -> dict:
    return {
        "ear": 0.31,
        "blink_counter": 4,
        "blink_rate": 12.0,
        "head_yaw": -8.0,
        "head_pitch": 3.0,
        "gaze_direction": "center",
        "gaze_x": 0.02,
        "gaze_y": -0.01,
        "is_blinking": False,
    }


def _fusion() -> FusionState:
    cognitive = CognitiveState(State.MODERATE, 0.82, _signals())
    emotion = EmotionState(Emotion.NEUTRAL, 0.7, {"mouth_open": 0.1})
    soft = SoftScores(0.62, 0.18, 0.22, 0.55)
    estimator = StateEstimator()
    return FusionState.build(
        cognitive,
        emotion,
        soft,
        estimator,
        profile_phase="neutral",
        profile_scores={"neutral": 0.9, "happy": 0.1, "sad": 0.0, "mad": 0.0},
        profile_confidence=0.9,
    )


def test_render_instrument_hud_draws_core_instruments_without_profile(monkeypatch):
    drawn: list[str] = []

    def capture(frame, text, pos, *, size=12, color=(0, 0, 0), label=False):
        drawn.append(text)
        return (len(text) * 6, size)

    monkeypatch.setattr(instrument_layout, "draw_hud_text", capture)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    fusion = _fusion()
    ear_history = deque([0.28, 0.29, 0.31, 0.30], maxlen=180)
    estimator = StateEstimator()

    instrument_layout.render_instrument_hud(
        frame,
        fusion.cognitive,
        ear_history,
        estimator,
        fusion=fusion,
        subtitle="478+37",
    )

    joined = " | ".join(drawn)
    assert "PROFILE" not in joined
    assert "neutral" not in joined.lower()
    assert "EYE" in drawn
    assert "ENG" in drawn
    assert any("DRIFT" in text for text in drawn)
    assert all(label not in drawn for label in ("N", "E", "S", "W"))


def test_render_instrument_hud_waiting_state(monkeypatch):
    called = {"waiting": False}

    def fake_waiting(frame):
        called["waiting"] = True

    monkeypatch.setattr(instrument_layout, "draw_waiting_state", fake_waiting)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    instrument_layout.render_instrument_hud(
        frame,
        None,
        deque(maxlen=180),
        StateEstimator(),
    )

    assert called["waiting"] is True
