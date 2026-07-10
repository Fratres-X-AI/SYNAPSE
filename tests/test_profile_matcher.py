import math

import pytest

from src.cognition.emotion_state import Emotion
from src.cognition.profile_matcher import (
    emotion_from_phase,
    match_profile,
    phase_display_label,
    weighted_distance,
)
from utils.emotion_profile import EmotionProfile


BASELINE = {
    "smile_score": 0.10,
    "cheek_raise": 0.10,
    "brow_raise": 0.10,
    "brow_furrow": 0.04,
    "brow_inner_pinch": 0.24,
    "mouth_open": 0.04,
    "ear": 0.30,
}


def test_weighted_distance_uses_expression_features_but_ignores_ear():
    current = {**BASELINE, "mouth_open": 0.14, "ear": 0.10}
    reference = {**BASELINE, "ear": 0.35}

    distance = weighted_distance(current, reference)

    assert distance == pytest.approx(math.sqrt(3.0 * 0.10**2))


def test_match_profile_prefers_relative_happy_expression():
    profile = EmotionProfile(
        neutral=BASELINE,
        happy={**BASELINE, "smile_score": 0.35, "cheek_raise": 0.22},
        sad={**BASELINE, "brow_inner_pinch": 0.12},
        mad={**BASELINE, "brow_furrow": 0.18},
    )
    current = {**BASELINE, "smile_score": 0.36, "cheek_raise": 0.23}

    phase, scores, confidence = match_profile(current, profile)

    assert phase == "happy"
    assert scores["happy"] == max(scores.values())
    assert 0.35 <= confidence <= 0.95


def test_match_profile_keeps_ambiguous_resting_face_neutral():
    profile = EmotionProfile(
        neutral=BASELINE,
        happy={**BASELINE, "smile_score": 0.101},
    )
    current = {**BASELINE, "smile_score": 0.101}

    phase, scores, confidence = match_profile(current, profile)

    assert phase == "neutral"
    assert set(scores) == {"neutral", "happy"}
    assert confidence >= 0.35


def test_match_profile_handles_empty_profile_and_phase_helpers():
    phase, scores, confidence = match_profile(BASELINE, EmotionProfile())

    assert phase is None
    assert scores == {}
    assert confidence == 0.0
    assert emotion_from_phase("mad") is Emotion.STRESSED
    assert emotion_from_phase(None) is None
    assert phase_display_label("sad") == "sad/stressed"
    assert phase_display_label(None) == "—"
