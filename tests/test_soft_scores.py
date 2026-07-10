import pytest

from src.cognition.cognitive_state import CognitiveState, State
from src.cognition.soft_scores import SoftScores, compute_soft_scores
from utils.emotion_profile import EmotionProfile


NEUTRAL_PROFILE = {
    "smile_score": 0.10,
    "cheek_raise": 0.10,
    "brow_raise": 0.10,
    "brow_furrow": 0.04,
    "brow_inner_pinch": 0.24,
    "mouth_open": 0.04,
    "ear": 0.30,
}


def cognitive_state(state=State.HIGH_ATTENTION, **signals):
    defaults = {
        "ear": 0.30,
        "blink_rate": 8.0,
        "gaze_x": 0.0,
        "gaze_y": 0.0,
    }
    defaults.update(signals)
    return CognitiveState(state=state, confidence=0.9, signals=defaults)


def emotion_signals(**overrides):
    defaults = dict(NEUTRAL_PROFILE)
    defaults.update({"smile_delta": 0.0, "cheek_delta": 0.0})
    defaults.update(overrides)
    return defaults


def test_soft_scores_as_dict_returns_all_score_fields():
    scores = SoftScores(engagement=0.1, fatigue=0.2, tension=0.3, positivity=0.4)

    assert scores.as_dict() == {
        "engagement": 0.1,
        "fatigue": 0.2,
        "tension": 0.3,
        "positivity": 0.4,
    }


def test_compute_soft_scores_rewards_attention_and_centered_gaze():
    scores = compute_soft_scores(
        cognitive_state(State.HIGH_ATTENTION),
        emotion_signals(),
        distraction_pct=0,
    )

    assert scores.engagement == pytest.approx(1.0)
    assert scores.fatigue == pytest.approx(0.157)
    assert scores.tension == pytest.approx(0.231)
    assert scores.positivity == pytest.approx(0.5)


def test_compute_soft_scores_detects_distracted_fatigued_tension_pattern():
    scores = compute_soft_scores(
        cognitive_state(
            State.DISTRACTED,
            ear=0.18,
            blink_rate=32.0,
            gaze_x=0.30,
            gaze_y=0.20,
        ),
        emotion_signals(brow_furrow=0.18, brow_inner_pinch=0.10),
        distraction_pct=85,
    )

    assert scores.engagement < 0.15
    assert scores.fatigue > 0.70
    assert scores.tension > 0.60
    assert 0.0 <= scores.positivity <= 1.0


def test_compute_soft_scores_applies_profile_phase_adjustments():
    profile = EmotionProfile(
        neutral=NEUTRAL_PROFILE,
        happy={**NEUTRAL_PROFILE, "smile_score": 0.35, "cheek_raise": 0.24},
        mad={**NEUTRAL_PROFILE, "brow_furrow": 0.20},
    )

    happy_scores = compute_soft_scores(
        cognitive_state(),
        emotion_signals(smile_score=0.36, cheek_raise=0.25),
        distraction_pct=0,
        profile=profile,
    )
    mad_scores = compute_soft_scores(
        cognitive_state(),
        emotion_signals(brow_furrow=0.21),
        distraction_pct=0,
        profile=profile,
    )

    assert happy_scores.positivity > 0.5
    assert mad_scores.tension > happy_scores.tension
