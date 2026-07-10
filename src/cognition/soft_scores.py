from dataclasses import dataclass

import numpy as np

from src.cognition.cognitive_state import CognitiveState, State
from src.cognition.profile_matcher import match_profile
from utils.emotion_profile import EmotionProfile, PROFILE_KEYS


@dataclass(frozen=True)
class SoftScores:
    engagement: float
    fatigue: float
    tension: float
    positivity: float

    def as_dict(self) -> dict[str, float]:
        return {
            "engagement": self.engagement,
            "fatigue": self.fatigue,
            "tension": self.tension,
            "positivity": self.positivity,
        }


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def compute_soft_scores(
    cognitive: CognitiveState,
    emotion_signals: dict,
    distraction_pct: int,
    profile: EmotionProfile | None = None,
) -> SoftScores:
    signals = cognitive.signals
    ear = float(signals.get("ear") or 0.25)
    blink_rate = float(signals.get("blink_rate") or 0.0)
    gaze_x = abs(float(signals.get("gaze_x") or 0.0))
    gaze_y = abs(float(signals.get("gaze_y") or 0.0))

    attention_weight = {
        State.HIGH_ATTENTION: 1.0,
        State.MODERATE: 0.72,
        State.FATIGUED: 0.45,
        State.DISTRACTED: 0.25,
    }[cognitive.state]
    gaze_center = 1.0 - _clip01((gaze_x + gaze_y) / 0.45)
    engagement = _clip01(
        0.45 * (1.0 - distraction_pct / 100.0)
        + 0.30 * gaze_center
        + 0.25 * attention_weight
    )

    fatigue = _clip01(0.55 * _clip01(blink_rate / 28.0) + 0.45 * _clip01(1.0 - ear / 0.30))

    brow_furrow = max(0.0, float(emotion_signals.get("brow_furrow") or 0.0))
    brow_pinch = _clip01(0.52 - float(emotion_signals.get("brow_inner_pinch") or 0.24))
    ear_tension = _clip01(1.0 - ear / 0.24)
    tension = _clip01(0.40 * _clip01(brow_furrow / 0.12) + 0.35 * brow_pinch + 0.25 * ear_tension)

    smile_delta = float(emotion_signals.get("smile_delta") or 0.0)
    cheek_delta = float(emotion_signals.get("cheek_delta") or 0.0)
    positivity = _clip01(0.50 + smile_delta * 12.0 + cheek_delta * 8.0)

    if profile and profile.has_neutral():
        current = {key: float(emotion_signals.get(key) or 0.0) for key in PROFILE_KEYS}
        matched_phase, phase_scores, match_conf = match_profile(current, profile)
        if matched_phase == "happy":
            positivity = _clip01(positivity + 0.12 * match_conf)
        elif matched_phase == "sad":
            positivity = _clip01(positivity - 0.12 * match_conf)
            tension = _clip01(tension + 0.10 * match_conf)
        elif matched_phase == "mad":
            tension = _clip01(tension + 0.15 * match_conf)
        elif matched_phase == "neutral":
            tension = _clip01(tension - 0.08 * match_conf)
            positivity = _clip01(positivity + 0.05 * match_conf)
        if phase_scores:
            engagement = _clip01(engagement + 0.05 * phase_scores.get("neutral", 0.0))

    return SoftScores(
        engagement=round(engagement, 3),
        fatigue=round(fatigue, 3),
        tension=round(tension, 3),
        positivity=round(positivity, 3),
    )
