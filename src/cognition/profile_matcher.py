from typing import Any

import numpy as np

from src.cognition.emotion_state import Emotion
from utils.emotion_profile import EmotionProfile, PHASE_LABELS, PROFILE_KEYS

# EAR varies with blinks and attention; expression matching uses face geometry only.
MATCH_KEYS = tuple(key for key in PROFILE_KEYS if key != "ear")

FEATURE_WEIGHTS: dict[str, float] = {
    "mouth_open": 3.0,
    "smile_score": 2.0,
    "brow_inner_pinch": 2.0,
    "cheek_raise": 1.5,
    "brow_raise": 1.0,
    "brow_furrow": 1.0,
}

PHASE_TO_EMOTION = {
    "neutral": Emotion.NEUTRAL,
    "happy": Emotion.HAPPY,
    "sad": Emotion.SAD,
    "mad": Emotion.STRESSED,
}


def weighted_distance(current: dict[str, Any], reference: dict[str, float]) -> float:
    total = 0.0
    for key in MATCH_KEYS:
        if key not in reference:
            continue
        weight = FEATURE_WEIGHTS.get(key, 1.0)
        delta = float(current.get(key, 0.0)) - float(reference[key])
        total += weight * delta * delta
    return float(np.sqrt(total))


def _relative_signals(
    current: dict[str, Any],
    reference: dict[str, float],
    neutral_reference: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(current.get(key, 0.0)) - float(neutral_reference.get(key, 0.0))
        for key in MATCH_KEYS
        if key in reference and key in neutral_reference
    }


def match_profile(
    current: dict[str, Any],
    profile: EmotionProfile,
) -> tuple[str | None, dict[str, float], float]:
    phases = {
        phase: getattr(profile, phase)
        for phase in ("neutral", "happy", "sad", "mad")
        if getattr(profile, phase)
    }
    if not phases:
        return None, {}, 0.0

    neutral_reference = profile.neutral if profile.has_neutral() else None
    distances: dict[str, float] = {}
    for phase, reference in phases.items():
        if neutral_reference is not None:
            current_rel = _relative_signals(current, reference, neutral_reference)
            phase_rel = _relative_signals(reference, reference, neutral_reference)
            distances[phase] = weighted_distance(current_rel, phase_rel)
        else:
            distances[phase] = weighted_distance(current, reference)

    best_phase = min(distances, key=distances.get)
    min_dist = distances[best_phase]
    sorted_distances = sorted(distances.values())
    margin = sorted_distances[1] - sorted_distances[0] if len(sorted_distances) > 1 else 0.0

    # Resting face should stay neutral unless another phase is clearly closer.
    if (
        neutral_reference is not None
        and best_phase != "neutral"
        and "neutral" in distances
        and margin < 0.02
    ):
        best_phase = "neutral"
        min_dist = distances["neutral"]

    spread = max(distances.values()) - min(distances.values())
    confidence = float(np.clip(0.45 + spread * 18.0 - min_dist * 3.0, 0.35, 0.95))

    raw_scores = {phase: 1.0 / (0.02 + distance) for phase, distance in distances.items()}
    total = sum(raw_scores.values()) or 1.0
    scores = {phase: round(value / total, 3) for phase, value in raw_scores.items()}
    return best_phase, scores, confidence


def emotion_from_phase(phase: str | None) -> Emotion | None:
    if phase is None:
        return None
    return PHASE_TO_EMOTION.get(phase)


def phase_display_label(phase: str | None) -> str:
    if not phase:
        return "—"
    return PHASE_LABELS.get(phase, phase)
