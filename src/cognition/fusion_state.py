from dataclasses import dataclass
from typing import Any

from src.cognition.cognitive_state import CognitiveState
from src.cognition.emotion_state import Emotion, EmotionState
from src.cognition.soft_scores import SoftScores
from src.perception.state_estimator import StateEstimator


@dataclass(frozen=True)
class FusionState:
    cognitive: CognitiveState
    emotion: EmotionState
    soft: SoftScores
    reliable: dict[str, Any]
    distraction: int
    labeled_phase: str = ""
    profile_phase: str = ""
    profile_scores: dict[str, float] | None = None
    profile_confidence: float = 0.0

    @property
    def display_emotion(self) -> Emotion:
        if self.profile_phase and self.profile_confidence >= 0.45:
            from src.cognition.profile_matcher import emotion_from_phase

            mapped = emotion_from_phase(self.profile_phase)
            if mapped is not None:
                return mapped
        return self.emotion.emotion

    @classmethod
    def build(
        cls,
        cognitive: CognitiveState,
        emotion: EmotionState,
        soft: SoftScores,
        estimator: StateEstimator,
        labeled_phase: str = "",
        profile_phase: str = "",
        profile_scores: dict[str, float] | None = None,
        profile_confidence: float = 0.0,
    ) -> "FusionState":
        signals = cognitive.signals
        reliable = {
            "state": cognitive.state.value,
            "confidence": cognitive.confidence,
            "ear": signals["ear"],
            "blink_rate": signals["blink_rate"],
            "blink_count": signals["blink_counter"],
            "head_yaw": signals["head_yaw"],
            "head_pitch": signals["head_pitch"],
            "gaze_direction": signals["gaze_direction"],
            "gaze_x": signals["gaze_x"],
            "gaze_y": signals["gaze_y"],
            "is_blinking": signals["is_blinking"],
        }
        return cls(
            cognitive=cognitive,
            emotion=emotion,
            soft=soft,
            reliable=reliable,
            distraction=estimator.distraction_score(signals),
            labeled_phase=labeled_phase,
            profile_phase=profile_phase,
            profile_scores=profile_scores,
            profile_confidence=profile_confidence,
        )
