from dataclasses import dataclass
from enum import Enum
from typing import Any


class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    STRESSED = "stressed"


@dataclass(frozen=True)
class EmotionState:
    emotion: Emotion
    confidence: float
    signals: dict[str, Any]
