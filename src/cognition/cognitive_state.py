from dataclasses import dataclass
from enum import Enum
from typing import Any


class State(Enum):
    HIGH_ATTENTION = "high_attention"
    MODERATE = "moderate"
    FATIGUED = "fatigued"
    DISTRACTED = "distracted"


@dataclass(frozen=True)
class CognitiveState:
    state: State
    confidence: float
    signals: dict[str, Any]
