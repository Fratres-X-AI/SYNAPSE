from src.cognition.cognitive_state import CognitiveState, State


class AdaptiveAgent:
    """Closed-loop behavior controller that reacts to cognitive state."""

    def __init__(self) -> None:
        self.autonomy_level = 0.5

    def adapt(self, cognitive_state: CognitiveState) -> None:
        if cognitive_state.state == State.FATIGUED:
            self.autonomy_level = min(0.85, self.autonomy_level + 0.15)
        elif cognitive_state.state == State.HIGH_ATTENTION:
            self.autonomy_level = max(0.4, self.autonomy_level - 0.1)
        elif cognitive_state.state == State.DISTRACTED:
            self.autonomy_level = 0.25

    def get_behavior(self) -> str:
        return f"Autonomy Level: {self.autonomy_level:.2f}"
