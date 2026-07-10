import cv2

from src.cognition.cognitive_state import CognitiveState


def render_status(frame, cognitive_state: CognitiveState | None, behavior: str | None):
    if cognitive_state is None:
        label = "State: waiting for face"
    else:
        signals = cognitive_state.signals
        label = (
            f"State: {cognitive_state.state.value} | "
            f"EAR: {signals['ear']:.3f} | "
            f"Blinks: {signals['blink_rate']:.1f}/min | "
            f"Yaw: {signals['head_yaw']:.1f} | "
            f"{behavior}"
        )

    cv2.putText(
        frame,
        label,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame
