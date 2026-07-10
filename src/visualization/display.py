import cv2

from src.cognition.cognitive_state import CognitiveState, State

STATE_COLORS = {
    State.HIGH_ATTENTION: (80, 220, 100),
    State.MODERATE: (0, 220, 255),
    State.FATIGUED: (0, 140, 255),
    State.DISTRACTED: (60, 60, 255),
}


def render_status(frame, cognitive_state: CognitiveState | None, behavior: str | None):
    if cognitive_state is None:
        lines = ["SYNAPSE", "Waiting for face..."]
        color = (180, 180, 180)
    else:
        signals = cognitive_state.signals
        blink_label = "BLINKING" if signals["is_blinking"] else "eyes open"
        color = STATE_COLORS.get(cognitive_state.state, (0, 255, 0))
        lines = [
            "SYNAPSE CLOSED LOOP",
            f"State: {cognitive_state.state.value} ({cognitive_state.confidence:.0%})",
            f"EAR: {signals['ear']:.3f} | Mean: {signals['mean_ear']:.3f} | {blink_label}",
            f"Blinks: {signals['blink_counter']} ({signals['blink_rate']:.1f}/min)",
            f"Yaw: {signals['head_yaw']:+.1f} | Pitch: {signals['head_pitch']:+.1f}",
            f"Gaze: {signals['gaze_direction']} ({signals['gaze_x']:+.2f}, {signals['gaze_y']:+.2f})",
            behavior or "",
        ]

    y = 28
    for line in lines:
        if not line:
            continue
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 26

    return frame
