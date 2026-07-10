import sys

import cv2

from src.cognition.cognitive_state import State

STATE_MESSAGES = {
    State.HIGH_ATTENTION: "Locked in - higher autonomy",
    State.MODERATE: "Neutral focus",
    State.FATIGUED: "Fatigue detected - easing up",
    State.DISTRACTED: "Distracted - autonomy reduced",
}

STATE_ALERT_COLORS = {
    State.HIGH_ATTENTION: (80, 220, 100),
    State.MODERATE: (0, 220, 255),
    State.FATIGUED: (0, 140, 255),
    State.DISTRACTED: (80, 80, 255),
}


class StateAlertTracker:
    def __init__(self, beep: bool = True) -> None:
        self.beep = beep
        self.previous_state: State | None = None
        self.flash_frames = 0
        self.alert_message = ""
        self.alert_frames = 0
        self.alert_state: State | None = None

    def update(self, state: State | None, autonomy: float | None = None) -> tuple[bool, str]:
        if state is not None and self.previous_state is not None and state != self.previous_state:
            self.flash_frames = 14
            self.alert_frames = 45
            self.alert_state = state
            self.alert_message = self._message(state, autonomy)
            if self.beep:
                self._play_beep(state)
            print(f"\nALERT: {self.alert_message}")

        if state is not None:
            self.previous_state = state

        flash = self.flash_frames > 0
        if self.flash_frames > 0:
            self.flash_frames -= 1

        message = self.alert_message if self.alert_frames > 0 else ""
        if self.alert_frames > 0:
            self.alert_frames -= 1
        elif self.alert_message:
            self.alert_message = ""

        return flash, message

    def _message(self, state: State, autonomy: float | None) -> str:
        base = STATE_MESSAGES[state]
        if autonomy is not None:
            return f"{base} ({autonomy:.2f})"
        return base

    def _play_beep(self, state: State) -> None:
        if sys.platform != "win32":
            return
        try:
            import winsound

            tones = {
                State.HIGH_ATTENTION: (900, 120),
                State.MODERATE: (700, 100),
                State.FATIGUED: (500, 160),
                State.DISTRACTED: (350, 220),
            }
            frequency, duration = tones[state]
            winsound.Beep(frequency, duration)
        except RuntimeError:
            pass


def draw_alert_banner(frame, message: str, state: State | None = None):
    if not message:
        return frame

    height, width = frame.shape[:2]
    color = STATE_ALERT_COLORS.get(state, (255, 255, 255)) if state else (255, 255, 255)
    banner_height = 54
    y1 = height // 2 - banner_height // 2
    y2 = y1 + banner_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (width, y2), (10, 10, 10), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

    cv2.putText(
        frame,
        message,
        (24, y1 + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (24, y1 + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame
