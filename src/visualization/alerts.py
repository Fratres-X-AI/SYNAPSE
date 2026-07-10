import sys

import cv2

from src.cognition.cognitive_state import State
from src.visualization.hud_text import HUD_ACCENT, HUD_LABEL, draw_hud_text
from src.visualization.instrument_theme import CAUTION, STATE_COLORS, WARN

STATE_MESSAGES = {
    State.HIGH_ATTENTION: "Locked in - higher autonomy",
    State.MODERATE: "Neutral focus",
    State.FATIGUED: "Fatigue detected - easing up",
    State.DISTRACTED: "Distracted - autonomy reduced",
}

STATE_ALERT_COLORS = STATE_COLORS


class StateAlertTracker:
    def __init__(self, beep: bool = True, quiet: bool = False) -> None:
        self.beep = beep
        self.quiet = quiet
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
            if not self.quiet:
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
    accent = STATE_ALERT_COLORS.get(state, CAUTION) if state else CAUTION
    y = height // 2
    draw_hud_text(frame, message.upper(), (24, y - 8), size=14, color=HUD_LABEL, label=True)
    cv2.line(frame, (20, y + 12), (width - 20, y + 12), accent, 2, cv2.LINE_AA)
    if state == State.DISTRACTED:
        draw_hud_text(frame, "CAUTION", (width - 100, y - 8), size=12, color=WARN, label=True)
    return frame
