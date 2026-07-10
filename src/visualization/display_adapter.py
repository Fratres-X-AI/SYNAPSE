from abc import ABC, abstractmethod

import cv2

from src.cognition.cognitive_state import CognitiveState


class DisplayAdapter(ABC):
    """Output layer for Synapse. Swap OpenCV today for a wearable HUD tomorrow."""

    @abstractmethod
    def show(self, frame, cognitive_state: CognitiveState | None = None, autonomy: float | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_key(self, delay_ms: int = 1) -> int:
        raise NotImplementedError

    def poll_quit(self) -> bool:
        return self.read_key() == ord("q")

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class OpenCVDisplayAdapter(DisplayAdapter):
    def __init__(self, window_name: str) -> None:
        self.window_name = window_name

    def show(self, frame, cognitive_state: CognitiveState | None = None, autonomy: float | None = None) -> None:
        cv2.imshow(self.window_name, frame)

    def read_key(self, delay_ms: int = 1) -> int:
        return cv2.waitKey(delay_ms) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()


class HudStubDisplayAdapter(DisplayAdapter):
    """Development stand-in for a helmet-mounted HUD output channel."""

    def __init__(self, window_name: str = "Synapse HUD Stub") -> None:
        self.window_name = window_name
        self._last_payload = ""

    def show(self, frame, cognitive_state: CognitiveState | None = None, autonomy: float | None = None) -> None:
        if cognitive_state is None:
            payload = "HUD: waiting for face"
        else:
            signals = cognitive_state.signals
            payload = (
                f"HUD: state={cognitive_state.state.value} "
                f"autonomy={autonomy:.2f} "
                f"gaze={signals['gaze_direction']} "
                f"distraction_ready"
            )
        if payload != self._last_payload:
            print(payload)
            self._last_payload = payload
        cv2.imshow(self.window_name, frame)

    def read_key(self, delay_ms: int = 1) -> int:
        return cv2.waitKey(delay_ms) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()


def create_display_adapter(mode: str, window_name: str) -> DisplayAdapter:
    if mode == "hud_stub":
        return HudStubDisplayAdapter(window_name)
    return OpenCVDisplayAdapter(window_name)
