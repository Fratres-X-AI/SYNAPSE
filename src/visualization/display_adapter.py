from abc import ABC, abstractmethod

import cv2

from src.cognition.cognitive_state import CognitiveState


def _screen_size() -> tuple[int, int]:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


class DisplayAdapter(ABC):
    """Output layer for Synapse. Swap OpenCV today for a wearable HUD tomorrow."""

    @abstractmethod
    def show(self, frame, cognitive_state: CognitiveState | None = None, autonomy: float | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_key(self, delay_ms: int = 1) -> int:
        raise NotImplementedError

    def poll_quit(self) -> bool:
        return self.try_handle_display_key(self.read_key())

    def try_handle_display_key(self, key: int) -> bool:
        if key in (ord("f"), ord("F")):
            self.toggle_fullscreen()
            return False
        return key == ord("q")

    def toggle_fullscreen(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class OpenCVDisplayAdapter(DisplayAdapter):
    def __init__(self, window_name: str, *, fullscreen: bool = False) -> None:
        self.window_name = window_name
        self._fullscreen = fullscreen
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            self._set_fullscreen(True)

    def _set_fullscreen(self, enabled: bool) -> None:
        self._fullscreen = enabled
        prop = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop)

    def toggle_fullscreen(self) -> None:
        self._set_fullscreen(not self._fullscreen)

    def show(self, frame, cognitive_state: CognitiveState | None = None, autonomy: float | None = None) -> None:
        if self._fullscreen:
            width, height = _screen_size()
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.window_name, frame)

    def read_key(self, delay_ms: int = 1) -> int:
        return cv2.waitKey(delay_ms) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()


class HudStubDisplayAdapter(DisplayAdapter):
    """Development stand-in for a helmet-mounted HUD output channel."""

    def __init__(self, window_name: str = "Synapse HUD Stub", *, fullscreen: bool = False) -> None:
        self.window_name = window_name
        self._last_payload = ""
        self._fullscreen = fullscreen
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            self._set_fullscreen(True)

    def _set_fullscreen(self, enabled: bool) -> None:
        self._fullscreen = enabled
        prop = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop)

    def toggle_fullscreen(self) -> None:
        self._set_fullscreen(not self._fullscreen)

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
        if self._fullscreen:
            width, height = _screen_size()
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.window_name, frame)

    def read_key(self, delay_ms: int = 1) -> int:
        return cv2.waitKey(delay_ms) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()


def create_display_adapter(
    mode: str,
    window_name: str,
    *,
    fullscreen: bool = False,
) -> DisplayAdapter:
    if mode == "hud_stub":
        return HudStubDisplayAdapter(window_name, fullscreen=fullscreen)
    return OpenCVDisplayAdapter(window_name, fullscreen=fullscreen)
