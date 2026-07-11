"""Lightweight smoothed FPS counter for live HUD telemetry."""

from __future__ import annotations

from time import monotonic


class FpsTracker:
    def __init__(self, smoothing: float = 0.12) -> None:
        self._smoothing = smoothing
        self._fps = 0.0
        self._last = monotonic()

    def tick(self) -> float:
        now = monotonic()
        dt = now - self._last
        self._last = now
        if dt <= 0:
            return self._fps
        instant = 1.0 / dt
        if self._fps <= 0:
            self._fps = instant
        else:
            blend = self._smoothing
            self._fps = self._fps * (1.0 - blend) + instant * blend
        return self._fps
