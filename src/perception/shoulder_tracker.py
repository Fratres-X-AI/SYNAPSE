"""Track shoulder height from MediaPipe Pose — inhale lifts vs normal pen chewing."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_MIN_VISIBILITY = 0.55


@dataclass(frozen=True)
class ShoulderSample:
    visible: bool
    center_y: float = 0.0
    lift: float = 0.0
    elevated: bool = False

    def hud_text(self) -> str:
        if not self.visible:
            return "SHL --"
        state = "UP" if self.elevated else "NOM"
        return f"SHL {state} {self.lift * 1000:.0f}"


@dataclass
class ShoulderTracker:
    """Baseline shoulder height; elevation catches inhale/exhale posture."""

    elevate_threshold: float = 0.014
    calm_blend: float = 0.04
    _baseline_y: float | None = None

    def update(self, pose_landmarks: Sequence[Any] | None) -> ShoulderSample:
        if pose_landmarks is None or len(pose_landmarks) <= _RIGHT_SHOULDER:
            return ShoulderSample(visible=False)

        left = pose_landmarks[_LEFT_SHOULDER]
        right = pose_landmarks[_RIGHT_SHOULDER]
        if left.visibility < _MIN_VISIBILITY or right.visibility < _MIN_VISIBILITY:
            return ShoulderSample(visible=False)

        center_y = (left.y + right.y) / 2.0
        if self._baseline_y is None:
            self._baseline_y = center_y
            return ShoulderSample(visible=True, center_y=center_y, lift=0.0, elevated=False)

        lift = self._baseline_y - center_y
        elevated = lift >= self.elevate_threshold
        if not elevated:
            self._baseline_y = ((1.0 - self.calm_blend) * self._baseline_y) + (self.calm_blend * center_y)

        return ShoulderSample(
            visible=True,
            center_y=center_y,
            lift=max(0.0, lift),
            elevated=elevated,
        )


def draw_shoulder_markers(frame, pose_landmarks: Sequence[Any] | None, sample: ShoulderSample | None) -> None:
    """Light shoulder dots for live calibration runs."""
    import cv2

    if pose_landmarks is None or sample is None or not sample.visible:
        return

    height, width = frame.shape[:2]
    color = (96, 220, 140) if sample.elevated else (140, 180, 210)
    for index in (_LEFT_SHOULDER, _RIGHT_SHOULDER):
        point = pose_landmarks[index]
        if point.visibility < _MIN_VISIBILITY:
            continue
        px = int(point.x * width)
        py = int(point.y * height)
        cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
