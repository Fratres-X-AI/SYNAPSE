from collections import deque
from time import monotonic
from typing import Any, Sequence

import numpy as np

from src.cognition.cognitive_state import CognitiveState, State


class StateEstimator:
    """Converts MediaPipe Face Mesh landmarks into coarse cognitive state signals."""

    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)
    NOSE_TIP = 1
    LEFT_FACE = 234
    RIGHT_FACE = 454

    def __init__(
        self,
        ear_blink_threshold: float = 0.21,
        low_ear_threshold: float = 0.24,
        high_blink_rate_threshold: float = 25.0,
        distracted_yaw_threshold: float = 25.0,
        history_seconds: float = 60.0,
    ) -> None:
        self.ear_blink_threshold = ear_blink_threshold
        self.low_ear_threshold = low_ear_threshold
        self.high_blink_rate_threshold = high_blink_rate_threshold
        self.distracted_yaw_threshold = distracted_yaw_threshold
        self.history_seconds = history_seconds

        self.blink_counter = 0
        self.ear_history: deque[tuple[float, float]] = deque()
        self.head_yaw_history: deque[tuple[float, float]] = deque()
        self.blink_timestamps: deque[float] = deque()
        self._currently_blinking = False
        self._started_at = monotonic()

    def update(self, landmarks: Sequence[Any]) -> CognitiveState:
        """
        landmarks: MediaPipe face landmarks, normalized to image coordinates.
        """
        now = monotonic()
        ear = self._calculate_ear(landmarks)
        is_blinking = self._detect_blink(ear, now)
        blink_rate = self._get_blink_rate(now)
        head_yaw = self._estimate_head_yaw(landmarks)

        self._append_history(self.ear_history, ear, now)
        self._append_history(self.head_yaw_history, head_yaw, now)

        state = self._classify_state(ear, is_blinking, head_yaw, blink_rate)
        confidence = self._estimate_confidence(ear, head_yaw, blink_rate, state)

        return CognitiveState(
            state=state,
            confidence=confidence,
            signals={
                "ear": ear,
                "mean_ear": self._mean_history(self.ear_history),
                "is_blinking": is_blinking,
                "blink_rate": blink_rate,
                "blink_counter": self.blink_counter,
                "head_yaw": head_yaw,
            },
        )

    def _calculate_ear(self, landmarks: Sequence[Any]) -> float:
        left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE)
        right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE)
        return float((left_ear + right_ear) / 2.0)

    def _detect_blink(self, ear: float, now: float | None = None) -> bool:
        now = monotonic() if now is None else now
        is_closed = ear < self.ear_blink_threshold

        if is_closed and not self._currently_blinking:
            self.blink_counter += 1
            self.blink_timestamps.append(now)

        self._currently_blinking = is_closed
        self._prune_old(self.blink_timestamps, now)
        return is_closed

    def _estimate_head_yaw(self, landmarks: Sequence[Any]) -> float:
        nose = landmarks[self.NOSE_TIP]
        left_face = landmarks[self.LEFT_FACE]
        right_face = landmarks[self.RIGHT_FACE]

        face_width = abs(right_face.x - left_face.x)
        if face_width == 0:
            return 0.0

        face_center_x = (left_face.x + right_face.x) / 2.0
        normalized_offset = (nose.x - face_center_x) / face_width
        return float(np.clip(normalized_offset * 90.0, -45.0, 45.0))

    def _classify_state(
        self,
        ear: float,
        is_blinking: bool,
        head_yaw: float,
        blink_rate: float,
    ) -> State:
        mean_ear = self._mean_history(self.ear_history, default=ear)

        if abs(head_yaw) >= self.distracted_yaw_threshold:
            return State.DISTRACTED

        if blink_rate >= self.high_blink_rate_threshold and mean_ear <= self.low_ear_threshold:
            return State.FATIGUED

        if (
            not is_blinking
            and ear > self.low_ear_threshold
            and blink_rate < self.high_blink_rate_threshold
            and abs(head_yaw) < self.distracted_yaw_threshold * 0.5
        ):
            return State.HIGH_ATTENTION

        return State.MODERATE

    def _eye_aspect_ratio(self, landmarks: Sequence[Any], eye_indices: tuple[int, ...]) -> float:
        p1, p2, p3, p4, p5, p6 = (self._point(landmarks[index]) for index in eye_indices)
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)

        if horizontal == 0:
            return 0.0

        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

    def _get_blink_rate(self, now: float) -> float:
        self._prune_old(self.blink_timestamps, now)
        elapsed = max(10.0, min(self.history_seconds, now - self._started_at))
        return float(len(self.blink_timestamps) * 60.0 / elapsed)

    def _estimate_confidence(
        self,
        ear: float,
        head_yaw: float,
        blink_rate: float,
        state: State,
    ) -> float:
        if state == State.DISTRACTED:
            margin = abs(head_yaw) - self.distracted_yaw_threshold
            return float(np.clip(0.65 + margin / 40.0, 0.65, 0.95))

        if state == State.FATIGUED:
            rate_margin = blink_rate - self.high_blink_rate_threshold
            ear_margin = self.low_ear_threshold - ear
            return float(np.clip(0.6 + rate_margin / 40.0 + ear_margin, 0.6, 0.95))

        if state == State.HIGH_ATTENTION:
            return 0.8

        return 0.6

    def _append_history(self, history: deque[tuple[float, float]], value: float, now: float) -> None:
        history.append((now, value))
        while history and now - history[0][0] > self.history_seconds:
            history.popleft()

    def _mean_history(self, history: deque[tuple[float, float]], default: float = 0.0) -> float:
        if not history:
            return default
        return float(np.mean([value for _, value in history]))

    def _prune_old(self, timestamps: deque[float], now: float) -> None:
        while timestamps and now - timestamps[0] > self.history_seconds:
            timestamps.popleft()

    def _point(self, landmark: Any) -> np.ndarray:
        return np.array([landmark.x, landmark.y], dtype=np.float64)
