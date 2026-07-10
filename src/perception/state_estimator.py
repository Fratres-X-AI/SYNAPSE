from collections import deque
from time import monotonic
from typing import Any, Sequence

import numpy as np

from src.cognition.cognitive_state import CognitiveState, State
from src.perception.face_geometry import compute_hairline, compute_peripheral_mesh


class StateEstimator:
    """Converts MediaPipe Face Mesh landmarks into coarse cognitive state signals."""

    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)
    NOSE_TIP = 1
    FOREHEAD = 10
    CHIN = 152
    LEFT_FACE = 234
    RIGHT_FACE = 454
    LEFT_IRIS = 468
    RIGHT_IRIS = 473

    def __init__(
        self,
        ear_blink_threshold: float = 0.21,
        ear_blink_open_threshold: float | None = None,
        low_ear_threshold: float = 0.24,
        high_blink_rate_threshold: float = 25.0,
        distracted_yaw_threshold: float = 25.0,
        distracted_pitch_threshold: float = 20.0,
        gaze_label_threshold: float = 0.12,
        distracted_gaze_threshold: float = 0.34,
        distracted_gaze_up_threshold: float = 0.22,
        gaze_meter_threshold: float = 0.30,
        distracted_pitch_up_threshold: float = 14.0,
        history_seconds: float = 60.0,
    ) -> None:
        self.ear_blink_close_threshold = ear_blink_threshold
        self.ear_blink_open_threshold = ear_blink_open_threshold or max(
            ear_blink_threshold + 0.025,
            low_ear_threshold * 0.9,
        )
        self.low_ear_threshold = low_ear_threshold
        self.high_blink_rate_threshold = high_blink_rate_threshold
        self.distracted_yaw_threshold = distracted_yaw_threshold
        self.distracted_pitch_threshold = distracted_pitch_threshold
        self.gaze_label_threshold = gaze_label_threshold
        self.distracted_gaze_threshold = distracted_gaze_threshold
        self.distracted_gaze_up_threshold = distracted_gaze_up_threshold
        self.gaze_meter_threshold = gaze_meter_threshold
        self.distracted_pitch_up_threshold = distracted_pitch_up_threshold
        self.history_seconds = history_seconds

        self.blink_counter = 0
        self.ear_history: deque[tuple[float, float]] = deque()
        self.head_yaw_history: deque[tuple[float, float]] = deque()
        self.head_pitch_history: deque[tuple[float, float]] = deque()
        self.blink_timestamps: deque[float] = deque()
        self._open_ear_samples: deque[float] = deque(maxlen=15)
        self._currently_blinking = False
        self._blink_trough = 1.0
        self._blink_started_at = 0.0
        self._close_frame_streak = 0
        self._distraction_frame_streak = 0
        self._started_at = monotonic()

    def update(self, landmarks: Sequence[Any]) -> CognitiveState:
        """
        landmarks: MediaPipe face landmarks, normalized to image coordinates.
        """
        now = monotonic()
        ear = self._calculate_ear(landmarks)
        left_ear, right_ear, blink_ear = self._calculate_blink_ears(landmarks)
        tracking_stable = self._face_tracking_stable(landmarks)
        is_blinking = (
            self._detect_blink(left_ear, right_ear, blink_ear, now)
            if tracking_stable
            else self._handle_occlusion(blink_ear)
        )
        blink_rate = self._get_blink_rate(now)
        head_yaw = self._estimate_head_yaw(landmarks)
        head_pitch = self._estimate_head_pitch(landmarks)
        gaze_x, gaze_y = self._estimate_gaze(landmarks)
        gaze_direction = self._gaze_direction(gaze_x, gaze_y)
        hairline = compute_hairline(landmarks)
        peripheral = compute_peripheral_mesh(landmarks)

        self._append_history(self.ear_history, ear, now)
        self._append_history(self.head_yaw_history, head_yaw, now)
        self._append_history(self.head_pitch_history, head_pitch, now)

        state = self._classify_state(
            ear, is_blinking, head_yaw, head_pitch, gaze_x, gaze_y, blink_rate
        )
        confidence = self._estimate_confidence(
            ear, head_yaw, head_pitch, gaze_x, gaze_y, blink_rate, state
        )

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
                "head_pitch": head_pitch,
                "gaze_x": gaze_x,
                "gaze_y": gaze_y,
                "gaze_direction": gaze_direction,
                "hairline_x": hairline.x,
                "hairline_y": hairline.y,
                "eye_to_hairline": hairline.eye_to_hairline,
                "chin_to_hairline": hairline.chin_to_hairline,
                "brow_to_hairline": hairline.brow_to_hairline,
                "ear_to_hairline": (hairline.left_ear_to_hairline + hairline.right_ear_to_hairline) / 2.0,
                "peripheral_landmarks": peripheral.point_count,
            },
        )

    def _calculate_ear(self, landmarks: Sequence[Any]) -> float:
        left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE)
        right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE)
        return float((left_ear + right_ear) / 2.0)

    def _calculate_blink_ears(self, landmarks: Sequence[Any]) -> tuple[float, float, float]:
        left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE)
        right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE)
        return left_ear, right_ear, float(min(left_ear, right_ear))

    def _face_tracking_stable(self, landmarks: Sequence[Any]) -> bool:
        hairline = compute_hairline(landmarks)
        return hairline.face_width > 0.08 and hairline.chin_to_hairline > 0.12

    def _handle_occlusion(self, ear: float) -> bool:
        self._close_frame_streak = 0
        if self._currently_blinking:
            self._currently_blinking = False
        return False

    def _can_register_blink(
        self,
        left_ear: float,
        right_ear: float,
        blink_ear: float,
        close_threshold: float,
    ) -> bool:
        if blink_ear < 0.035:
            return False
        if abs(left_ear - right_ear) > 0.18:
            return False
        if len(self._open_ear_samples) >= 3:
            recent_open = list(self._open_ear_samples)[-3:]
            if float(np.mean(recent_open)) < close_threshold:
                return False
        return True

    def _open_ear_baseline(self) -> float:
        if self._open_ear_samples:
            return float(np.mean(self._open_ear_samples))
        return self.low_ear_threshold

    def _dynamic_close_threshold(self) -> float:
        baseline = self._open_ear_baseline()
        return min(self.ear_blink_close_threshold, baseline * 0.86)

    def _eyes_reopened(self, ear: float) -> bool:
        baseline = self._open_ear_baseline()
        recovery = ear - self._blink_trough
        recovery_target = max(0.010, (baseline - self._blink_trough) * 0.28)
        absolute_open = ear >= max(
            self.ear_blink_open_threshold,
            baseline - max(0.018, (baseline - self.ear_blink_close_threshold) * 0.45),
        )
        return absolute_open or recovery >= recovery_target

    def _detect_blink(
        self,
        left_ear: float,
        right_ear: float,
        blink_ear: float,
        now: float | None = None,
    ) -> bool:
        now = monotonic() if now is None else now
        close_threshold = self._dynamic_close_threshold()
        mean_ear = (left_ear + right_ear) / 2.0
        either_eye_closed = left_ear < close_threshold or right_ear < close_threshold
        mean_closed = mean_ear < close_threshold
        both_closed = mean_closed and either_eye_closed

        if not self._currently_blinking and both_closed:
            self._close_frame_streak += 1
        else:
            self._close_frame_streak = 0

        if not self._currently_blinking:
            if not both_closed:
                self._open_ear_samples.append(blink_ear)

        if self._currently_blinking:
            self._blink_trough = min(self._blink_trough, blink_ear)
            if self._eyes_reopened(blink_ear) or now - self._blink_started_at > 0.35:
                self._currently_blinking = False
                self._close_frame_streak = 0
                if blink_ear >= close_threshold * 0.9:
                    self._open_ear_samples.append(blink_ear)
        elif (
            both_closed
            and self._close_frame_streak >= 1
            and self._can_register_blink(left_ear, right_ear, blink_ear, close_threshold)
        ):
            self._currently_blinking = True
            self._blink_trough = blink_ear
            self._blink_started_at = now
            self.blink_counter += 1
            self.blink_timestamps.append(now)
            self._close_frame_streak = 0

        self._prune_old(self.blink_timestamps, now)
        return self._currently_blinking

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

    def _estimate_head_pitch(self, landmarks: Sequence[Any]) -> float:
        nose = landmarks[self.NOSE_TIP]
        chin = landmarks[self.CHIN]
        hairline = compute_hairline(landmarks)

        face_height = hairline.chin_to_hairline
        if face_height == 0:
            return 0.0

        face_center_y = (hairline.y + chin.y) / 2.0
        nose_offset = (nose.y - face_center_y) / face_height
        eye_y = float(np.mean([landmarks[index].y for index in self.LEFT_EYE + self.RIGHT_EYE]))
        eye_offset = (eye_y - face_center_y) / face_height
        combined = nose_offset * 0.45 + eye_offset * 0.55
        return float(np.clip(combined * 70.0, -35.0, 35.0))

    def _estimate_gaze(self, landmarks: Sequence[Any]) -> tuple[float, float]:
        left_x, left_y = self._eye_gaze_offset(landmarks, self.LEFT_EYE, self.LEFT_IRIS)
        right_x, right_y = self._eye_gaze_offset(landmarks, self.RIGHT_EYE, self.RIGHT_IRIS)
        return (
            float(np.clip((left_x + right_x) / 2.0, -0.5, 0.5)),
            float(np.clip((left_y + right_y) / 2.0, -0.5, 0.5)),
        )

    def _eye_gaze_offset(
        self,
        landmarks: Sequence[Any],
        eye_indices: tuple[int, ...],
        iris_index: int,
    ) -> tuple[float, float]:
        outer, top_a, top_b, inner, bottom_a, bottom_b = (
            self._point(landmarks[index]) for index in eye_indices
        )
        iris = self._point(landmarks[iris_index])

        eye_width = inner[0] - outer[0]
        eye_height = ((bottom_a[1] + bottom_b[1]) / 2.0) - ((top_a[1] + top_b[1]) / 2.0)

        if eye_width == 0 or eye_height == 0:
            return 0.0, 0.0

        gaze_x = (iris[0] - outer[0]) / eye_width - 0.5
        gaze_y = (iris[1] - ((top_a[1] + top_b[1]) / 2.0)) / eye_height - 0.5
        return float(gaze_x), float(gaze_y)

    def _gaze_direction(self, gaze_x: float, gaze_y: float) -> str:
        threshold = self.gaze_label_threshold
        horizontal = "center"
        vertical = "center"

        if gaze_x <= -threshold:
            horizontal = "left"
        elif gaze_x >= threshold:
            horizontal = "right"

        if gaze_y <= -threshold:
            vertical = "up"
        elif gaze_y >= threshold:
            vertical = "down"

        if horizontal == "center" and vertical == "center":
            return "center"
        if horizontal == "center":
            return vertical
        if vertical == "center":
            return horizontal
        return f"{horizontal}-{vertical}"

    def _classify_state(
        self,
        ear: float,
        is_blinking: bool,
        head_yaw: float,
        head_pitch: float,
        gaze_x: float,
        gaze_y: float,
        blink_rate: float,
    ) -> State:
        mean_ear = self._mean_history(self.ear_history, default=ear)

        head_distracted = (
            abs(head_yaw) >= self.distracted_yaw_threshold
            or abs(head_pitch) >= self.distracted_pitch_threshold
            or head_pitch <= -self.distracted_pitch_up_threshold
        )
        gaze_distracted = (
            abs(gaze_x) >= self.distracted_gaze_threshold
            or gaze_y >= self.distracted_gaze_threshold
            or gaze_y <= -self.distracted_gaze_up_threshold
        )
        if head_distracted or gaze_distracted:
            self._distraction_frame_streak += 1
        else:
            self._distraction_frame_streak = 0

        if head_distracted or (gaze_distracted and self._distraction_frame_streak >= 6):
            return State.DISTRACTED

        if blink_rate >= self.high_blink_rate_threshold and mean_ear <= self.low_ear_threshold:
            return State.FATIGUED

        if (
            not is_blinking
            and ear > self.low_ear_threshold
            and blink_rate < self.high_blink_rate_threshold
            and abs(head_yaw) < self.distracted_yaw_threshold * 0.5
            and abs(head_pitch) < self.distracted_pitch_threshold * 0.5
            and abs(gaze_x) < self.distracted_gaze_threshold * 0.5
            and abs(gaze_y) < self.distracted_gaze_threshold * 0.5
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

    def distraction_score(self, signals: dict[str, Any]) -> int:
        """Head pose weighs more than gaze so screen-corner glances stay sub-distracted."""
        head_component = max(
            abs(signals["head_yaw"]) / self.distracted_yaw_threshold,
            abs(signals["head_pitch"]) / self.distracted_pitch_threshold,
        )
        gaze_component = max(
            abs(signals["gaze_x"]) / self.gaze_meter_threshold,
            max(signals["gaze_y"], 0.0) / self.gaze_meter_threshold,
            max(-signals["gaze_y"], 0.0) / self.distracted_gaze_up_threshold,
        ) * 0.55
        return int(np.clip(max(head_component, gaze_component) * 100, 0, 100))

    def _estimate_confidence(
        self,
        ear: float,
        head_yaw: float,
        head_pitch: float,
        gaze_x: float,
        gaze_y: float,
        blink_rate: float,
        state: State,
    ) -> float:
        if state == State.DISTRACTED:
            yaw_margin = abs(head_yaw) - self.distracted_yaw_threshold
            pitch_margin = abs(head_pitch) - self.distracted_pitch_threshold
            gaze_margin = max(abs(gaze_x), abs(gaze_y)) - self.distracted_gaze_threshold
            margin = max(yaw_margin, pitch_margin, gaze_margin)
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
