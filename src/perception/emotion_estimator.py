from collections import deque
from typing import Any, Sequence

import numpy as np

from src.cognition.emotion_state import Emotion, EmotionState


class EmotionEstimator:
    """Expression cues from MediaPipe landmarks. Webcam-friendly, no ML model."""

    # MediaPipe anatomical left/right (subject's face).
    LEFT_BROW = (293, 334, 336)
    RIGHT_BROW = (63, 105, 107)
    # Closed eye rings in contour order (image left / image right).
    LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
    RIGHT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
    LEFT_CHEEK = (50, 123, 147, 187, 205, 207)
    RIGHT_CHEEK = (280, 352, 376, 411, 425, 427)
    MOUTH = (61, 291, 13, 14, 78, 308, 82, 312, 87, 317)
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_BOTTOM = 374
    UPPER_LIP = 13
    LOWER_LIP = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    LEFT_EYE_TOP = 159
    RIGHT_EYE_TOP = 386
    FOREHEAD = 10
    CHIN = 152
    LEFT_FACE = 234
    RIGHT_FACE = 454

    REGION_COLORS = {
        "left_brow": (0, 220, 255),
        "right_brow": (0, 220, 255),
        "left_eye": (255, 220, 0),
        "right_eye": (255, 220, 0),
        "left_cheek": (180, 0, 255),
        "right_cheek": (180, 0, 255),
        "mouth": (255, 180, 0),
    }

    def __init__(
        self,
        smile_threshold: float = 0.014,
        sad_smile_threshold: float = 0.018,
        neutral_smile_band: float = 0.012,
        cheek_raise_threshold: float = 0.008,
        brow_raise_threshold: float = 0.020,
        mouth_open_threshold: float = 0.35,
        stress_ear_threshold: float = 0.22,
        smooth_window: int = 6,
        calibration_frames: int = 45,
    ) -> None:
        self.smile_threshold = smile_threshold
        self.sad_smile_threshold = sad_smile_threshold
        self.neutral_smile_band = neutral_smile_band
        self.cheek_raise_threshold = cheek_raise_threshold
        self.brow_raise_threshold = brow_raise_threshold
        self.mouth_open_threshold = mouth_open_threshold
        self.stress_ear_threshold = stress_ear_threshold
        self._smooth_window = smooth_window
        self._calibration_frames = calibration_frames
        self._calibration_count = 0
        self._baseline_ready = False
        self._baseline_smile = 0.0
        self._baseline_cheek = 0.0
        self._history: dict[str, deque[float]] = {
            "smile_score": deque(maxlen=smooth_window),
            "cheek_raise": deque(maxlen=smooth_window),
            "brow_raise": deque(maxlen=smooth_window),
            "mouth_open": deque(maxlen=smooth_window),
            "brow_furrow": deque(maxlen=smooth_window),
            "brow_inner_pinch": deque(maxlen=smooth_window),
        }

    @classmethod
    def tracking_regions(cls) -> dict[str, tuple[int, ...]]:
        return {
            "left_brow": cls.LEFT_BROW,
            "right_brow": cls.RIGHT_BROW,
            "left_eye": cls.LEFT_EYE,
            "right_eye": cls.RIGHT_EYE,
            "left_cheek": cls.LEFT_CHEEK,
            "right_cheek": cls.RIGHT_CHEEK,
            "mouth": cls.MOUTH,
        }

    def update(self, landmarks: Sequence[Any], ear: float | None = None) -> EmotionState:
        raw = {
            "smile_score": self._smile_score(landmarks),
            "cheek_raise": self._cheek_raise(landmarks),
            "brow_raise": self._brow_raise(landmarks),
            "mouth_open": self._mouth_open(landmarks),
            "brow_furrow": self._brow_furrow(landmarks),
            "brow_inner_pinch": self._brow_inner_pinch(landmarks),
        }
        signals = {key: self._smooth(key, value) for key, value in raw.items()}
        signals["ear"] = ear
        signals["eye_open"] = ear
        self._update_baseline(signals["smile_score"], signals["cheek_raise"])
        signals["smile_delta"] = signals["smile_score"] - self._baseline_smile
        signals["cheek_delta"] = signals["cheek_raise"] - self._baseline_cheek
        signals["calibrating"] = not self._baseline_ready

        emotion, confidence = self._classify_emotion(signals)

        return EmotionState(
            emotion=emotion,
            confidence=confidence,
            signals=signals,
        )

    def set_neutral_baseline(self, smile_score: float, cheek_raise: float) -> None:
        self._baseline_smile = smile_score
        self._baseline_cheek = cheek_raise
        self._baseline_ready = True
        self._calibration_count = self._calibration_frames

    @staticmethod
    def snapshot_signals(signals: dict[str, Any]) -> dict[str, float]:
        keys = (
            "smile_score",
            "smile_delta",
            "cheek_raise",
            "cheek_delta",
            "brow_raise",
            "brow_furrow",
            "brow_inner_pinch",
            "mouth_open",
            "ear",
        )
        return {key: float(signals[key]) for key in keys if signals.get(key) is not None}

    def _smooth(self, key: str, value: float) -> float:
        self._history[key].append(value)
        return float(np.mean(self._history[key]))

    def _face_height(self, landmarks: Sequence[Any]) -> float:
        return max(abs(landmarks[self.CHIN].y - landmarks[self.FOREHEAD].y), 1e-6)

    def _face_width(self, landmarks: Sequence[Any]) -> float:
        return max(abs(landmarks[self.RIGHT_FACE].x - landmarks[self.LEFT_FACE].x), 1e-6)

    def _smile_score(self, landmarks: Sequence[Any]) -> float:
        left = landmarks[self.MOUTH_LEFT]
        right = landmarks[self.MOUTH_RIGHT]
        upper = landmarks[self.UPPER_LIP]
        lower = landmarks[self.LOWER_LIP]

        face_height = self._face_height(landmarks)
        corner_y = (left.y + right.y) / 2.0
        lip_center_y = (upper.y + lower.y) / 2.0
        return float((lip_center_y - corner_y) / face_height)

    def _cheek_raise(self, landmarks: Sequence[Any]) -> float:
        face_height = self._face_height(landmarks)
        cheek_y = self._avg_y(landmarks, self.LEFT_CHEEK + self.RIGHT_CHEEK)
        eye_bottom_y = (
            landmarks[self.LEFT_EYE_BOTTOM].y + landmarks[self.RIGHT_EYE_BOTTOM].y
        ) / 2.0
        return float((eye_bottom_y - cheek_y) / face_height)

    def _mouth_open(self, landmarks: Sequence[Any]) -> float:
        upper = self._point(landmarks[self.UPPER_LIP])
        lower = self._point(landmarks[self.LOWER_LIP])
        left = self._point(landmarks[self.MOUTH_LEFT])
        right = self._point(landmarks[self.MOUTH_RIGHT])

        vertical = np.linalg.norm(upper - lower)
        horizontal = np.linalg.norm(left - right)
        if horizontal == 0:
            return 0.0
        return float(vertical / horizontal)

    def _brow_raise(self, landmarks: Sequence[Any]) -> float:
        face_height = self._face_height(landmarks)
        brow_y = self._avg_y(landmarks, self.LEFT_BROW + self.RIGHT_BROW)
        eye_y = (landmarks[self.LEFT_EYE_TOP].y + landmarks[self.RIGHT_EYE_TOP].y) / 2.0
        return float((eye_y - brow_y) / face_height)

    def _brow_furrow(self, landmarks: Sequence[Any]) -> float:
        face_height = self._face_height(landmarks)
        brow_y = self._avg_y(landmarks, self.LEFT_BROW + self.RIGHT_BROW)
        eye_y = (landmarks[self.LEFT_EYE_TOP].y + landmarks[self.RIGHT_EYE_TOP].y) / 2.0
        return float((brow_y - eye_y) / face_height)

    def _brow_inner_pinch(self, landmarks: Sequence[Any]) -> float:
        left_inner = self._point(landmarks[self.LEFT_BROW[2]])
        right_inner = self._point(landmarks[self.RIGHT_BROW[2]])
        return float(np.linalg.norm(left_inner - right_inner) / self._face_width(landmarks))

    def _update_baseline(self, smile: float, cheek: float) -> None:
        if self._baseline_ready or self._calibration_frames <= 0:
            return
        if self._calibration_count >= self._calibration_frames:
            self._baseline_ready = True
            return
        count = self._calibration_count
        self._baseline_smile = ((self._baseline_smile * count) + smile) / (count + 1)
        self._baseline_cheek = ((self._baseline_cheek * count) + cheek) / (count + 1)
        self._calibration_count += 1
        if self._calibration_count >= self._calibration_frames:
            self._baseline_ready = True

    def _classify_emotion(self, signals: dict[str, Any]) -> tuple[Emotion, float]:
        if signals.get("calibrating"):
            if self._calibration_frames > 0:
                progress = self._calibration_count / self._calibration_frames
                return Emotion.NEUTRAL, float(0.5 + progress * 0.2)
            return Emotion.NEUTRAL, 0.5

        smile = float(signals["smile_delta"])
        cheek = float(signals["cheek_delta"])
        brow_raise = float(signals["brow_raise"])
        mouth_open = float(signals["mouth_open"])
        brow_furrow = float(signals["brow_furrow"])
        brow_inner_pinch = float(signals["brow_inner_pinch"])
        ear = signals.get("ear")

        if smile >= self.smile_threshold and cheek >= self.cheek_raise_threshold:
            margin = smile - self.smile_threshold
            return Emotion.HAPPY, float(np.clip(0.68 + margin * 8.0, 0.68, 0.95))

        if smile >= self.smile_threshold * 0.85:
            margin = smile - self.smile_threshold * 0.85
            return Emotion.HAPPY, float(np.clip(0.62 + margin * 6.0, 0.62, 0.88))

        if brow_raise >= self.brow_raise_threshold and mouth_open >= self.mouth_open_threshold:
            margin = min(brow_raise, mouth_open)
            return Emotion.SURPRISED, float(np.clip(0.65 + margin * 2.0, 0.65, 0.95))

        if brow_furrow >= self.brow_raise_threshold * 0.8 and (
            brow_inner_pinch <= 0.42 or ear is None or ear <= self.stress_ear_threshold
        ):
            return Emotion.STRESSED, 0.72

        if smile <= -self.sad_smile_threshold:
            margin = abs(smile) - self.sad_smile_threshold
            return Emotion.SAD, float(np.clip(0.68 + margin * 8.0, 0.68, 0.95))

        if (
            abs(smile) <= self.neutral_smile_band
            and brow_raise < self.brow_raise_threshold * 0.75
            and brow_furrow < self.brow_raise_threshold * 0.75
            and mouth_open < self.mouth_open_threshold * 0.65
        ):
            return Emotion.NEUTRAL, 0.78

        return Emotion.NEUTRAL, 0.6

    def _avg_y(self, landmarks: Sequence[Any], indices: Sequence[int]) -> float:
        return float(sum(landmarks[index].y for index in indices) / len(indices))

    def _point(self, landmark: Any) -> np.ndarray:
        return np.array([landmark.x, landmark.y], dtype=np.float64)
