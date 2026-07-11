"""Secondary presence detection — labeled people and desk items around the primary user."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import mediapipe as mp
import numpy as np

from utils.presence_model import object_model_bytes

_NOSE_TIP = 1
_MOUTH_INDICES = (13, 14, 78, 308)

DISPLAY_LABELS = {
    "user": "User",
    "you": "User",
    "person": "Person",
    "phone": "Phone",
    "vape": "Vape",
    "food": "Food",
    "drink": "Drink",
    "pen": "Pen",
    "glasses": "Glasses",
    "hand": "Hand",
    "desk": "Desk",
    "laptop": "Laptop",
    "book": "Book",
    "scissors": "Scissors",
    "smoking": "Smoking",
}

DESK_OBJECT_ALIASES = {
    "cell phone": "phone",
    "mobile phone": "phone",
    "phone": "phone",
    "cup": "drink",
    "bottle": "drink",
    "wine glass": "drink",
    "sandwich": "food",
    "apple": "food",
    "banana": "food",
    "pizza": "food",
    "donut": "food",
    "orange": "food",
    "cake": "food",
    "hot dog": "food",
    "book": "book",
    "laptop": "laptop",
    "keyboard": "desk",
    "mouse": "desk",
    "remote": "desk",
    "scissors": "scissors",
    "toothbrush": "handheld",
    "tie": "desk",
    "backpack": "desk",
    "handbag": "desk",
    "suitcase": "desk",
    "sports ball": "desk",
}


@dataclass(frozen=True)
class PresenceBox:
    """Normalized bounding box for a labeled person or desk item."""

    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    is_primary: bool = False

    @property
    def width(self) -> float:
        return max(self.x_max - self.x_min, 0.0)

    @property
    def height(self) -> float:
        return max(self.y_max - self.y_min, 0.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.width / 2.0, self.y_min + self.height / 2.0)


@dataclass(frozen=True)
class PresenceFrame:
    """Labeled people and desk objects in the current frame."""

    people: tuple[PresenceBox, ...] = ()
    objects: tuple[PresenceBox, ...] = ()
    primary_index: int | None = None
    events: tuple[str, ...] = ()

    @property
    def face_count(self) -> int:
        return len(self.people)

    @property
    def extra_people(self) -> int:
        return max(0, self.face_count - 1)

    def active_labels(self) -> set[str]:
        labels: set[str] = set()
        primary = (
            self.people[self.primary_index]
            if self.primary_index is not None and self.primary_index < len(self.people)
            else None
        )
        for person in self.people:
            labels.add(person.label)
        for obj in self.objects:
            if obj.label == "person" and primary is not None and _same_face_region(obj, primary):
                continue
            labels.add(obj.label)
        labels.update(self.events)
        return labels

    def all_boxes(self) -> tuple[PresenceBox, ...]:
        return self.people + self.objects


@dataclass
class PresenceTracker:
    """Track short-lived visitor events for CSV alerts."""

    sustain_seconds: float = 2.0
    _active_since: float | None = None
    _last_extra_faces: int = 0

    def update(self, extra_people: int, now: float) -> str:
        if extra_people <= 0:
            event = "cleared" if self._last_extra_faces > 0 else ""
            self._active_since = None
            self._last_extra_faces = 0
            return event

        if self._active_since is None:
            self._active_since = now

        self._last_extra_faces = extra_people
        if now - self._active_since >= self.sustain_seconds:
            return "visitor"
        return ""


@dataclass
class SmokingEventTracker:
    """Log smoking when vapor appears while a vape pod or pen is at the mouth."""

    sustain_seconds: float = 0.35
    cooldown_seconds: float = 6.0
    vapor_boost: float = 9.0
    heavy_vapor_boost: float = 18.0
    event_hold_seconds: float = 2.5
    _mouth_luma_baseline: float | None = None
    _active_since: float | None = None
    _last_event_at: float = -999.0
    _visible_until: float = -999.0

    def update(
        self,
        presence: PresenceFrame,
        rgb_frame,
        landmarks: Sequence[Any] | None,
        now: float,
        shoulder: Any | None = None,
    ) -> tuple[str, ...]:
        if now <= self._visible_until:
            return ("smoking",)

        if landmarks is None:
            self._active_since = None
            return ()

        mouth = _mouth_center(landmarks)
        vape_at_mouth = any(
            obj.label == "vape"
            and _mouth_device_is_vape(obj)
            and _near_mouth(obj, mouth, margin=0.14)
            for obj in presence.objects
        )
        pen_at_mouth = any(
            obj.label == "pen" and _near_mouth(obj, mouth, margin=0.12) for obj in presence.objects
        )
        hand_at_mouth = any(
            obj.label == "hand" and _near_mouth(obj, mouth, margin=0.16)
            for obj in presence.objects
        )
        vapor_level = self._vapor_level(rgb_frame, mouth)
        if vapor_level is None:
            self._active_since = None
            return ()

        vapor, heavy_vapor = vapor_level
        mouth_activity = vape_at_mouth or pen_at_mouth or hand_at_mouth or heavy_vapor
        posture_ok = self._posture_allows_smoking(shoulder, heavy_vapor=heavy_vapor)

        if mouth_activity and vapor and posture_ok:
            if self._active_since is None:
                self._active_since = now
            elif (
                now - self._active_since >= self.sustain_seconds
                and now - self._last_event_at >= self.cooldown_seconds
            ):
                self._last_event_at = now
                self._visible_until = now + self.event_hold_seconds
                return ("smoking",)
        else:
            self._active_since = None
        return ()

    def _roi_mean_luma(self, rgb_frame, x1: int, y1: int, x2: int, y2: int) -> float | None:
        roi = rgb_frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return float(roi.mean())

    def _vapor_level(self, rgb_frame, mouth: tuple[float, float]) -> tuple[bool, bool] | None:
        height, width = rgb_frame.shape[:2]
        mx = int(mouth[0] * width)
        my = int(mouth[1] * height)

        # Rising vapor above the mouth.
        above = self._roi_mean_luma(
            rgb_frame,
            max(0, mx - 55),
            max(0, my - 75),
            min(width, mx + 55),
            max(0, my - 6),
        )
        # Blown exhale in front of / below the mouth.
        exhale = self._roi_mean_luma(
            rgb_frame,
            max(0, mx - 70),
            max(0, my - 18),
            min(width, mx + 70),
            min(height, my + 65),
        )
        readings = [value for value in (above, exhale) if value is not None]
        if not readings:
            return None

        luma = max(readings)
        if self._mouth_luma_baseline is None:
            self._mouth_luma_baseline = luma
            return (False, False)

        self._mouth_luma_baseline = (0.88 * self._mouth_luma_baseline) + (0.12 * luma)
        baseline = self._mouth_luma_baseline
        vapor = luma >= baseline + self.vapor_boost
        heavy_vapor = luma >= baseline + self.heavy_vapor_boost
        return vapor, heavy_vapor

    @staticmethod
    def _posture_allows_smoking(shoulder: Any | None, *, heavy_vapor: bool) -> bool:
        if heavy_vapor:
            return True
        if shoulder is None or not getattr(shoulder, "visible", False):
            return True
        return bool(getattr(shoulder, "elevated", False))


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label.strip().lower(), label.strip().replace("_", " ").title())


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _bbox_from_relative(relative_bbox, frame_width: int, frame_height: int) -> tuple[float, float, float, float]:
    x_min = _clamp01(relative_bbox.xmin)
    y_min = _clamp01(relative_bbox.ymin)
    x_max = _clamp01(x_min + relative_bbox.width)
    y_max = _clamp01(y_min + relative_bbox.height)
    return x_min, y_min, x_max, y_max


def _bbox_from_pixels(origin_x: int, origin_y: int, width: int, height: int, frame_width: int, frame_height: int):
    x_min = _clamp01(origin_x / frame_width)
    y_min = _clamp01(origin_y / frame_height)
    x_max = _clamp01((origin_x + width) / frame_width)
    y_max = _clamp01((origin_y + height) / frame_height)
    return x_min, y_min, x_max, y_max


def _bbox_from_landmarks(landmarks: Sequence[Any], padding: float = 0.04) -> tuple[float, float, float, float]:
    xs = [point.x for point in landmarks]
    ys = [point.y for point in landmarks]
    x_min = _clamp01(min(xs) - padding)
    y_min = _clamp01(min(ys) - padding)
    x_max = _clamp01(max(xs) + padding)
    y_max = _clamp01(max(ys) + padding)
    return x_min, y_min, x_max, y_max


def _point_in_box(x: float, y: float, box: PresenceBox) -> bool:
    return box.x_min <= x <= box.x_max and box.y_min <= y <= box.y_max


def _boxes_overlap(a: PresenceBox, b: PresenceBox) -> bool:
    return not (a.x_max < b.x_min or b.x_max < a.x_min or a.y_max < b.y_min or b.y_max < a.y_min)


def _distance_to_box_center(x: float, y: float, box: PresenceBox) -> float:
    cx, cy = box.center
    return float(np.hypot(x - cx, y - cy))


def _mouth_center(landmarks: Sequence[Any]) -> tuple[float, float]:
    xs = [landmarks[index].x for index in _MOUTH_INDICES]
    ys = [landmarks[index].y for index in _MOUTH_INDICES]
    return float(np.mean(xs)), float(np.mean(ys))


def _faces_overlap(a: PresenceBox, b: PresenceBox, *, threshold: float = 0.15) -> bool:
    ix1 = max(a.x_min, b.x_min)
    iy1 = max(a.y_min, b.y_min)
    ix2 = min(a.x_max, b.x_max)
    iy2 = min(a.y_max, b.y_max)
    if ix1 >= ix2 or iy1 >= iy2:
        return False
    intersection = (ix2 - ix1) * (iy2 - iy1)
    smaller = min(a.area, b.area)
    if smaller <= 0:
        return False
    return (intersection / smaller) >= threshold


def _same_face_region(a: PresenceBox, b: PresenceBox, *, margin: float = 0.24) -> bool:
    if _faces_overlap(a, b):
        return True
    ax, ay = a.center
    bx, by = b.center
    return abs(ax - bx) < margin and abs(ay - by) < margin


def dedupe_face_boxes(faces: list[PresenceBox]) -> list[PresenceBox]:
    """Drop duplicate MediaPipe face boxes that land on the same person."""
    ordered = sorted(faces, key=lambda face: (face.confidence, face.area), reverse=True)
    kept: list[PresenceBox] = []
    for face in ordered:
        if any(_same_face_region(face, existing) for existing in kept):
            continue
        kept.append(face)
    return kept


def select_primary_face(
    faces: list[PresenceBox],
    landmarks: Sequence[Any] | None,
) -> int | None:
    if not faces:
        return None
    if landmarks is None:
        return max(range(len(faces)), key=lambda index: faces[index].area)

    nose = landmarks[_NOSE_TIP]
    containing = [index for index, box in enumerate(faces) if _point_in_box(nose.x, nose.y, box)]
    if containing:
        return max(containing, key=lambda index: faces[index].area)
    return min(range(len(faces)), key=lambda index: _distance_to_box_center(nose.x, nose.y, faces[index]))


def normalize_object_label(raw_label: str) -> str:
    lowered = raw_label.strip().lower()
    return DESK_OBJECT_ALIASES.get(lowered, lowered.replace(" ", "_"))


def _near_mouth(box: PresenceBox, mouth: tuple[float, float], *, margin: float = 0.12) -> bool:
    cx, cy = box.center
    return abs(cx - mouth[0]) < margin and abs(cy - mouth[1]) < margin


def _looks_like_pen(box: PresenceBox) -> bool:
    aspect = box.width / max(box.height, 1e-6)
    thin = aspect > 1.8 or aspect < 0.5
    desk_level = box.center[1] > 0.36
    small_enough = 0.001 < box.area < 0.12
    return thin and desk_level and small_enough


def _looks_like_vape(box: PresenceBox) -> bool:
    """Square/boxy pod form factor — not pen-shaped."""
    aspect = box.width / max(box.height, 1e-6)
    squareish = 0.55 <= aspect <= 1.85
    compact = 0.0025 < box.area < 0.08
    not_pen_thin = not (aspect > 2.0 or aspect < 0.45)
    return squareish and compact and not_pen_thin


def _mouth_device_is_vape(box: PresenceBox) -> bool:
    """Vape at mouth — allow pods held at angles; still reject thin pens."""
    if _looks_like_vape(box):
        return True
    if _looks_like_pen(box):
        return False
    aspect = box.width / max(box.height, 1e-6)
    blocky = 0.4 <= aspect <= 2.5
    in_hand_size = 0.0015 < box.area < 0.14
    return blocky and in_hand_size


def _looks_like_phone(box: PresenceBox) -> bool:
    """Handheld phone slab — not clothing, bags, or wall clutter."""
    aspect = box.width / max(box.height, 1e-6)
    slab = 0.3 <= aspect <= 2.6
    compact = 0.0012 < box.area < 0.05
    not_wall_sized = max(box.width, box.height) < 0.38
    return slab and compact and not_wall_sized


def _qualifies_as_phone(box: PresenceBox, hands: list[PresenceBox]) -> bool:
    if not _looks_like_phone(box):
        return False
    if any(_boxes_overlap(box, hand) for hand in hands):
        return box.confidence >= 0.40
    return box.center[1] > 0.50 and box.area < 0.03 and box.confidence >= 0.48


def _classify_handheld(box: PresenceBox, mouth: tuple[float, float] | None) -> str:
    if _looks_like_pen(box):
        return "pen"
    if _looks_like_vape(box):
        return "vape"
    if mouth is not None and _near_mouth(box, mouth):
        if _mouth_device_is_vape(box):
            return "vape"
        return "pen"
    if box.label == "handheld":
        return "desk"
    return box.label


def _refine_object_labels(
    objects: list[PresenceBox],
    hands: list[PresenceBox],
    landmarks: Sequence[Any] | None,
    primary_face: PresenceBox | None,
) -> list[PresenceBox]:
    refined: list[PresenceBox] = []
    mouth = _mouth_center(landmarks) if landmarks is not None else None

    for obj in objects:
        if obj.label == "person" and primary_face is not None and _same_face_region(obj, primary_face):
            continue
        label = obj.label
        if label in {"vape", "handheld"} or _looks_like_pen(obj):
            label = _classify_handheld(
                PresenceBox(label, obj.x_min, obj.y_min, obj.x_max, obj.y_max, obj.confidence),
                mouth,
            )
        elif label == "vape" and not _looks_like_vape(obj):
            label = "pen" if _looks_like_pen(obj) else "desk"
        if label == "phone" and not _qualifies_as_phone(obj, hands):
            continue
        refined.append(
            PresenceBox(
                label=label,
                x_min=obj.x_min,
                y_min=obj.y_min,
                x_max=obj.x_max,
                y_max=obj.y_max,
                confidence=obj.confidence,
            )
        )

    if primary_face is not None:
        for index, obj in enumerate(refined):
            if obj.label in {"desk", "book", "laptop", "pen"}:
                continue
            if obj.center[1] < primary_face.y_min + primary_face.height * 0.45:
                if obj.label in {"drink", "food", "phone", "vape"}:
                    continue
                if obj.area < 0.03 and _point_in_box(obj.center[0], obj.center[1], primary_face):
                    refined[index] = PresenceBox(
                        label="glasses",
                        x_min=obj.x_min,
                        y_min=obj.y_min,
                        x_max=obj.x_max,
                        y_max=obj.y_max,
                        confidence=obj.confidence,
                    )

    if mouth is not None:
        for hand in hands:
            if not _near_mouth(hand, mouth, margin=0.10):
                continue
            for index, obj in enumerate(refined):
                if not _boxes_overlap(hand, obj):
                    continue
                if _mouth_device_is_vape(obj):
                    refined[index] = PresenceBox(
                        label="vape",
                        x_min=obj.x_min,
                        y_min=obj.y_min,
                        x_max=obj.x_max,
                        y_max=obj.y_max,
                        confidence=max(obj.confidence, 0.65),
                    )
                elif _looks_like_pen(obj):
                    refined[index] = PresenceBox(
                        label="pen",
                        x_min=obj.x_min,
                        y_min=obj.y_min,
                        x_max=obj.x_max,
                        y_max=obj.y_max,
                        confidence=max(obj.confidence, 0.6),
                    )

    for obj in list(refined):
        if obj.label == "scissors" and obj.center[1] > 0.45:
            refined.append(
                PresenceBox(
                    label="pen",
                    x_min=obj.x_min,
                    y_min=obj.y_min,
                    x_max=obj.x_max,
                    y_max=obj.y_max,
                    confidence=obj.confidence * 0.9,
                )
            )
        if _looks_like_pen(obj) and obj.label not in {"pen", "phone", "user", "person"}:
            refined.append(
                PresenceBox(
                    label="pen",
                    x_min=obj.x_min,
                    y_min=obj.y_min,
                    x_max=obj.x_max,
                    y_max=obj.y_max,
                    confidence=max(0.5, obj.confidence * 0.8),
                )
            )

    deduped: list[PresenceBox] = []
    for box in refined:
        if box.label == "vape" and any(
            kept.label == "pen" and _boxes_overlap(box, kept) for kept in deduped
        ):
            continue
        if any(_boxes_overlap(box, kept) and box.label == kept.label for kept in deduped):
            continue
        deduped.append(box)
    return deduped


class PresenceDetector:
    """Detect labeled people and desk objects without landmarking non-primary subjects."""

    def __init__(
        self,
        *,
        min_face_confidence: float = 0.5,
        min_hand_confidence: float = 0.45,
        min_object_confidence: float = 0.38,
        enable_objects: bool = True,
        enable_hands: bool = True,
    ) -> None:
        self._frame_width = 0
        self._frame_height = 0
        self._face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_face_confidence,
        )
        self._hands = None
        if enable_hands:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=min_hand_confidence,
                min_tracking_confidence=0.4,
            )
        self._object_detector = None
        self._mp_image = None
        if enable_objects:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(model_asset_buffer=object_model_bytes()),
                score_threshold=min_object_confidence,
                max_results=10,
            )
            self._object_detector = vision.ObjectDetector.create_from_options(options)
            self._mp_image = mp.Image

    def close(self) -> None:
        self._face_detection.close()
        if self._hands is not None:
            self._hands.close()
        if self._object_detector is not None:
            self._object_detector.close()

    def detect(self, rgb_frame, landmarks: Sequence[Any] | None = None) -> PresenceFrame:
        self._frame_height, self._frame_width = rgb_frame.shape[:2]
        people = dedupe_face_boxes(self._detect_people(rgb_frame))
        primary_index = select_primary_face(people, landmarks)
        primary_face = people[primary_index] if primary_index is not None else None

        if primary_index is not None:
            relabeled: list[PresenceBox] = []
            for index, box in enumerate(people):
                relabeled.append(
                    PresenceBox(
                        label="user" if index == primary_index else "person",
                        x_min=box.x_min,
                        y_min=box.y_min,
                        x_max=box.x_max,
                        y_max=box.y_max,
                        confidence=box.confidence,
                        is_primary=index == primary_index,
                    )
                )
            people = relabeled
            primary_face = people[primary_index]

        hands = self._detect_hands(rgb_frame)
        objects = self._detect_objects(rgb_frame)
        objects = _refine_object_labels(objects, hands, landmarks, primary_face)

        covered_by_object = {
            hand
            for hand in hands
            if any(
                _boxes_overlap(hand, obj)
                for obj in objects
                if obj.label in {"phone", "vape", "food", "drink", "pen"}
            )
        }
        for hand in hands:
            if hand in covered_by_object:
                continue
            objects = list(objects) + [
                PresenceBox(
                    label="hand",
                    x_min=hand.x_min,
                    y_min=hand.y_min,
                    x_max=hand.x_max,
                    y_max=hand.y_max,
                    confidence=hand.confidence,
                )
            ]

        return PresenceFrame(
            people=tuple(people),
            objects=tuple(objects),
            primary_index=primary_index,
        )

    def _detect_people(self, rgb_frame) -> list[PresenceBox]:
        results = self._face_detection.process(rgb_frame)
        if not results.detections:
            return []

        faces: list[PresenceBox] = []
        for detection in results.detections:
            score = float(detection.score[0]) if detection.score else 0.0
            bbox = detection.location_data.relative_bounding_box
            x_min, y_min, x_max, y_max = _bbox_from_relative(
                bbox,
                self._frame_width,
                self._frame_height,
            )
            faces.append(
                PresenceBox(
                    label="person",
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    confidence=score,
                )
            )
        return faces

    def _detect_hands(self, rgb_frame) -> list[PresenceBox]:
        if self._hands is None:
            return []
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return []

        hands: list[PresenceBox] = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min, x_max, y_max = _bbox_from_landmarks(hand_landmarks.landmark, padding=0.03)
            hands.append(
                PresenceBox(
                    label="hand",
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    confidence=0.7,
                )
            )
        return hands

    def _detect_objects(self, rgb_frame) -> list[PresenceBox]:
        if self._object_detector is None or self._mp_image is None:
            return []

        mp_image = self._mp_image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self._object_detector.detect(mp_image)
        objects: list[PresenceBox] = []
        for detection in results.detections:
            if not detection.categories:
                continue
            category = detection.categories[0]
            label = normalize_object_label(category.category_name)
            bbox = detection.bounding_box
            x_min, y_min, x_max, y_max = _bbox_from_pixels(
                bbox.origin_x,
                bbox.origin_y,
                bbox.width,
                bbox.height,
                self._frame_width,
                self._frame_height,
            )
            objects.append(
                PresenceBox(
                    label=label,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    confidence=float(category.score),
                )
            )
        return objects
