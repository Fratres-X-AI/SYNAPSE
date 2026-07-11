"""Secondary presence detection — labeled people and desk items around the primary user."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import cv2
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
    "food": "Food",
    "drink": "Drink",
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
    "carrot": "",
    "broccoli": "",
    "chair": "",
    "couch": "",
    "potted plant": "",
    "teddy bear": "",
    "toilet": "",
    "tv": "",
    "clock": "",
    "vase": "",
    "bowl": "",
    "fork": "",
    "knife": "",
    "spoon": "",
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

ALLOWED_OBJECT_LABELS = frozenset({"phone"})


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
    detected_hands: tuple[PresenceBox, ...] = ()

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


def _phone_use_posture(hand: PresenceBox, mouth: tuple[float, float]) -> bool:
    """Hand held in front of face for phone use — not a pen-at-lips grip."""
    hx, hy = hand.center
    mx, my = mouth
    if hand.area < 0.009:
        return False
    if hy < 0.24 or hy > 0.82:
        return False
    if _near_mouth(hand, mouth, margin=0.09):
        return False
    if abs(hx - mx) < 0.44 and my - 0.12 < hy < my + 0.42:
        return True
    return False


def _is_hand_masquerading_as_phone(box: PresenceBox, hands: list[PresenceBox]) -> bool:
    """Bare palm mislabeled as phone — not a gripped device."""
    if not hands:
        return False
    for hand in hands:
        if not _boxes_overlap(box, hand):
            continue
        if hand.area <= 0 or box.area <= 0:
            continue
        area_ratio = box.area / hand.area
        ix1 = max(box.x_min, hand.x_min)
        iy1 = max(box.y_min, hand.y_min)
        ix2 = min(box.x_max, hand.x_max)
        iy2 = min(box.y_max, hand.y_max)
        if ix1 >= ix2 or iy1 >= iy2:
            continue
        intersection = (ix2 - ix1) * (iy2 - iy1)
        coverage = intersection / box.area
        if area_ratio >= 0.74 and coverage >= 0.80:
            return True
        if area_ratio >= 0.90 and coverage >= 0.65:
            return True
    return False


def _phone_is_bare_hand(phone: PresenceBox, hands: list[PresenceBox]) -> bool:
    return _is_hand_masquerading_as_phone(phone, hands)


def _overlapping_hands(box: PresenceBox, hands: list[PresenceBox]) -> list[PresenceBox]:
    return [hand for hand in hands if _boxes_overlap(box, hand)]


def _area_ratio_to_hand(box: PresenceBox, hand: PresenceBox) -> float:
    return box.area / max(hand.area, 1e-6)


def _device_slab_on_hand(box: PresenceBox, hand: PresenceBox) -> bool:
    """From live capture: phone slab ~50% of hand area; bare palm ~75%+."""
    ratio = _area_ratio_to_hand(box, hand)
    aspect = box.width / max(box.height, 1e-6)
    if ratio > 0.62:
        return False
    if ratio > 0.48 and aspect > 0.68:
        return False
    return True


def _coco_phone_trusted(box: PresenceBox) -> bool:
    return box.confidence >= 0.48 and _looks_like_phone(box)


def _phone_passes_hand_overlap(box: PresenceBox, hands: list[PresenceBox]) -> bool:
    """Trust confident COCO phones in hand unless the box is a bare palm."""
    overlap_hands = _overlapping_hands(box, hands)
    if not overlap_hands:
        return True
    if _is_hand_masquerading_as_phone(box, hands):
        return False
    if _coco_phone_trusted(box):
        return True
    hand = max(overlap_hands, key=lambda item: item.area)
    return _device_slab_on_hand(box, hand)


def _handheld_false_positive(
    box: PresenceBox,
    hands: list[PresenceBox],
    mouth: tuple[float, float] | None = None,
) -> bool:
    if box.label != "phone":
        return False
    if _blocks_phone_label(box, mouth, hands):
        return True
    if not _looks_like_phone(box):
        return True
    if not _phone_passes_hand_overlap(box, hands):
        return True
    return box.confidence < 0.42


def _strip_false_handheld(
    objects: list[PresenceBox],
    hands: list[PresenceBox],
    mouth: tuple[float, float] | None = None,
) -> list[PresenceBox]:
    if not hands:
        return objects
    return [
        obj
        for obj in objects
        if not _handheld_false_positive(obj, hands, mouth) and not (
            obj.label == "phone" and _phone_is_bare_hand(obj, hands)
        )
    ]


def _drop_hand_shaped_phones(
    objects: list[PresenceBox],
    hands: list[PresenceBox],
) -> list[PresenceBox]:
    return _strip_false_handheld(objects, hands)


def _smooth_phone_box(
    previous: PresenceBox,
    current: PresenceBox,
    *,
    alpha: float = 0.38,
) -> PresenceBox:
    """Ease box jitter so the overlay does not jump frame to frame."""
    blend = max(0.0, min(1.0, alpha))
    keep = 1.0 - blend
    return PresenceBox(
        label=current.label,
        x_min=(previous.x_min * keep) + (current.x_min * blend),
        y_min=(previous.y_min * keep) + (current.y_min * blend),
        x_max=(previous.x_max * keep) + (current.x_max * blend),
        y_max=(previous.y_max * keep) + (current.y_max * blend),
        confidence=max(previous.confidence, current.confidence),
        is_primary=current.is_primary,
    )


def _infer_phone_from_use_posture(
    hands: list[PresenceBox],
    landmarks: Sequence[Any] | None,
) -> PresenceBox | None:
    if landmarks is None or not hands:
        return None
    mouth = _mouth_center(landmarks)
    for hand in hands:
        if not _phone_use_posture(hand, mouth):
            continue
        pad = 0.02
        hand_w = hand.width
        hand_h = hand.height
        cx, cy = hand.center
        slab_w = min(hand_w * 0.58, hand_h * 0.42)
        slab_h = min(hand_h * 0.92, hand_w * 1.35)
        return PresenceBox(
            label="phone",
            x_min=max(0.0, cx - slab_w / 2 - pad),
            y_min=max(0.0, cy - slab_h / 2 - pad),
            x_max=min(1.0, cx + slab_w / 2 + pad),
            y_max=min(1.0, cy + slab_h / 2 + pad),
            confidence=0.58,
        )
    return None


def _presence_has_phone(presence: PresenceFrame) -> bool:
    return any(obj.label == "phone" for obj in presence.objects)


@dataclass
class SmokingEventTracker:
    """Log smoking from vapor plus hand activity at the mouth."""

    watch_seconds: float = 10.0
    sustain_seconds: float = 0.35
    cooldown_seconds: float = 6.0
    vapor_boost: float = 9.0
    heavy_vapor_boost: float = 18.0
    event_hold_seconds: float = 6.0
    _mouth_luma_baseline: float | None = None
    _active_since: float | None = None
    _last_event_at: float = -999.0
    _visible_until: float = -999.0
    _hand_watch_until: float = -999.0

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
        hands = list(presence.detected_hands)
        phones = [obj for obj in presence.objects if obj.label == "phone"]
        from src.perception.phone_gaze import eyes_drawn_to_phone_use

        hand_at_mouth = any(
            _near_mouth(hand, mouth, margin=0.16) for hand in hands
        )
        if hand_at_mouth:
            self._hand_watch_until = max(self._hand_watch_until, now + self.watch_seconds)

        watching = now <= self._hand_watch_until

        vapor_level = self._vapor_level(rgb_frame, mouth)
        if vapor_level is None:
            self._active_since = None
            return ()

        vapor, heavy_vapor = vapor_level
        phone_active = _presence_has_phone(presence) or any(
            _phone_use_posture(hand, mouth) for hand in hands
        )
        mouth_watch = watching and hand_at_mouth
        if phone_active and not heavy_vapor and not mouth_watch:
            self._active_since = None
            return ()
        if not vapor and not heavy_vapor:
            if mouth_watch:
                return ()
            if eyes_drawn_to_phone_use(landmarks, hands, phones):
                self._active_since = None
                return ()
            if phone_active:
                self._active_since = None
                return ()

        if mouth_watch:
            mouth_activity = True
            posture_ok = self._posture_allows_smoking(
                shoulder,
                heavy_vapor=heavy_vapor,
                require_shoulder=True,
            )
        else:
            mouth_activity = hand_at_mouth or heavy_vapor
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
                self._hand_watch_until = -999.0
                return ("smoking",)
        elif not mouth_watch:
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
    def _posture_allows_smoking(
        shoulder: Any | None,
        *,
        heavy_vapor: bool,
        require_shoulder: bool = False,
    ) -> bool:
        if heavy_vapor:
            return True
        if shoulder is None or not getattr(shoulder, "visible", False):
            return not require_shoulder
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
    mapped = DESK_OBJECT_ALIASES.get(lowered)
    if mapped is not None:
        return mapped
    return ""


def _near_mouth(box: PresenceBox, mouth: tuple[float, float], *, margin: float = 0.12) -> bool:
    cx, cy = box.center
    return abs(cx - mouth[0]) < margin and abs(cy - mouth[1]) < margin


def _looks_like_pen(box: PresenceBox) -> bool:
    aspect = box.width / max(box.height, 1e-6)
    if 0.22 <= aspect <= 0.55 and box.area >= 0.035:
        return False
    thin_horizontal = aspect > 1.8
    thin_vertical = aspect < 0.35 and box.area < 0.025
    thin = thin_horizontal or thin_vertical
    desk_level = box.center[1] > 0.36
    small_enough = 0.001 < box.area < 0.12
    return thin and desk_level and small_enough


def _mouth_device_blocks_phone(box: PresenceBox) -> bool:
    """Small blocky mouth objects (pods) — not a phone held in front of the face."""
    if _looks_like_pen(box):
        return False
    if _looks_like_phone(box) and box.area >= 0.04:
        return False
    aspect = box.width / max(box.height, 1e-6)
    blocky = 0.4 <= aspect <= 2.5
    pod_size = 0.0015 < box.area < 0.055
    return blocky and pod_size


def _blocks_phone_label(
    box: PresenceBox,
    mouth: tuple[float, float] | None,
    hands: list[PresenceBox] | None = None,
) -> bool:
    """Pods at the lips must not be labeled phone."""
    if _looks_like_phone(box) and box.area >= 0.04:
        return False
    if mouth is None:
        return False
    if _near_mouth(box, mouth, margin=0.15) and _mouth_device_blocks_phone(box):
        return True
    if not hands:
        return False
    mouth_hands = [hand for hand in hands if _near_mouth(hand, mouth, margin=0.12)]
    if not mouth_hands:
        return False
    return any(_boxes_overlap(box, hand) for hand in mouth_hands) and _mouth_device_blocks_phone(box)


def _looks_like_phone(box: PresenceBox) -> bool:
    """Handheld phone slab — portrait, landscape, or phone-in-front-of-face."""
    aspect = box.width / max(box.height, 1e-6)
    if aspect > 1.8 or (aspect < 0.35 and box.area < 0.014):
        return False
    portrait = 0.32 <= aspect <= 0.72
    in_front_of_face = portrait and 0.38 <= box.height <= 0.65 and box.area < 0.22
    slab = 0.22 <= aspect <= 3.0
    max_area = 0.35 if in_front_of_face else (0.22 if portrait else 0.12)
    max_span = 0.75 if in_front_of_face else 0.48
    compact = 0.0010 < box.area < max_area
    not_wall_sized = max(box.width, box.height) < max_span
    return slab and compact and not_wall_sized


def _near_or_overlaps_hand(box: PresenceBox, hands: list[PresenceBox], *, margin: float = 0.12) -> bool:
    if not hands:
        return False
    if any(_boxes_overlap(box, hand) for hand in hands):
        return True
    cx, cy = box.center
    for hand in hands:
        hx, hy = hand.center
        if abs(cx - hx) < margin and abs(cy - hy) < margin:
            return True
        expanded = PresenceBox(
            "hand",
            max(0.0, hand.x_min - margin * 0.6),
            max(0.0, hand.y_min - margin * 0.6),
            min(1.0, hand.x_max + margin * 0.6),
            min(1.0, hand.y_max + margin * 0.6),
            hand.confidence,
        )
        if _point_in_box(cx, cy, expanded):
            return True
    return False


def _phone_in_active_use(
    box: PresenceBox,
    hands: list[PresenceBox],
    mouth: tuple[float, float] | None,
) -> bool:
    """Phone must be in hand or held in front of the face — not wall clutter."""
    if hands and _near_or_overlaps_hand(box, hands):
        return True
    cx, cy = box.center
    if mouth is not None:
        mx, my = mouth
        near_face = abs(cx - mx) < 0.38 and (my - 0.18) < cy < (my + 0.52)
        if near_face and _looks_like_phone(box):
            return True
    return 0.28 < cx < 0.72 and cy > 0.48 and _looks_like_phone(box)


def _qualifies_as_phone(
    box: PresenceBox,
    hands: list[PresenceBox],
    mouth: tuple[float, float] | None = None,
) -> bool:
    if _blocks_phone_label(box, mouth, hands):
        return False
    if not _looks_like_phone(box):
        return False
    if _looks_like_background_clutter(box):
        return False
    if not _phone_in_active_use(box, hands, mouth):
        return False
    in_hand = _near_or_overlaps_hand(box, hands)
    if in_hand:
        if not _phone_passes_hand_overlap(box, hands):
            return False
        return box.confidence >= 0.48
    if not _looks_like_background_clutter(box):
        return box.confidence >= 0.45
    return False


def _looks_like_background_clutter(box: PresenceBox) -> bool:
    """Jerseys, backpacks, and wall hangings — not a visitor."""
    aspect = box.width / max(box.height, 1e-6)
    cx, cy = box.center
    if _looks_like_phone(box) and box.area < 0.18:
        return False
    if box.area > 0.09:
        return True
    if box.height > 0.36 and box.area > 0.045:
        return True
    if box.area > 0.055 and aspect > 1.15:
        return True
    edge_zone = cx < 0.20 or cx > 0.74
    if edge_zone and cy < 0.58 and box.area > 0.03:
        return True
    return False


def _qualifies_as_extra_person_face(
    box: PresenceBox,
    primary: PresenceBox | None,
) -> bool:
    if primary is not None and _same_face_region(box, primary):
        return False
    if _looks_like_background_clutter(box):
        return False
    aspect = box.width / max(box.height, 1e-6)
    if not (0.48 <= aspect <= 1.55):
        return False
    if not (0.006 < box.area < 0.13):
        return False
    cx, cy = box.center
    if cy < 0.34 and box.confidence < 0.74:
        return False
    if (cx < 0.16 or cx > 0.80) and box.confidence < 0.70:
        return False
    return box.confidence >= 0.54


def _qualifies_as_coco_person(box: PresenceBox, primary_face: PresenceBox | None) -> bool:
    if primary_face is not None and _same_face_region(box, primary_face):
        return False
    if _looks_like_background_clutter(box):
        return False
    aspect = box.width / max(box.height, 1e-6)
    if box.area > 0.07:
        return False
    if box.height > 0.34 and aspect < 0.85:
        return False
    cx, _ = box.center
    if (cx < 0.18 or cx > 0.78) and box.confidence < 0.62:
        return False
    return box.confidence >= 0.50


def _classify_handheld(
    box: PresenceBox,
    mouth: tuple[float, float] | None,
    hands: list[PresenceBox] | None = None,
) -> str:
    if _blocks_phone_label(box, mouth, hands):
        return ""
    at_mouth = mouth is not None and _near_mouth(box, mouth, margin=0.14)
    if hands and not at_mouth:
        for hand in hands:
            if not _near_or_overlaps_hand(box, [hand]):
                continue
            if not _device_slab_on_hand(box, hand):
                return ""
    if hands and _near_or_overlaps_hand(box, hands) and _looks_like_phone(box):
        if at_mouth:
            return ""
        return "phone"
    if box.label == "handheld":
        return "desk"
    return "phone" if box.label == "phone" else ""


def _boxes_distinct_enough(a: PresenceBox, b: PresenceBox, *, margin: float = 0.06) -> bool:
    ax, ay = a.center
    bx, by = b.center
    return abs(ax - bx) >= margin or abs(ay - by) >= margin


def _keep_overlapping_handheld_pair(a: PresenceBox, b: PresenceBox) -> bool:
    """Keep distinct phone boxes when overlap is real but centers differ."""
    if not _boxes_overlap(a, b):
        return True
    if {a.label, b.label} != {"phone"}:
        return True
    return _boxes_distinct_enough(a, b, margin=0.08)


def _phones_compatible(a: PresenceBox, b: PresenceBox, *, margin: float = 0.12) -> bool:
    if _boxes_overlap(a, b):
        return True
    ax, ay = a.center
    bx, by = b.center
    return abs(ax - bx) < margin and abs(ay - by) < margin


def _apply_phone_gaze_split(
    objects: list[PresenceBox],
    hands: list[PresenceBox],
    landmarks: Sequence[Any],
) -> list[PresenceBox]:
    """Gaze confirms phone use but never turns a bare hand into a phone box."""
    phone_boxes = [obj for obj in objects if obj.label == "phone"]
    if not phone_boxes:
        return objects
    from src.perception.phone_gaze import eyes_drawn_to_phone_use

    if eyes_drawn_to_phone_use(landmarks, hands, phone_boxes):
        return objects
    return objects


def _filter_allowed_objects(objects: list[PresenceBox]) -> list[PresenceBox]:
    return [obj for obj in objects if obj.label in ALLOWED_OBJECT_LABELS]


def _refine_object_labels(
    objects: list[PresenceBox],
    hands: list[PresenceBox],
    landmarks: Sequence[Any] | None,
    primary_face: PresenceBox | None,
) -> list[PresenceBox]:
    refined: list[PresenceBox] = []
    mouth = _mouth_center(landmarks) if landmarks is not None else None

    for obj in objects:
        if obj.label == "person":
            if primary_face is not None and _same_face_region(obj, primary_face):
                continue
            if not _qualifies_as_coco_person(obj, primary_face):
                continue
        label = obj.label
        if not label:
            continue
        if label == "phone":
            if _blocks_phone_label(obj, mouth, hands):
                continue
            if not _qualifies_as_phone(obj, hands, mouth):
                continue
        elif label == "handheld":
            label = _classify_handheld(
                PresenceBox(label, obj.x_min, obj.y_min, obj.x_max, obj.y_max, obj.confidence),
                mouth,
                hands,
            )
            if not label:
                continue
            if label == "phone" and not _qualifies_as_phone(
                PresenceBox(label, obj.x_min, obj.y_min, obj.x_max, obj.y_max, obj.confidence),
                hands,
                mouth,
            ):
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
            if obj.label in {"desk", "book", "laptop"}:
                continue
            if obj.center[1] < primary_face.y_min + primary_face.height * 0.45:
                if obj.label in {"drink", "food", "phone"}:
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

    for hand in hands:
        has_phone = any(
            obj.label == "phone" and _near_or_overlaps_hand(obj, [hand]) for obj in refined
        )
        if has_phone:
            continue
        for obj in objects:
            if obj.label not in {"phone", "cell phone", "handheld", "remote"}:
                continue
            candidate = PresenceBox(
                obj.label,
                obj.x_min,
                obj.y_min,
                obj.x_max,
                obj.y_max,
                obj.confidence,
            )
            if not _near_or_overlaps_hand(candidate, [hand]) or not _looks_like_phone(candidate):
                continue
            if _blocks_phone_label(candidate, mouth, hands):
                continue
            if not _phone_passes_hand_overlap(candidate, hands) or candidate.confidence < 0.42:
                continue
            if _looks_like_background_clutter(candidate):
                continue
            refined.append(
                PresenceBox(
                    label="phone",
                    x_min=obj.x_min,
                    y_min=obj.y_min,
                    x_max=obj.x_max,
                    y_max=obj.y_max,
                    confidence=max(obj.confidence, 0.55),
                )
            )
            break

    deduped: list[PresenceBox] = []
    for box in refined:
        drop = False
        for kept in deduped:
            if box.label == kept.label and _boxes_overlap(box, kept):
                drop = True
                break
            if not _keep_overlapping_handheld_pair(box, kept):
                if box.confidence <= kept.confidence:
                    drop = True
                    break
                deduped.remove(kept)
        if drop:
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
        self._phone_hold_until = -999.0
        self._phone_hold_seconds = 2.2
        self._last_phone_box: PresenceBox | None = None
        if enable_objects:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(model_asset_buffer=object_model_bytes()),
                score_threshold=min_object_confidence,
                max_results=15,
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
        if primary_index is not None:
            primary_ref = people[primary_index]
            kept: list[PresenceBox] = []
            kept_primary = 0
            for index, box in enumerate(people):
                if index == primary_index:
                    kept.append(box)
                    kept_primary = len(kept) - 1
                elif _qualifies_as_extra_person_face(box, primary_ref):
                    kept.append(box)
            people = kept
            primary_index = kept_primary if kept else None
        else:
            people = [
                box
                for box in people
                if not _looks_like_background_clutter(box) and box.confidence >= 0.54
            ]
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
        mouth = _mouth_center(landmarks) if landmarks is not None else None
        objects = _refine_object_labels(objects, hands, landmarks, primary_face)
        objects = _strip_false_handheld(objects, hands, mouth)
        if landmarks is not None:
            objects = _apply_phone_gaze_split(objects, hands, landmarks)

        objects = _strip_false_handheld(list(objects), hands, mouth)
        objects = self._stick_phone_label(list(objects), hands, mouth, time.monotonic())
        objects = _filter_allowed_objects(objects)

        return PresenceFrame(
            people=tuple(people),
            objects=tuple(objects),
            primary_index=primary_index,
            detected_hands=tuple(hands),
        )

    def _choose_phone_box(
        self,
        phone_boxes: list[PresenceBox],
        hands: list[PresenceBox],
        mouth: tuple[float, float] | None = None,
    ) -> PresenceBox | None:
        if not phone_boxes:
            return None
        detector_phones = [
            box
            for box in phone_boxes
            if box.confidence >= 0.42
            and not _handheld_false_positive(box, hands, mouth)
            and not _blocks_phone_label(box, mouth, hands)
        ]
        if not detector_phones:
            return None
        if self._last_phone_box is None:
            return max(detector_phones, key=lambda box: box.confidence)
        compatible = [
            box for box in detector_phones if _phones_compatible(box, self._last_phone_box)
        ]
        if compatible:
            return max(compatible, key=lambda box: box.confidence)
        return None

    def _single_phone_objects(
        self,
        objects: list[PresenceBox],
        phone: PresenceBox,
    ) -> list[PresenceBox]:
        return [obj for obj in objects if obj.label != "phone"] + [phone]

    def _stick_phone_label(
        self,
        objects: list[PresenceBox],
        hands: list[PresenceBox],
        mouth: tuple[float, float] | None = None,
        now: float | None = None,
    ) -> list[PresenceBox]:
        tick = time.monotonic() if now is None else now
        phone_boxes = [obj for obj in objects if obj.label == "phone"]
        if phone_boxes:
            chosen = self._choose_phone_box(phone_boxes, hands, mouth)
            if chosen is not None:
                if (
                    self._last_phone_box is not None
                    and _phones_compatible(chosen, self._last_phone_box)
                ):
                    chosen = _smooth_phone_box(self._last_phone_box, chosen)
                self._last_phone_box = chosen
                self._phone_hold_until = tick + self._phone_hold_seconds
                return self._single_phone_objects(objects, chosen)
        if (
            self._last_phone_box is not None
            and tick <= self._phone_hold_until
            and not _handheld_false_positive(self._last_phone_box, hands, mouth)
            and _phone_in_active_use(self._last_phone_box, hands, mouth)
        ):
            return self._single_phone_objects(objects, self._last_phone_box)
        self._last_phone_box = None
        return [obj for obj in objects if obj.label != "phone"]

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

    def _detail_scan_hands(
        self,
        rgb_frame,
        hands: list[PresenceBox],
        objects: list[PresenceBox],
    ) -> list[PresenceBox]:
        if not hands or self._object_detector is None:
            return objects
        merged = list(objects)
        seen: set[tuple[str, int, int, int, int]] = set()
        for hand in hands:
            pad = 0.08
            region = (
                max(0.0, hand.x_min - pad),
                max(0.0, hand.y_min - pad),
                min(1.0, hand.x_max + pad),
                min(1.0, hand.y_max + pad),
            )
            for obj in self._detect_objects(rgb_frame, region=region, detail_boost=0.04):
                key = (
                    obj.label,
                    int(obj.x_min * 1000),
                    int(obj.y_min * 1000),
                    int(obj.x_max * 1000),
                    int(obj.y_max * 1000),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(obj)
        return merged

    def _detect_objects(
        self,
        rgb_frame,
        *,
        region: tuple[float, float, float, float] | None = None,
        detail_boost: float = 0.0,
    ) -> list[PresenceBox]:
        if self._object_detector is None or self._mp_image is None:
            return []

        frame = rgb_frame
        offset_x = 0.0
        offset_y = 0.0
        span_x = 1.0
        span_y = 1.0
        if region is not None:
            x_min, y_min, x_max, y_max = region
            px1 = int(_clamp01(x_min) * self._frame_width)
            py1 = int(_clamp01(y_min) * self._frame_height)
            px2 = int(_clamp01(x_max) * self._frame_width)
            py2 = int(_clamp01(y_max) * self._frame_height)
            if px2 <= px1 or py2 <= py1:
                return []
            frame = rgb_frame[py1:py2, px1:px2]
            if frame.size == 0:
                return []
            crop_h, crop_w = frame.shape[:2]
            target = 160
            scale = max(target / max(crop_w, 1), target / max(crop_h, 1), 1.0)
            if scale > 1.0:
                frame = cv2.resize(
                    frame,
                    (int(crop_w * scale), int(crop_h * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
            offset_x = x_min
            offset_y = y_min
            span_x = x_max - x_min
            span_y = y_max - y_min

        mp_image = self._mp_image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame),
        )
        results = self._object_detector.detect(mp_image)
        objects: list[PresenceBox] = []
        crop_h, crop_w = frame.shape[:2]
        for detection in results.detections:
            if not detection.categories:
                continue
            category = detection.categories[0]
            label = normalize_object_label(category.category_name)
            if not label:
                continue
            bbox = detection.bounding_box
            if region is None:
                x_min, y_min, x_max, y_max = _bbox_from_pixels(
                    bbox.origin_x,
                    bbox.origin_y,
                    bbox.width,
                    bbox.height,
                    self._frame_width,
                    self._frame_height,
                )
            else:
                rel_x_min = bbox.origin_x / max(crop_w, 1)
                rel_y_min = bbox.origin_y / max(crop_h, 1)
                rel_x_max = (bbox.origin_x + bbox.width) / max(crop_w, 1)
                rel_y_max = (bbox.origin_y + bbox.height) / max(crop_h, 1)
                x_min = _clamp01(offset_x + rel_x_min * span_x)
                y_min = _clamp01(offset_y + rel_y_min * span_y)
                x_max = _clamp01(offset_x + rel_x_max * span_x)
                y_max = _clamp01(offset_y + rel_y_max * span_y)
            score = float(category.score) + detail_boost
            objects.append(
                PresenceBox(
                    label=label,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    confidence=min(score, 0.99),
                )
            )
        return objects
