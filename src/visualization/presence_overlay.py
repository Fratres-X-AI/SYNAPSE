"""Draw labeled presence boxes for users, visitors, and desk items."""

from __future__ import annotations

import cv2
import numpy as np

from src.perception.presence_detector import PresenceBox, PresenceFrame, display_label

USER_COLOR = (120, 245, 255)
PERSON_COLOR = (72, 196, 255)
PHONE_COLOR = (255, 210, 72)
VAPE_COLOR = (96, 96, 255)
FOOD_COLOR = (96, 210, 140)
DRINK_COLOR = (255, 170, 96)
PEN_COLOR = (210, 180, 255)
GLASSES_COLOR = (180, 220, 255)
HAND_COLOR = (255, 196, 72)
OBJECT_COLOR = (188, 210, 228)

_HIDDEN_BOX_LABELS = frozenset({"user", "hand"})


def _primary_person(presence: PresenceFrame) -> PresenceBox | None:
    if presence.primary_index is None or presence.primary_index >= len(presence.people):
        return None
    return presence.people[presence.primary_index]


def _overlaps_primary(box: PresenceBox, primary: PresenceBox, *, threshold: float = 0.15) -> bool:
    ix1 = max(box.x_min, primary.x_min)
    iy1 = max(box.y_min, primary.y_min)
    ix2 = min(box.x_max, primary.x_max)
    iy2 = min(box.y_max, primary.y_max)
    if ix1 >= ix2 or iy1 >= iy2:
        return False
    intersection = (ix2 - ix1) * (iy2 - iy1)
    smaller = min(box.area, primary.area)
    if smaller <= 0:
        return False
    return (intersection / smaller) >= threshold


def _same_face_region(a: PresenceBox, b: PresenceBox, *, margin: float = 0.24) -> bool:
    if _overlaps_primary(a, b, threshold=0.15):
        return True
    ax, ay = a.center
    bx, by = b.center
    return abs(ax - bx) < margin and abs(ay - by) < margin


def _should_draw_person_box(person: PresenceBox, primary: PresenceBox | None) -> bool:
    if person.is_primary or person.label in _HIDDEN_BOX_LABELS:
        return False
    if primary is not None and _same_face_region(person, primary):
        return False
    return True


def _should_draw_object_box(obj: PresenceBox, primary: PresenceBox | None) -> bool:
    if obj.label in _HIDDEN_BOX_LABELS:
        return False
    if obj.label == "person" and primary is not None and _same_face_region(obj, primary):
        return False
    return True


def _color_for_label(label: str) -> tuple[int, int, int]:
    return {
        "user": USER_COLOR,
        "person": PERSON_COLOR,
        "phone": PHONE_COLOR,
        "vape": VAPE_COLOR,
        "food": FOOD_COLOR,
        "drink": DRINK_COLOR,
        "pen": PEN_COLOR,
        "glasses": GLASSES_COLOR,
        "hand": HAND_COLOR,
        "laptop": OBJECT_COLOR,
        "book": OBJECT_COLOR,
        "desk": OBJECT_COLOR,
        "scissors": PEN_COLOR,
    }.get(label.lower(), OBJECT_COLOR)


def _box_to_pixels(box: PresenceBox, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(box.x_min * width)
    y1 = int(box.y_min * height)
    x2 = int(box.x_max * width)
    y2 = int(box.y_max * height)
    return x1, y1, x2, y2


def _draw_box(frame: np.ndarray, box: PresenceBox) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _box_to_pixels(box, width, height)
    color = _color_for_label(box.label)
    thickness = 3 if box.is_primary else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    tag = display_label(box.label)
    text_y = max(18, y1 - 8)
    cv2.putText(
        frame,
        tag,
        (x1 + 4, text_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_presence_overlay(frame, presence: PresenceFrame | None) -> np.ndarray:
    """Outline visitors and identified desk items. User and hands stay unboxed."""
    if presence is None:
        return frame

    primary = _primary_person(presence)
    if presence.extra_people > 0:
        for person in presence.people:
            if not _should_draw_person_box(person, primary):
                continue
            _draw_box(frame, person)

    for obj in presence.objects:
        if not _should_draw_object_box(obj, primary):
            continue
        _draw_box(frame, obj)

    return frame


def presence_hud_note(presence: PresenceFrame | None) -> str:
    if presence is None:
        return ""
    labels = [display_label(label) for label in sorted(presence.active_labels())]
    return " | ".join(labels)
