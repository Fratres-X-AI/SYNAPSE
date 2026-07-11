"""Detect phone use from iris gaze toward hands or phone objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from src.perception.presence_detector import PresenceBox

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LEFT_IRIS = 468
RIGHT_IRIS = 473
_MOUTH_INDICES = (13, 14, 78, 308)


def _point(landmark) -> tuple[float, float]:
    return float(landmark.x), float(landmark.y)


def _mouth_center(landmarks: Sequence[Any]) -> tuple[float, float]:
    xs = [landmarks[index].x for index in _MOUTH_INDICES]
    ys = [landmarks[index].y for index in _MOUTH_INDICES]
    return float(np.mean(xs)), float(np.mean(ys))


def _eye_gaze_offset(
    landmarks: Sequence[Any],
    eye_indices: tuple[int, ...],
    iris_index: int,
) -> tuple[float, float]:
    outer, top_a, top_b, inner, bottom_a, bottom_b = (
        _point(landmarks[index]) for index in eye_indices
    )
    iris = _point(landmarks[iris_index])
    eye_width = inner[0] - outer[0]
    eye_height = ((bottom_a[1] + bottom_b[1]) / 2.0) - ((top_a[1] + top_b[1]) / 2.0)
    if eye_width == 0 or eye_height == 0:
        return 0.0, 0.0
    gaze_x = (iris[0] - outer[0]) / eye_width - 0.5
    gaze_y = (iris[1] - ((top_a[1] + top_b[1]) / 2.0)) / eye_height - 0.5
    return float(gaze_x), float(gaze_y)


def estimate_gaze_vector(landmarks: Sequence[Any]) -> tuple[float, float]:
    left_x, left_y = _eye_gaze_offset(landmarks, LEFT_EYE, LEFT_IRIS)
    right_x, right_y = _eye_gaze_offset(landmarks, RIGHT_EYE, RIGHT_IRIS)
    return (
        float(np.clip((left_x + right_x) / 2.0, -0.5, 0.5)),
        float(np.clip((left_y + right_y) / 2.0, -0.5, 0.5)),
    )


def gaze_ray_point(landmarks: Sequence[Any], gaze_x: float, gaze_y: float) -> tuple[float, float]:
    """Project iris gaze into normalized frame coordinates."""
    nose = _point(landmarks[1])
    return (
        float(np.clip(nose[0] + gaze_x * 1.75, 0.0, 1.0)),
        float(np.clip(nose[1] + gaze_y * 1.55, 0.0, 1.0)),
    )


def _point_near_box(
    x: float,
    y: float,
    box: "PresenceBox",
    *,
    margin: float,
) -> bool:
    return (
        box.x_min - margin <= x <= box.x_max + margin
        and box.y_min - margin <= y <= box.y_max + margin
    )


def _near_mouth_point(x: float, y: float, mouth: tuple[float, float], *, margin: float) -> bool:
    return abs(x - mouth[0]) < margin and abs(y - mouth[1]) < margin


def _hand_at_mouth(hand: "PresenceBox", mouth: tuple[float, float]) -> bool:
    hx, hy = hand.center
    return abs(hx - mouth[0]) < 0.11 and abs(hy - mouth[1]) < 0.11


def _near_box_center(
    box: "PresenceBox",
    x: float,
    y: float,
    *,
    margin: float,
) -> bool:
    cx, cy = box.center
    return abs(cx - x) < margin and abs(cy - y) < margin


def hand_holds_phone(hand: "PresenceBox", phones: list["PresenceBox"]) -> bool:
    if not phones:
        return False
    for phone in phones:
        if _boxes_overlap(hand, phone):
            return True
        if _near_box_center(phone, hand.center[0], hand.center[1], margin=0.12):
            return True
    return False


def _boxes_overlap(a: "PresenceBox", b: "PresenceBox") -> bool:
    return not (a.x_max < b.x_min or b.x_max < a.x_min or a.y_max < b.y_min or b.y_max < a.y_min)


def _gazed_hand_for_phone(
    hands: list["PresenceBox"],
    landmarks: Sequence[Any],
    gaze_x: float,
    gaze_y: float,
    mouth: tuple[float, float],
    *,
    phones: list["PresenceBox"] | None = None,
) -> "PresenceBox | None":
    gaze_x_pt, gaze_y_pt = gaze_ray_point(landmarks, gaze_x, gaze_y)
    phone_boxes = phones or []

    phone_hands = [
        hand
        for hand in hands
        if hand_holds_phone(hand, phone_boxes) and not _hand_at_mouth(hand, mouth)
    ]
    if phone_hands:
        return min(
            phone_hands,
            key=lambda hand: float(
                np.hypot(gaze_x_pt - hand.center[0], gaze_y_pt - hand.center[1])
            ),
        )

    best: PresenceBox | None = None
    best_score = -1.0
    for hand in hands:
        if _hand_at_mouth(hand, mouth):
            continue
        margin = 0.24
        if not _point_near_box(gaze_x_pt, gaze_y_pt, hand, margin=margin):
            cx, cy = hand.center
            if not (abs(gaze_x_pt - cx) < 0.40 and gaze_y_pt <= cy + 0.18):
                continue
        score = 1.0 - float(np.hypot(gaze_x_pt - hand.center[0], gaze_y_pt - hand.center[1]))
        if score > best_score:
            best_score = score
            best = hand
    return best


gazed_hand_for_phone = _gazed_hand_for_phone


def eyes_drawn_to_phone_use(
    landmarks: Sequence[Any],
    hands: list["PresenceBox"],
    phones: list["PresenceBox"],
) -> bool:
    if landmarks is None or len(landmarks) < 474:
        return False

    gaze_x, gaze_y = estimate_gaze_vector(landmarks)
    gaze_x_pt, gaze_y_pt = gaze_ray_point(landmarks, gaze_x, gaze_y)
    mouth = _mouth_center(landmarks)

    if _near_mouth_point(gaze_x_pt, gaze_y_pt, mouth, margin=0.10):
        return False

    for phone in phones:
        if _point_near_box(gaze_x_pt, gaze_y_pt, phone, margin=0.20):
            return True

    if _gazed_hand_for_phone(hands, landmarks, gaze_x, gaze_y, mouth) is not None:
        return True

    return False


def phone_box_from_hand(hand: PresenceBox, *, confidence: float = 0.62) -> PresenceBox:
    from src.perception.presence_detector import PresenceBox as Box

    pad = 0.03
    return Box(
        label="phone",
        x_min=max(0.0, hand.x_min - pad),
        y_min=max(0.0, hand.y_min - pad),
        x_max=min(1.0, hand.x_max + pad),
        y_max=min(1.0, hand.y_max + pad),
        confidence=confidence,
    )
