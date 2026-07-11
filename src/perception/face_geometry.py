"""Derived face landmarks and elite helmet shell geometry from MediaPipe Face Mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# MediaPipe indices shared with StateEstimator / EmotionEstimator.
_NOSE_TIP = 1
_FOREHEAD = 10
_CHIN = 152
_LEFT_FACE = 234
_RIGHT_FACE = 454
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)
_LEFT_BROW = (293, 334, 336)
_RIGHT_BROW = (63, 105, 107)

# Key oval anchors the helmet shell locks onto (Lens Studio-style hierarchy).
_SHELL_BRIDGE_TARGETS = (10, 338, 297, 332, 152, 234, 454, 127, 356)

_RIM_COUNT = 32
_CROWN_COUNT = 5
_EAR_POINTS_PER_SIDE = 3


@dataclass(frozen=True)
class HairlineGeometry:
    """Computed hairline point and distances to stable facial anchors."""

    x: float
    y: float
    eye_to_hairline: float
    chin_to_hairline: float
    brow_to_hairline: float
    left_ear_to_hairline: float
    right_ear_to_hairline: float
    forehead_to_hairline: float
    face_height: float
    face_width: float
    left_temple_x: float
    right_temple_x: float

    @property
    def point(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def left_temple(self) -> tuple[float, float]:
        return self.left_temple_x, self.y

    @property
    def right_temple(self) -> tuple[float, float]:
        return self.right_temple_x, self.y


@dataclass(frozen=True)
class PeripheralLandmarkMesh:
    """Elite helmet shell: rim + crown + optional ears, bridged to MediaPipe anchors."""

    points: tuple[tuple[float, float], ...]
    connections: tuple[tuple[int, int], ...]
    shell_lines: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()
    mediapipe_bridges: tuple[tuple[int, int], ...] = ()
    crown_offset: int = 0
    left_ear_offset: int = 0
    right_ear_offset: int = 0
    left_ear_count: int = 0
    right_ear_count: int = 0

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def rim_count(self) -> int:
        return self.crown_offset


def _add_edge(edges: set[tuple[int, int]], a: int, b: int) -> None:
    if a == b:
        return
    edges.add((a, b) if a < b else (b, a))


def _chain_edges(count: int, offset: int, edges: set[tuple[int, int]]) -> None:
    for index in range(count - 1):
        _add_edge(edges, offset + index, offset + index + 1)


def _ring_edges(count: int, offset: int, edges: set[tuple[int, int]]) -> None:
    for index in range(count):
        _add_edge(edges, offset + index, offset + ((index + 1) % count))


def _ellipse_point(
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    angle_deg: float,
) -> tuple[float, float]:
    radians = np.deg2rad(angle_deg)
    return (
        center[0] + radius_x * float(np.cos(radians)),
        center[1] + radius_y * float(np.sin(radians)),
    )


def _mean_xy(landmarks: Sequence[Any], indices: tuple[int, ...]) -> tuple[float, float]:
    xs = [landmarks[index].x for index in indices]
    ys = [landmarks[index].y for index in indices]
    return float(np.mean(xs)), float(np.mean(ys))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _normalized_head_yaw(landmarks: Sequence[Any]) -> float:
    nose = landmarks[_NOSE_TIP]
    left_face = landmarks[_LEFT_FACE]
    right_face = landmarks[_RIGHT_FACE]
    face_width = max(right_face.x - left_face.x, 1e-6)
    face_center_x = (left_face.x + right_face.x) / 2.0
    return float(np.clip((nose.x - face_center_x) / face_width, -1.0, 1.0))


def _ear_side_visibility(yaw: float, side: str) -> float:
    turn = max(0.0, -yaw - 0.12) if side == "left" else max(0.0, yaw - 0.12)
    return float(np.clip(turn * 1.1, 0.0, 1.0))


def _face_shell_axes(
    landmarks: Sequence[Any],
    hairline: HairlineGeometry,
) -> tuple[tuple[float, float], float, float]:
    left_face = landmarks[_LEFT_FACE]
    right_face = landmarks[_RIGHT_FACE]
    chin = landmarks[_CHIN]
    center = (
        (left_face.x + right_face.x) / 2.0,
        (hairline.y + chin.y) / 2.0 + hairline.face_width * 0.01,
    )
    radius_x = max(hairline.face_width * 0.58, 1e-6)
    radius_y = max(hairline.chin_to_hairline * 0.56, 1e-6)
    return center, radius_x, radius_y


def _rim_angles() -> np.ndarray:
    return np.linspace(0.0, 360.0, _RIM_COUNT, endpoint=False)


def _rim_point(
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    angle_deg: float,
    hairline: HairlineGeometry,
) -> tuple[float, float]:
    x, y = _ellipse_point(center, radius_x, radius_y, angle_deg)
    # Keep the shell face-hugging and smooth: slightly tuck the side walls,
    # and keep the top/bottom from becoming a boxy cage.
    if 115.0 <= angle_deg <= 245.0 or angle_deg >= 295.0 or angle_deg <= 65.0:
        x = center[0] + (x - center[0]) * 0.94
    if 225.0 <= angle_deg <= 315.0:
        y -= hairline.face_width * 0.012
    if 55.0 <= angle_deg <= 125.0:
        y -= hairline.face_width * 0.01
    return x, y


def _helmet_rim(
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    hairline: HairlineGeometry,
    yaw: float,
) -> tuple[list[tuple[float, float]], int, int]:
    points = [
        _rim_point(center, radius_x, radius_y, float(angle), hairline)
        for angle in _rim_angles()
    ]
    face_width = hairline.face_width
    adjusted: list[tuple[float, float]] = []
    for x, y in points:
        if x <= center[0]:
            x -= face_width * max(0.0, -yaw) * 0.07
        if x >= center[0]:
            x += face_width * max(0.0, yaw) * 0.07
        adjusted.append((x, y))
    left_rear = int(_RIM_COUNT * 0.58)
    right_rear = int(_RIM_COUNT * 0.42)
    return adjusted, left_rear, right_rear


def _crown_band(
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    hairline: HairlineGeometry,
) -> list[tuple[float, float]]:
    return [
        _rim_point(center, radius_x * 0.78, radius_y * 0.84, float(angle), hairline)
        for angle in np.linspace(235.0, 305.0, _CROWN_COUNT)
    ]


def _rear_shell_lines(
    left_anchor: tuple[float, float],
    right_anchor: tuple[float, float],
    hairline: HairlineGeometry,
    yaw: float,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    if abs(yaw) < 0.26:
        return ()
    apex = (hairline.x, min(left_anchor[1], right_anchor[1]) - hairline.face_height * 0.06)
    left_mid = (left_anchor[0] * 0.86 + apex[0] * 0.14, left_anchor[1] * 0.62 + apex[1] * 0.38)
    right_mid = (right_anchor[0] * 0.86 + apex[0] * 0.14, right_anchor[1] * 0.62 + apex[1] * 0.38)
    return (
        (right_anchor, right_mid),
        (right_mid, apex),
        (apex, left_mid),
        (left_mid, left_anchor),
    )


def _compute_ear_wing(
    landmarks: Sequence[Any],
    hairline: HairlineGeometry,
    side: str,
    yaw: float,
) -> list[tuple[float, float]]:
    visibility = _ear_side_visibility(yaw, side)
    if visibility < 0.18:
        return []

    width = hairline.face_width
    extent = width * (0.04 + 0.13 * visibility)
    if side == "left":
        cheek = landmarks[_LEFT_FACE]
        upper = landmarks[127]
        lower = landmarks[162]
        sign = -1.0
    else:
        cheek = landmarks[_RIGHT_FACE]
        upper = landmarks[356]
        lower = landmarks[323]
        sign = 1.0

    return [
        (upper.x + sign * extent * 0.42, upper.y - width * 0.012),
        (cheek.x + sign * extent * 0.48, cheek.y),
        (lower.x + sign * extent * 0.30, lower.y + width * 0.025),
    ]


def compute_peripheral_mesh(landmarks: Sequence[Any]) -> PeripheralLandmarkMesh:
    """Elite helmet shell: open rim, crown band, profile ears, clean rear closure."""
    hairline = compute_hairline(landmarks)
    center, radius_x, radius_y = _face_shell_axes(landmarks, hairline)
    yaw = _normalized_head_yaw(landmarks)

    rim, left_rear_index, right_rear_index = _helmet_rim(center, radius_x, radius_y, hairline, yaw)
    crown = _crown_band(center, radius_x, radius_y, hairline)
    left_ear = _compute_ear_wing(landmarks, hairline, "left", yaw)
    right_ear = _compute_ear_wing(landmarks, hairline, "right", yaw)

    points = rim + crown + left_ear + right_ear
    rim_offset = 0
    crown_offset = len(rim)
    left_ear_offset = crown_offset + len(crown)
    right_ear_offset = left_ear_offset + len(left_ear)

    edges: set[tuple[int, int]] = set()
    bridges: list[tuple[int, int]] = []

    _ring_edges(len(rim), rim_offset, edges)
    _chain_edges(len(crown), crown_offset, edges)

    for index, crown_point in enumerate(crown):
        rim_index = min(
            range(len(rim)),
            key=lambda candidate: abs(rim[candidate][0] - crown_point[0]),
        )
        _add_edge(edges, crown_offset + index, rim_offset + rim_index)

    if left_ear:
        _chain_edges(len(left_ear), left_ear_offset, edges)
        bridges.append((left_ear_offset + 1, _LEFT_FACE))
        bridges.append((left_ear_offset + 2, 127))

    if right_ear:
        _chain_edges(len(right_ear), right_ear_offset, edges)
        bridges.append((right_ear_offset + 1, _RIGHT_FACE))
        bridges.append((right_ear_offset + 2, 356))

    for rim_index, rim_point in enumerate(rim):
        if rim_index % 5 != 0:
            continue
        target = min(
            _SHELL_BRIDGE_TARGETS,
            key=lambda landmark_index: _distance(
                rim_point,
                (landmarks[landmark_index].x, landmarks[landmark_index].y),
            ),
        )
        bridges.append((rim_offset + rim_index, target))

    return PeripheralLandmarkMesh(
        points=tuple(points),
        connections=tuple(sorted(edges)),
        shell_lines=_rear_shell_lines(rim[left_rear_index], rim[right_rear_index], hairline, yaw),
        mediapipe_bridges=tuple(bridges),
        crown_offset=crown_offset,
        left_ear_offset=left_ear_offset,
        right_ear_offset=right_ear_offset,
        left_ear_count=len(left_ear),
        right_ear_count=len(right_ear),
    )


def compute_hairline(landmarks: Sequence[Any]) -> HairlineGeometry:
    """
    Estimate the hairline from eye, brow, ear, forehead, and chin geometry.

    MediaPipe landmark 10 sits on the upper forehead contour, not the true
    hairline. We project upward using the eye-to-forehead segment and clamp
    using classic eye-to-chin proportions so the point tracks with head pose.
    """
    forehead = landmarks[_FOREHEAD]
    chin = landmarks[_CHIN]
    left_face = landmarks[_LEFT_FACE]
    right_face = landmarks[_RIGHT_FACE]

    left_eye = _mean_xy(landmarks, _LEFT_EYE)
    right_eye = _mean_xy(landmarks, _RIGHT_EYE)
    eye_mid = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)

    left_brow = _mean_xy(landmarks, _LEFT_BROW)
    right_brow = _mean_xy(landmarks, _RIGHT_BROW)
    brow_mid = ((left_brow[0] + right_brow[0]) / 2.0, (left_brow[1] + right_brow[1]) / 2.0)

    forehead_pt = (forehead.x, forehead.y)
    chin_pt = (chin.x, chin.y)
    left_ear = (left_face.x, left_face.y)
    right_ear = (right_face.x, right_face.y)

    face_width = max(abs(right_face.x - left_face.x), 1e-6)
    face_height = max(chin.y - forehead.y, 1e-6)
    eye_to_forehead = max(forehead.y - eye_mid[1], 1e-6)
    eye_to_chin = max(chin.y - eye_mid[1], 1e-6)

    hairline_from_forehead = forehead.y - eye_to_forehead * 0.92
    hairline_from_proportion = eye_mid[1] - eye_to_chin * 0.36
    hairline_y = min(hairline_from_forehead, hairline_from_proportion)

    brow_clearance = max(brow_mid[1] - eye_mid[1], 1e-6)
    hairline_y = min(hairline_y, brow_mid[1] - brow_clearance * 0.55)
    hairline_y = max(hairline_y, forehead.y - face_height * 0.42)

    face_center_x = (left_face.x + right_face.x) / 2.0
    hairline_x = eye_mid[0] * 0.62 + face_center_x * 0.38

    temple_inset = face_width * 0.04
    return HairlineGeometry(
        x=hairline_x,
        y=hairline_y,
        eye_to_hairline=_distance(eye_mid, (hairline_x, hairline_y)),
        chin_to_hairline=_distance(chin_pt, (hairline_x, hairline_y)),
        brow_to_hairline=_distance(brow_mid, (hairline_x, hairline_y)),
        left_ear_to_hairline=_distance(left_ear, (hairline_x, hairline_y)),
        right_ear_to_hairline=_distance(right_ear, (hairline_x, hairline_y)),
        forehead_to_hairline=_distance(forehead_pt, (hairline_x, hairline_y)),
        face_height=face_height,
        face_width=face_width,
        left_temple_x=left_face.x + temple_inset,
        right_temple_x=right_face.x - temple_inset,
    )


def hairline_xy(
    landmarks: Sequence[Any],
    width: int,
    height: int,
) -> tuple[int, int]:
    geometry = compute_hairline(landmarks)
    return int(geometry.x * width), int(geometry.y * height)
