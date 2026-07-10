"""Elite landmark overlays — MediaPipe core mesh + tight helmet shell."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np

from src.perception.face_geometry import compute_hairline, compute_peripheral_mesh
from src.perception.state_estimator import StateEstimator

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Layer 1 — dense face mesh (hero)
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(175, 195, 215), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(210, 225, 240), thickness=1, circle_radius=0)
FACE_OVAL_STYLE = mp_drawing.DrawingSpec(color=(235, 242, 250), thickness=2, circle_radius=0)
FEATURE_STYLE = mp_drawing.DrawingSpec(color=(140, 200, 255), thickness=1, circle_radius=0)

# Layer 2 — helmet shell (rim + crown + ears)
SHELL_RIM_COLOR = (220, 238, 252)
SHELL_CROWN_COLOR = (190, 225, 245)
SHELL_EAR_COLOR = (150, 220, 255)
SHELL_BRIDGE_COLOR = (155, 185, 210)
SHELL_REAR_COLOR = (170, 194, 216)

# Layer 3 — tracking accents
IRIS_COLOR = (255, 196, 48)
IRIS_RING = (248, 246, 240)
HAIRLINE_COLOR = (120, 245, 255)
POSE_COLOR = (140, 175, 210)


def landmark_xy(landmark, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def _blend_center_overlay(frame: np.ndarray, overlay: np.ndarray, alpha: float, layout=None) -> np.ndarray:
    if layout is None:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame
    x, y, w, h = _center_roi(frame, layout)
    region = frame[y : y + h, x : x + w]
    source = overlay[y : y + h, x : x + w]
    cv2.addWeighted(source, alpha, region, 1.0 - alpha, 0, region)
    return frame


def _center_roi(frame: np.ndarray, layout) -> tuple[int, int, int, int]:
    x1 = layout.left_x + layout.left_w + layout.margin
    x2 = layout.right_x - layout.margin
    y1 = layout.content_top
    y2 = layout.content_bottom
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _draw_line_norm(
    overlay: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    width: int,
    height: int,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.line(
        overlay,
        (int(p1[0] * width), int(p1[1] * height)),
        (int(p2[0] * width), int(p2[1] * height)),
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_elite_face_mesh(frame, landmarks, *, layout=None) -> np.ndarray:
    """MediaPipe 478 mesh — tessellation + contours + features. No dot spam."""
    overlay = frame.copy()
    face_landmarks = SimpleNamespace(landmark=landmarks)

    mp_drawing.draw_landmarks(
        image=overlay,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=TESSELATION_STYLE,
    )
    mp_drawing.draw_landmarks(
        image=overlay,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=CONTOUR_STYLE,
    )
    for connections, style in (
        (mp_face_mesh.FACEMESH_FACE_OVAL, FACE_OVAL_STYLE),
        (mp_face_mesh.FACEMESH_LEFT_EYE, FEATURE_STYLE),
        (mp_face_mesh.FACEMESH_RIGHT_EYE, FEATURE_STYLE),
        (mp_face_mesh.FACEMESH_LEFT_EYEBROW, FEATURE_STYLE),
        (mp_face_mesh.FACEMESH_RIGHT_EYEBROW, FEATURE_STYLE),
        (mp_face_mesh.FACEMESH_LIPS, FEATURE_STYLE),
        (mp_face_mesh.FACEMESH_IRISES, FEATURE_STYLE),
    ):
        mp_drawing.draw_landmarks(
            image=overlay,
            landmark_list=face_landmarks,
            connections=connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=style,
        )

    return _blend_center_overlay(frame, overlay, 0.44, layout)


def draw_elite_helmet_shell(frame, landmarks, *, layout=None) -> np.ndarray:
    """Tight helmet rim, crown band, profile ears, clean rear — bridged to MediaPipe."""
    overlay = frame.copy()
    height, width = overlay.shape[:2]
    mesh = compute_peripheral_mesh(landmarks)
    hairline = compute_hairline(landmarks)

    for start, end in mesh.connections:
        color = SHELL_RIM_COLOR
        if mesh.crown_offset <= start < mesh.left_ear_offset or mesh.crown_offset <= end < mesh.left_ear_offset:
            color = SHELL_CROWN_COLOR
        if mesh.left_ear_count and (
            mesh.left_ear_offset <= start < mesh.right_ear_offset
            or mesh.left_ear_offset <= end < mesh.right_ear_offset
        ):
            color = SHELL_EAR_COLOR
        if mesh.right_ear_count and (
            mesh.right_ear_offset <= start < mesh.point_count
            or mesh.right_ear_offset <= end < mesh.point_count
        ):
            color = SHELL_EAR_COLOR
        _draw_line_norm(overlay, mesh.points[start], mesh.points[end], width, height, color, 1)

    for shell_index, mp_index in mesh.mediapipe_bridges:
        shell_pt = mesh.points[shell_index]
        mp_pt = (landmarks[mp_index].x, landmarks[mp_index].y)
        _draw_line_norm(overlay, shell_pt, mp_pt, width, height, SHELL_BRIDGE_COLOR, 1)

    for start, end in mesh.shell_lines:
        _draw_line_norm(overlay, start, end, width, height, SHELL_REAR_COLOR, 2)

    hairline_pt = (int(hairline.x * width), int(hairline.y * height))
    cv2.circle(overlay, hairline_pt, 3, HAIRLINE_COLOR, -1, cv2.LINE_AA)
    cv2.circle(overlay, hairline_pt, 5, (248, 252, 255), 1, cv2.LINE_AA)

    for offset, count in (
        (mesh.left_ear_offset, mesh.left_ear_count),
        (mesh.right_ear_offset, mesh.right_ear_count),
    ):
        for index in range(offset, offset + count):
            point = mesh.points[index]
            cv2.circle(
                overlay,
                (int(point[0] * width), int(point[1] * height)),
                2,
                SHELL_EAR_COLOR,
                -1,
                cv2.LINE_AA,
            )

    return _blend_center_overlay(frame, overlay, 0.52, layout)


def draw_gaze_tracking(frame, landmarks, *, layout=None) -> np.ndarray:
    overlay = frame.copy()
    height, width = overlay.shape[:2]
    for iris_index in (StateEstimator.LEFT_IRIS, StateEstimator.RIGHT_IRIS):
        iris = landmarks[iris_index]
        iris_pt = (int(iris.x * width), int(iris.y * height))
        cv2.circle(overlay, iris_pt, 3, IRIS_COLOR, -1, cv2.LINE_AA)
        cv2.circle(overlay, iris_pt, 4, IRIS_RING, 1, cv2.LINE_AA)
    return _blend_center_overlay(frame, overlay, 0.9, layout)


def draw_head_pose(frame, landmarks, *, layout=None) -> np.ndarray:
    overlay = frame.copy()
    height, width = overlay.shape[:2]
    nose = landmark_xy(landmarks[StateEstimator.NOSE_TIP], width, height)
    cv2.circle(overlay, nose, 3, POSE_COLOR, -1, cv2.LINE_AA)
    return _blend_center_overlay(frame, overlay, 0.55, layout)


def draw_peripheral_landmark_web(frame, landmarks, *, layout=None) -> np.ndarray:
    return draw_elite_helmet_shell(frame, landmarks, layout=layout)


def draw_expanded_face_shell(frame, landmarks, *, layout=None) -> np.ndarray:
    return draw_elite_helmet_shell(frame, landmarks, layout=layout)


def draw_full_face_mesh(frame, landmarks, *, layout=None) -> np.ndarray:
    frame = draw_elite_face_mesh(frame, landmarks, layout=layout)
    return draw_elite_helmet_shell(frame, landmarks, layout=layout)


def draw_expression_regions(frame, landmarks, *, layout=None) -> np.ndarray:
    del landmarks, layout
    return frame


def draw_hairline_tracking(frame, landmarks, *, layout=None) -> np.ndarray:
    del landmarks, layout
    return frame


def draw_all_tracking_overlays(frame, landmarks, *, subtle: bool = True) -> np.ndarray:
    del subtle
    frame = draw_elite_face_mesh(frame, landmarks, layout=None)
    frame = draw_elite_helmet_shell(frame, landmarks, layout=None)
    frame = draw_gaze_tracking(frame, landmarks, layout=None)
    frame = draw_head_pose(frame, landmarks, layout=None)
    return frame


def showcase_subtitle(landmark_count: int, signals: dict | None = None) -> str:
    shell_count = int(signals.get("peripheral_landmarks", 0)) if signals else 0
    return f"{landmark_count}+{shell_count}"
