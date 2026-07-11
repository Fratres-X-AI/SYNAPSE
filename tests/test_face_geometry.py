from types import SimpleNamespace

import numpy as np

from src.perception.face_geometry import compute_hairline, compute_peripheral_mesh


def _landmark(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0)


def _synthetic_landmarks() -> list[SimpleNamespace]:
    landmarks = [_landmark(0.0, 0.0) for _ in range(478)]

    landmarks[1] = _landmark(0.50, 0.56)
    landmarks[10] = _landmark(0.50, 0.34)
    landmarks[152] = _landmark(0.50, 0.78)
    landmarks[234] = _landmark(0.30, 0.52)
    landmarks[454] = _landmark(0.70, 0.52)
    landmarks[127] = _landmark(0.28, 0.58)
    landmarks[162] = _landmark(0.27, 0.66)
    landmarks[21] = _landmark(0.29, 0.72)
    landmarks[356] = _landmark(0.72, 0.58)
    landmarks[323] = _landmark(0.73, 0.66)
    landmarks[361] = _landmark(0.71, 0.72)
    landmarks[338] = _landmark(0.48, 0.32)
    landmarks[297] = _landmark(0.40, 0.30)
    landmarks[332] = _landmark(0.60, 0.30)

    for index in (33, 160, 158, 133, 153, 144):
        landmarks[index] = _landmark(0.38, 0.48)
    for index in (362, 385, 387, 263, 373, 380):
        landmarks[index] = _landmark(0.62, 0.48)
    for index in (293, 334, 336):
        landmarks[index] = _landmark(0.40, 0.42)
    for index in (63, 105, 107):
        landmarks[index] = _landmark(0.60, 0.42)

    return landmarks


def test_elite_shell_has_rim_crown_and_rear():
    geometry = compute_hairline(_synthetic_landmarks())
    mesh = compute_peripheral_mesh(_synthetic_landmarks())

    assert mesh.rim_count == 32
    assert mesh.crown_offset == 32
    assert mesh.left_ear_offset == 37
    assert len(mesh.shell_lines) == 0
    assert len(mesh.mediapipe_bridges) >= 4
    assert min(point[1] for point in mesh.points[:32]) < geometry.y


def test_helmet_rear_closure_has_no_extra_landmarks():
    landmarks = _synthetic_landmarks()
    landmarks[1] = _landmark(0.36, 0.56)
    mesh = compute_peripheral_mesh(landmarks)

    assert len(mesh.shell_lines) == 4
    for start, end in mesh.shell_lines:
        assert start[1] < end[1] or end[1] < start[1]


def test_profile_ears_only_when_turned():
    frontal = compute_peripheral_mesh(_synthetic_landmarks())
    assert frontal.left_ear_count == 0
    assert frontal.right_ear_count == 0

    turned = _synthetic_landmarks()
    turned[1] = _landmark(0.36, 0.56)
    profile = compute_peripheral_mesh(turned)
    assert profile.left_ear_count == 3

    moderate = _synthetic_landmarks()
    moderate[1] = _landmark(0.62, 0.56)
    partial = compute_peripheral_mesh(moderate)
    assert partial.left_ear_count == 0
    assert partial.right_ear_count == 3


def test_hairline_sits_above_forehead_and_brows():
    geometry = compute_hairline(_synthetic_landmarks())

    assert geometry.y < 0.34
    assert geometry.forehead_to_hairline > 0.0
    assert geometry.eye_to_hairline > geometry.forehead_to_hairline
    assert geometry.chin_to_hairline > geometry.eye_to_hairline


def test_hairline_distances_use_ears_and_brows():
    geometry = compute_hairline(_synthetic_landmarks())

    assert geometry.left_ear_to_hairline > 0.15
    assert geometry.right_ear_to_hairline > 0.15
    assert geometry.brow_to_hairline > 0.02
    assert geometry.left_temple_x < geometry.x < geometry.right_temple_x
