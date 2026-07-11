from types import SimpleNamespace

import numpy as np

from src.visualization import landmark_overlay
from src.visualization.landmark_overlay import (
    draw_all_tracking_overlays,
    draw_elite_face_mesh,
    draw_elite_helmet_shell,
    showcase_subtitle,
)


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


def test_draw_elite_face_mesh_returns_same_shape(monkeypatch):
    calls = {"count": 0}

    def fake_draw_landmarks(*args, **kwargs):
        calls["count"] += 1

    monkeypatch.setattr(landmark_overlay.mp_drawing, "draw_landmarks", fake_draw_landmarks)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = _synthetic_landmarks()

    result = draw_elite_face_mesh(frame, landmarks)

    assert result.shape == frame.shape
    assert calls["count"] > 0


def test_draw_elite_helmet_shell_draws_shell_geometry():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    landmarks = _synthetic_landmarks()

    result = draw_elite_helmet_shell(frame, landmarks)

    assert result.shape == frame.shape
    assert result is frame
    assert not np.array_equal(result, original)
    assert result.max() > 0


def test_draw_all_tracking_overlays_runs_full_stack(monkeypatch):
    def fake_draw_landmarks(*args, **kwargs):
        return None

    monkeypatch.setattr(landmark_overlay.mp_drawing, "draw_landmarks", fake_draw_landmarks)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    original = frame.copy()
    landmarks = _synthetic_landmarks()

    result = draw_all_tracking_overlays(frame, landmarks)

    assert result.shape == frame.shape
    assert result is frame
    assert not np.array_equal(result, original)


def test_showcase_subtitle_includes_shell_count():
    subtitle = showcase_subtitle(478, {"peripheral_landmarks": 37})
    assert subtitle == "478+37"
