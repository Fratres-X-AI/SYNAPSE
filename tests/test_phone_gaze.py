from types import SimpleNamespace

from src.perception.phone_gaze import (
    eyes_drawn_to_phone_use,
    estimate_gaze_vector,
    gaze_ray_point,
    gazed_hand_for_phone,
)
from src.perception.presence_detector import PresenceBox


def _landmarks_with_gaze(*, gaze_x: float = 0.0, gaze_y: float = 0.0):
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    points[1] = SimpleNamespace(x=0.5, y=0.36, z=0.0)
    for index in (13, 14, 78, 308):
        points[index] = SimpleNamespace(x=0.5, y=0.52, z=0.0)

    def set_eye(indices, iris_index, iris_dx: float, iris_dy: float) -> None:
        outer, top_a, top_b, inner, bottom_a, bottom_b = indices
        points[outer] = SimpleNamespace(x=0.42, y=0.34, z=0.0)
        points[inner] = SimpleNamespace(x=0.58, y=0.34, z=0.0)
        points[top_a] = SimpleNamespace(x=0.48, y=0.31, z=0.0)
        points[top_b] = SimpleNamespace(x=0.52, y=0.31, z=0.0)
        points[bottom_a] = SimpleNamespace(x=0.48, y=0.37, z=0.0)
        points[bottom_b] = SimpleNamespace(x=0.52, y=0.37, z=0.0)
        points[iris_index] = SimpleNamespace(x=0.5 + iris_dx, y=0.34 + iris_dy, z=0.0)

    set_eye((33, 160, 158, 133, 153, 144), 468, gaze_x * 0.08, gaze_y * 0.08)
    set_eye((362, 385, 387, 263, 373, 380), 473, gaze_x * 0.08, gaze_y * 0.08)
    return points


def test_gaze_ray_points_down_toward_phone_hand():
    landmarks = _landmarks_with_gaze(gaze_x=0.05, gaze_y=0.22)
    gaze_x, gaze_y = estimate_gaze_vector(landmarks)
    ray_x, ray_y = gaze_ray_point(landmarks, gaze_x, gaze_y)
    hand = PresenceBox("hand", 0.42, 0.55, 0.58, 0.78, 0.7)

    assert ray_y > landmarks[1].y
    assert eyes_drawn_to_phone_use(landmarks, [hand], [])
    assert gazed_hand_for_phone([hand], landmarks, gaze_x, gaze_y, (0.5, 0.52)) == hand


def test_gaze_at_phone_box_any_angle():
    landmarks = _landmarks_with_gaze(gaze_x=-0.18, gaze_y=0.10)
    phone = PresenceBox("phone", 0.30, 0.48, 0.46, 0.72, 0.8)

    assert eyes_drawn_to_phone_use(landmarks, [], [phone])


def test_mouth_hand_excluded_from_gazed_hand():
    landmarks = _landmarks_with_gaze(gaze_x=0.05, gaze_y=0.22)
    gaze_x, gaze_y = estimate_gaze_vector(landmarks)
    phone_hand = PresenceBox("hand", 0.42, 0.55, 0.58, 0.78, 0.7)
    mouth_hand = PresenceBox("hand", 0.46, 0.46, 0.54, 0.58, 0.7)
    phone = PresenceBox("phone", 0.44, 0.56, 0.56, 0.76, 0.85)

    chosen = gazed_hand_for_phone(
        [mouth_hand, phone_hand],
        landmarks,
        gaze_x,
        gaze_y,
        (0.5, 0.52),
        phones=[phone],
    )
    assert chosen == phone_hand


def test_mouth_hand_not_inferred_as_phone_gaze():
    landmarks = _landmarks_with_gaze(gaze_x=0.05, gaze_y=0.22)
    gaze_x, gaze_y = estimate_gaze_vector(landmarks)
    mouth_hand = PresenceBox("hand", 0.46, 0.46, 0.54, 0.58, 0.7)

    assert (
        gazed_hand_for_phone(
            [mouth_hand],
            landmarks,
            gaze_x,
            gaze_y,
            (0.5, 0.52),
        )
        is None
    )


def test_hand_at_mouth_is_not_phone_gaze():
    landmarks = _landmarks_with_gaze(gaze_x=0.0, gaze_y=0.0)
    hand = PresenceBox("hand", 0.46, 0.46, 0.54, 0.58, 0.7)

    assert not eyes_drawn_to_phone_use(landmarks, [hand], [])
