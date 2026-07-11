import numpy as np

from src.perception.frame_quality import assess_frame_quality, draw_quality_pill


def test_frame_quality_good_when_face_and_contrast():
    frame = np.full((120, 160, 3), 130, dtype=np.uint8)
    frame[::3, ::3] = 40
    frame[1::3, 2::3] = 210
    message, color = assess_frame_quality(frame, face_detected=True)
    assert message == "QUALITY OK"
    assert color == (80, 220, 100)


def test_frame_quality_warns_without_face():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    message, _ = assess_frame_quality(frame, face_detected=False)
    assert message == "FACE NOT DETECTED"


def test_quality_pill_draws_without_error():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    draw_quality_pill(frame, "QUALITY OK", (80, 220, 100))
    assert frame.shape == (200, 320, 3)
