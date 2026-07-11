import time

from utils.fps_tracker import FpsTracker


def test_fps_tracker_returns_positive_after_ticks():
    tracker = FpsTracker()
    for _ in range(5):
        time.sleep(0.01)
        fps = tracker.tick()
    assert fps > 0
