from types import SimpleNamespace

from src.perception.shoulder_tracker import ShoulderSample, ShoulderTracker


def _pose(y: float, visibility: float = 0.9):
    landmarks = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)]
    landmarks[11] = SimpleNamespace(x=0.35, y=y, z=0.0, visibility=visibility)
    landmarks[12] = SimpleNamespace(x=0.65, y=y, z=0.0, visibility=visibility)
    return landmarks


def test_shoulder_tracker_marks_elevation_on_inhale():
    tracker = ShoulderTracker(elevate_threshold=0.01)

    calm = tracker.update(_pose(0.62))
    assert calm.visible
    assert calm.elevated is False

    inhale = tracker.update(_pose(0.58))
    assert inhale.visible
    assert inhale.elevated is True


def test_posture_allows_smoking_uses_shoulder_lift():
    from src.perception.presence_detector import SmokingEventTracker
    from src.perception.shoulder_tracker import ShoulderSample

    calm = ShoulderSample(visible=True, elevated=False)
    raised = ShoulderSample(visible=True, elevated=True)

    assert SmokingEventTracker._posture_allows_smoking(calm, heavy_vapor=False) is False
    assert SmokingEventTracker._posture_allows_smoking(raised, heavy_vapor=False) is True
    assert SmokingEventTracker._posture_allows_smoking(calm, heavy_vapor=True) is True
    assert SmokingEventTracker._posture_allows_smoking(None, heavy_vapor=False) is True
