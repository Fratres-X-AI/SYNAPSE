from datetime import datetime
from pathlib import Path

from src.monitoring.presence_logger import PresenceEventLogger
from src.perception.presence_detector import (
    PresenceBox,
    PresenceFrame,
    PresenceTracker,
    display_label,
    normalize_object_label,
    select_primary_face,
)


def test_select_primary_face_uses_mesh_nose_when_available():
    from types import SimpleNamespace

    faces = [
        PresenceBox("person", 0.10, 0.20, 0.30, 0.50, 0.9),
        PresenceBox("person", 0.55, 0.20, 0.80, 0.55, 0.8),
    ]
    landmarks = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(478)]
    landmarks[1] = SimpleNamespace(x=0.62, y=0.35, z=0.0)

    assert select_primary_face(faces, landmarks) == 1


def test_dedupe_face_boxes_drops_duplicate_detection():
    from src.perception.presence_detector import dedupe_face_boxes

    faces = [
        PresenceBox("person", 0.2, 0.2, 0.5, 0.7, 0.9),
        PresenceBox("person", 0.22, 0.22, 0.48, 0.68, 0.75),
        PresenceBox("person", 0.55, 0.2, 0.8, 0.6, 0.8),
    ]

    assert len(dedupe_face_boxes(faces)) == 2


def test_active_labels_skip_coco_person_on_user():
    frame = PresenceFrame(
        people=(PresenceBox("user", 0.2, 0.2, 0.5, 0.7, 0.9, is_primary=True),),
        objects=(PresenceBox("person", 0.15, 0.15, 0.55, 0.85, 0.72),),
        primary_index=0,
    )

    assert frame.active_labels() == {"user"}


def test_presence_frame_active_labels_include_user_and_objects():
    frame = PresenceFrame(
        people=(PresenceBox("user", 0.2, 0.2, 0.5, 0.7, 0.9, is_primary=True),),
        objects=(
            PresenceBox("vape", 0.42, 0.45, 0.5, 0.62, 0.7),
            PresenceBox("phone", 0.55, 0.55, 0.7, 0.8, 0.8),
        ),
        primary_index=0,
    )

    assert frame.active_labels() == {"user", "vape", "phone"}


def test_presence_tracker_emits_visitor_after_sustain():
    tracker = PresenceTracker(sustain_seconds=2.0)
    assert tracker.update(1, 0.0) == ""
    assert tracker.update(1, 2.1) == "visitor"
    assert tracker.update(0, 3.0) == "cleared"


def test_large_background_object_is_not_phone():
    from src.perception.presence_detector import _qualifies_as_phone

    backpack_like = PresenceBox("phone", 0.58, 0.12, 0.92, 0.72, 0.55)
    jersey_like = PresenceBox("phone", 0.05, 0.18, 0.42, 0.78, 0.51)

    assert not _qualifies_as_phone(backpack_like, [])
    assert not _qualifies_as_phone(jersey_like, [])


def test_small_phone_in_hand_qualifies():
    from src.perception.presence_detector import _qualifies_as_phone

    phone = PresenceBox("phone", 0.44, 0.52, 0.52, 0.72, 0.62)
    hand = PresenceBox("hand", 0.40, 0.48, 0.56, 0.76, 0.7)

    assert _qualifies_as_phone(phone, [hand])


def test_normalize_object_label_maps_phone_and_toothbrush():
    assert normalize_object_label("cell phone") == "phone"
    assert normalize_object_label("toothbrush") == "handheld"


def test_thin_handheld_on_desk_becomes_pen_not_vape():
    from src.perception.presence_detector import _classify_handheld

    box = PresenceBox("handheld", 0.45, 0.62, 0.52, 0.82, 0.7)
    assert _classify_handheld(box, mouth=(0.5, 0.4)) == "pen"


def test_square_pod_near_mouth_is_vape():
    from src.perception.presence_detector import _classify_handheld, _looks_like_vape

    box = PresenceBox("handheld", 0.46, 0.44, 0.56, 0.58, 0.7)
    assert _looks_like_vape(box)
    assert _classify_handheld(box, mouth=(0.5, 0.5)) == "vape"


def test_thin_object_near_mouth_stays_pen():
    from src.perception.presence_detector import _classify_handheld

    box = PresenceBox("handheld", 0.47, 0.42, 0.53, 0.72, 0.7)
    assert _classify_handheld(box, mouth=(0.5, 0.5)) == "pen"


def test_presence_frame_active_labels_include_events():
    frame = PresenceFrame(
        people=(PresenceBox("user", 0.2, 0.2, 0.5, 0.7, 0.9, is_primary=True),),
        objects=(PresenceBox("pen", 0.47, 0.48, 0.53, 0.62, 0.7),),
        events=("smoking",),
        primary_index=0,
    )

    assert frame.active_labels() == {"user", "pen", "smoking"}


def test_smoking_tracker_logs_when_vapor_and_pen_at_mouth():
    from types import SimpleNamespace

    import numpy as np

    from src.perception.presence_detector import SmokingEventTracker, _MOUTH_INDICES
    from src.perception.shoulder_tracker import ShoulderSample

    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for index in _MOUTH_INDICES:
        landmarks[index] = SimpleNamespace(x=0.5, y=0.55, z=0.0)

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    presence = PresenceFrame(objects=(PresenceBox("pen", 0.47, 0.48, 0.53, 0.62, 0.7),))
    tracker = SmokingEventTracker(sustain_seconds=0.0, vapor_boost=8.0, cooldown_seconds=0.0)
    shoulders = ShoulderSample(visible=True, center_y=0.58, lift=0.02, elevated=True)

    assert tracker.update(presence, frame, landmarks, 0.0, shoulder=shoulders) == ()
    bright = frame.copy()
    bright[180:240, 260:380] = 250
    assert tracker.update(presence, bright, landmarks, 0.5, shoulder=shoulders) == ()
    assert tracker.update(presence, bright, landmarks, 0.6, shoulder=shoulders) == ("smoking",)
    assert tracker.update(presence, bright, landmarks, 1.0, shoulder=shoulders) == ("smoking",)


def test_smoking_tracker_fires_on_vapor_with_hand_at_mouth():
    from types import SimpleNamespace

    import numpy as np

    from src.perception.presence_detector import SmokingEventTracker, _MOUTH_INDICES
    from src.perception.shoulder_tracker import ShoulderSample

    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for index in _MOUTH_INDICES:
        landmarks[index] = SimpleNamespace(x=0.5, y=0.55, z=0.0)

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    presence = PresenceFrame(objects=(PresenceBox("hand", 0.47, 0.48, 0.53, 0.62, 0.7),))
    tracker = SmokingEventTracker(sustain_seconds=0.0, vapor_boost=8.0, cooldown_seconds=0.0)
    shoulders = ShoulderSample(visible=True, center_y=0.58, lift=0.02, elevated=True)

    assert tracker.update(presence, frame, landmarks, 0.0, shoulder=shoulders) == ()
    bright = frame.copy()
    bright[180:240, 260:380] = 250
    assert tracker.update(presence, bright, landmarks, 0.5, shoulder=shoulders) == ()
    assert tracker.update(presence, bright, landmarks, 0.6, shoulder=shoulders) == ("smoking",)


def test_smoking_tracker_detects_blown_exhale_without_hand():
    from types import SimpleNamespace

    import numpy as np

    from src.perception.presence_detector import SmokingEventTracker, _MOUTH_INDICES

    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for index in _MOUTH_INDICES:
        landmarks[index] = SimpleNamespace(x=0.5, y=0.55, z=0.0)

    frame = np.full((480, 640, 3), 35, dtype=np.uint8)
    presence = PresenceFrame(objects=())
    tracker = SmokingEventTracker(
        sustain_seconds=0.0,
        vapor_boost=9.0,
        heavy_vapor_boost=14.0,
        cooldown_seconds=0.0,
    )

    assert tracker.update(presence, frame, landmarks, 0.0) == ()
    bright = frame.copy()
    bright[300:360, 240:400] = 245
    assert tracker.update(presence, bright, landmarks, 0.4) == ()
    assert tracker.update(presence, bright, landmarks, 0.5) == ("smoking",)


def test_display_label_formats_for_log():
    assert display_label("user") == "User"
    assert display_label("vape") == "Vape"
    assert display_label("smoking") == "Smoking"


def test_presence_event_logger_writes_timestamped_line(tmp_path: Path):
    log_path = tmp_path / "presence.log"
    logger = PresenceEventLogger(log_path, sustain_seconds=0.0)
    when = datetime(2026, 7, 11, 7, 17, 0)

    lines = logger.update({"user", "vape"}, when=when)

    assert lines == ["7:17 AM | User | Vape"]
    assert log_path.read_text(encoding="utf-8").strip().endswith("7:17 AM | User | Vape")
