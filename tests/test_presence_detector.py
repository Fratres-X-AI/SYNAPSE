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
        objects=(PresenceBox("phone", 0.55, 0.55, 0.7, 0.8, 0.8),),
        primary_index=0,
    )

    assert frame.active_labels() == {"user", "phone"}


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


def test_background_clutter_is_not_person():
    from src.perception.presence_detector import (
        _looks_like_background_clutter,
        _qualifies_as_coco_person,
        _qualifies_as_extra_person_face,
    )

    user = PresenceBox("user", 0.30, 0.18, 0.62, 0.82, 0.92, is_primary=True)
    backpack = PresenceBox("person", 0.58, 0.12, 0.92, 0.72, 0.55)
    jersey = PresenceBox("person", 0.05, 0.18, 0.42, 0.78, 0.51)
    visitor = PresenceBox("person", 0.62, 0.22, 0.82, 0.58, 0.78)

    assert _looks_like_background_clutter(backpack)
    assert _looks_like_background_clutter(jersey)
    assert not _qualifies_as_coco_person(backpack, user)
    assert not _qualifies_as_coco_person(jersey, user)
    assert not _qualifies_as_extra_person_face(backpack, user)
    assert _qualifies_as_extra_person_face(visitor, user)


def test_small_phone_in_hand_qualifies():
    from src.perception.presence_detector import _qualifies_as_phone

    phone = PresenceBox("phone", 0.44, 0.52, 0.52, 0.72, 0.62)
    hand = PresenceBox("hand", 0.40, 0.48, 0.56, 0.76, 0.7)

    assert _qualifies_as_phone(phone, [hand])


def test_phone_near_hand_without_overlap_qualifies():
    from src.perception.presence_detector import _qualifies_as_phone

    phone = PresenceBox("phone", 0.50, 0.38, 0.58, 0.62, 0.50)
    hand = PresenceBox("hand", 0.42, 0.44, 0.54, 0.70, 0.7)

    assert _qualifies_as_phone(phone, [hand])


def test_handheld_near_hand_becomes_phone():
    from src.perception.presence_detector import _classify_handheld

    box = PresenceBox("handheld", 0.46, 0.40, 0.54, 0.64, 0.55)
    hand = PresenceBox("hand", 0.42, 0.38, 0.58, 0.68, 0.7)

    assert _classify_handheld(box, mouth=None, hands=[hand]) == "phone"


def test_carrot_and_unknown_coco_labels_are_discarded():
    assert normalize_object_label("carrot") == ""
    assert normalize_object_label("cell phone") == "phone"
    assert normalize_object_label("parking meter") == ""


def test_filter_allowed_objects_drops_food_and_desk():
    from src.perception.presence_detector import _filter_allowed_objects

    objects = [
        PresenceBox("phone", 0.4, 0.5, 0.55, 0.75, 0.8),
        PresenceBox("carrot", 0.1, 0.1, 0.2, 0.2, 0.6),
        PresenceBox("food", 0.2, 0.2, 0.3, 0.3, 0.6),
    ]
    kept = _filter_allowed_objects(objects)
    assert {obj.label for obj in kept} == {"phone"}


def test_normalize_object_label_maps_phone_and_toothbrush():
    assert normalize_object_label("cell phone") == "phone"
    assert normalize_object_label("toothbrush") == "handheld"


def test_thin_handheld_on_desk_is_discarded():
    from src.perception.presence_detector import _classify_handheld

    box = PresenceBox("handheld", 0.45, 0.62, 0.52, 0.82, 0.7)
    assert _classify_handheld(box, mouth=(0.5, 0.4)) == "desk"


def test_blocky_handheld_at_mouth_is_not_phone():
    from src.perception.presence_detector import _classify_handheld

    box = PresenceBox("handheld", 0.47, 0.42, 0.53, 0.72, 0.7)
    assert _classify_handheld(box, mouth=(0.5, 0.5)) != "phone"


def test_square_pod_near_mouth_is_not_phone():
    from src.perception.presence_detector import _classify_handheld

    box = PresenceBox("handheld", 0.46, 0.44, 0.56, 0.58, 0.7)
    mouth_hand = PresenceBox("hand", 0.44, 0.42, 0.58, 0.62, 0.7)
    assert _classify_handheld(box, mouth=(0.5, 0.5), hands=[mouth_hand]) == ""


def test_presence_frame_active_labels_include_events():
    frame = PresenceFrame(
        people=(PresenceBox("user", 0.2, 0.2, 0.5, 0.7, 0.9, is_primary=True),),
        objects=(),
        events=("smoking",),
        primary_index=0,
        detected_hands=(PresenceBox("hand", 0.47, 0.48, 0.53, 0.62, 0.7),),
    )

    assert frame.active_labels() == {"user", "smoking"}


def test_smoking_tracker_logs_when_hand_at_mouth():
    from types import SimpleNamespace

    import numpy as np

    from src.perception.presence_detector import SmokingEventTracker, _MOUTH_INDICES
    from src.perception.shoulder_tracker import ShoulderSample

    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for index in _MOUTH_INDICES:
        landmarks[index] = SimpleNamespace(x=0.5, y=0.55, z=0.0)

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    presence = PresenceFrame(
        detected_hands=(PresenceBox("hand", 0.47, 0.48, 0.53, 0.62, 0.7),),
    )
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
    presence = PresenceFrame(
        detected_hands=(PresenceBox("hand", 0.47, 0.48, 0.53, 0.62, 0.7),),
    )
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


def test_smoking_tracker_suppressed_when_phone_present():
    from types import SimpleNamespace

    import numpy as np

    from src.perception.presence_detector import SmokingEventTracker, _MOUTH_INDICES
    from src.perception.shoulder_tracker import ShoulderSample

    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for index in _MOUTH_INDICES:
        landmarks[index] = SimpleNamespace(x=0.5, y=0.55, z=0.0)

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    bright = frame.copy()
    bright[180:240, 260:380] = 250
    presence = PresenceFrame(
        objects=(
            PresenceBox("phone", 0.44, 0.46, 0.56, 0.72, 0.7),
        ),
        detected_hands=(PresenceBox("hand", 0.42, 0.44, 0.58, 0.74, 0.7),),
    )
    tracker = SmokingEventTracker(sustain_seconds=0.0, vapor_boost=8.0, cooldown_seconds=0.0)
    shoulders = ShoulderSample(visible=True, center_y=0.58, lift=0.02, elevated=True)

    assert tracker.update(presence, bright, landmarks, 0.6, shoulder=shoulders) == ()


def test_phone_use_posture_infers_phone_not_pen():
    from src.perception.presence_detector import _infer_phone_from_use_posture, _phone_use_posture

    hand = PresenceBox("hand", 0.40, 0.48, 0.58, 0.78, 0.7)
    assert _phone_use_posture(hand, (0.5, 0.42))
    inferred = _infer_phone_from_use_posture([hand], None)
    assert inferred is None


def test_apply_phone_gaze_split_keeps_phone_when_mouth_hand_on_screen():
    from types import SimpleNamespace

    from src.perception.presence_detector import _apply_phone_gaze_split

    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    landmarks[1] = SimpleNamespace(x=0.5, y=0.36, z=0.0)
    for index in (13, 14, 78, 308):
        landmarks[index] = SimpleNamespace(x=0.5, y=0.52, z=0.0)
    for indices, iris_index, iris_dx, iris_dy in (
        ((33, 160, 158, 133, 153, 144), 468, 0.004, 0.018),
        ((362, 385, 387, 263, 373, 380), 473, 0.004, 0.018),
    ):
        outer, top_a, top_b, inner, bottom_a, bottom_b = indices
        landmarks[outer] = SimpleNamespace(x=0.42, y=0.34, z=0.0)
        landmarks[inner] = SimpleNamespace(x=0.58, y=0.34, z=0.0)
        landmarks[top_a] = SimpleNamespace(x=0.48, y=0.31, z=0.0)
        landmarks[top_b] = SimpleNamespace(x=0.52, y=0.31, z=0.0)
        landmarks[bottom_a] = SimpleNamespace(x=0.48, y=0.37, z=0.0)
        landmarks[bottom_b] = SimpleNamespace(x=0.52, y=0.37, z=0.0)
        landmarks[iris_index] = SimpleNamespace(x=0.5 + iris_dx, y=0.34 + iris_dy, z=0.0)

    phone = PresenceBox("phone", 0.44, 0.50, 0.56, 0.74, 0.82)
    phone_hand = PresenceBox("hand", 0.42, 0.48, 0.58, 0.76, 0.7)
    mouth_hand = PresenceBox("hand", 0.62, 0.56, 0.72, 0.74, 0.7)
    objects = [phone]
    kept = _apply_phone_gaze_split(objects, [phone_hand, mouth_hand], landmarks)
    phone_boxes = [obj for obj in kept if obj.label == "phone"]
    assert len(phone_boxes) == 1
    assert phone_boxes[0].center[0] < 0.55


def test_portrait_phone_stays_phone():
    from src.perception.presence_detector import _refine_object_labels

    phone = PresenceBox("phone", 0.43, 0.59, 0.61, 0.99, 0.58)
    refined = _refine_object_labels([phone], [], None, None)
    assert len(refined) == 1
    assert refined[0].label == "phone"


def test_device_slab_on_hand_from_capture_profile():
    from src.perception.presence_detector import _device_slab_on_hand

    hand = PresenceBox("hand", 0.42, 0.55, 0.58, 0.78, 0.7)
    phone_slab = PresenceBox("phone", 0.44, 0.56, 0.52, 0.72, 0.50)
    bare_hand_blob = PresenceBox("phone", 0.41, 0.54, 0.59, 0.79, 0.74)

    assert _device_slab_on_hand(phone_slab, hand)
    assert not _device_slab_on_hand(bare_hand_blob, hand)


def test_confident_coco_phone_passes_even_with_high_hand_overlap():
    from src.perception.presence_detector import _qualifies_as_phone

    # Grip overlap from phone-only capture: ratio ~0.65, still a real phone.
    phone = PresenceBox("phone", 0.44, 0.52, 0.54, 0.76, 0.55)
    hand = PresenceBox("hand", 0.42, 0.55, 0.58, 0.78, 0.7)

    assert _qualifies_as_phone(phone, [hand])


def test_stick_phone_label_holds_brief_gap():
    from time import monotonic

    from src.perception.presence_detector import PresenceDetector

    detector = PresenceDetector(enable_objects=False, enable_hands=False)
    hand = PresenceBox("hand", 0.42, 0.48, 0.58, 0.76, 0.7)
    phone = PresenceBox("phone", 0.44, 0.52, 0.52, 0.70, 0.72)
    now = monotonic()
    with_phone = detector._stick_phone_label([phone], [hand], None, now)
    assert {obj.label for obj in with_phone} == {"phone"}
    held = detector._stick_phone_label([], [hand], None, now + 0.5)
    assert {obj.label for obj in held} == {"phone"}
    cleared = detector._stick_phone_label([], [hand], None, now + 2.5)
    assert cleared == []
    detector.close()


def test_stick_phone_label_drops_hand_sized_phone():
    from src.perception.presence_detector import PresenceDetector

    detector = PresenceDetector(enable_objects=False, enable_hands=False)
    hand = PresenceBox("hand", 0.42, 0.55, 0.58, 0.78, 0.7)
    false_phone = PresenceBox("phone", 0.41, 0.54, 0.59, 0.79, 0.74)

    kept = detector._stick_phone_label([false_phone], [hand])
    assert [obj.label for obj in kept] == []
    detector.close()


def test_bare_hand_not_labeled_as_phone():
    from src.perception.presence_detector import _drop_hand_shaped_phones

    hand = PresenceBox("hand", 0.42, 0.55, 0.58, 0.78, 0.7)
    hand_as_phone = PresenceBox("phone", 0.41, 0.54, 0.59, 0.79, 0.62)
    real_phone = PresenceBox("phone", 0.44, 0.52, 0.52, 0.70, 0.72)

    kept = _drop_hand_shaped_phones([hand_as_phone, real_phone], [hand])
    assert len(kept) == 1
    assert kept[0].center[0] < 0.52


def test_display_label_formats_for_log():
    assert display_label("user") == "User"
    assert display_label("phone") == "Phone"
    assert display_label("smoking") == "Smoking"


def test_phone_in_front_of_face_is_not_blocked():
    from types import SimpleNamespace

    from src.perception.presence_detector import (
        _MOUTH_INDICES,
        _blocks_phone_label,
        _qualifies_as_phone,
    )

    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for index in _MOUTH_INDICES:
        landmarks[index] = SimpleNamespace(x=0.5, y=0.48, z=0.0)

    mouth = (0.5, 0.48)
    hand = PresenceBox("hand", 0.38, 0.42, 0.62, 0.82, 0.7)
    phone = PresenceBox("phone", 0.36, 0.30, 0.64, 0.88, 0.62)

    assert not _blocks_phone_label(phone, mouth, [hand])
    assert _qualifies_as_phone(phone, [hand], mouth)


def test_coco_phone_at_mouth_is_dropped_not_phone():
    from types import SimpleNamespace

    from src.perception.presence_detector import _MOUTH_INDICES, _refine_object_labels

    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for index in _MOUTH_INDICES:
        landmarks[index] = SimpleNamespace(x=0.5, y=0.5, z=0.0)

    pod = PresenceBox("phone", 0.46, 0.44, 0.56, 0.58, 0.72)
    mouth_hand = PresenceBox("hand", 0.44, 0.42, 0.58, 0.62, 0.7)
    refined = _refine_object_labels([pod], [mouth_hand], landmarks, None)
    assert refined == []


def test_smoking_tracker_arms_watch_when_hand_at_mouth_then_fires():
    from types import SimpleNamespace

    import numpy as np

    from src.perception.presence_detector import SmokingEventTracker, _MOUTH_INDICES
    from src.perception.shoulder_tracker import ShoulderSample

    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for index in _MOUTH_INDICES:
        landmarks[index] = SimpleNamespace(x=0.5, y=0.55, z=0.0)

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    presence = PresenceFrame(
        detected_hands=(PresenceBox("hand", 0.47, 0.48, 0.53, 0.62, 0.7),),
    )
    tracker = SmokingEventTracker(
        watch_seconds=10.0,
        sustain_seconds=0.0,
        vapor_boost=8.0,
        cooldown_seconds=0.0,
    )
    calm = ShoulderSample(visible=True, center_y=0.58, lift=0.0, elevated=False)
    raised = ShoulderSample(visible=True, center_y=0.58, lift=0.02, elevated=True)

    assert tracker.update(presence, frame, landmarks, 0.0, shoulder=calm) == ()
    bright = frame.copy()
    bright[180:240, 260:380] = 250
    assert tracker.update(presence, bright, landmarks, 2.0, shoulder=calm) == ()
    assert tracker.update(presence, bright, landmarks, 2.5, shoulder=calm) == ("smoking",)


def test_smoking_tracker_fires_with_phone_and_vapor():
    from types import SimpleNamespace

    import numpy as np

    from src.perception.presence_detector import SmokingEventTracker, _MOUTH_INDICES
    from src.perception.shoulder_tracker import ShoulderSample

    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for index in _MOUTH_INDICES:
        landmarks[index] = SimpleNamespace(x=0.5, y=0.55, z=0.0)

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    bright = frame.copy()
    bright[180:240, 260:380] = 250
    presence = PresenceFrame(
        objects=(PresenceBox("phone", 0.44, 0.50, 0.56, 0.74, 0.7),),
        detected_hands=(PresenceBox("hand", 0.47, 0.48, 0.53, 0.62, 0.7),),
    )
    tracker = SmokingEventTracker(sustain_seconds=0.0, vapor_boost=8.0, cooldown_seconds=0.0)
    shoulders = ShoulderSample(visible=True, center_y=0.58, lift=0.02, elevated=True)

    assert tracker.update(presence, frame, landmarks, 0.0, shoulder=shoulders) == ()
    assert tracker.update(presence, bright, landmarks, 0.5, shoulder=shoulders) == ()
    assert tracker.update(presence, bright, landmarks, 0.6, shoulder=shoulders) == ("smoking",)


def test_presence_event_logger_writes_timestamped_line(tmp_path: Path):
    log_path = tmp_path / "presence.log"
    logger = PresenceEventLogger(log_path, sustain_seconds=0.0, label_sustain={})
    when = datetime(2026, 7, 11, 7, 17, 0)

    lines = logger.update({"user", "phone"}, when=when)

    assert lines == ["7:17 AM | Phone"]
    assert log_path.read_text(encoding="utf-8").strip().endswith("7:17 AM | Phone")


def test_presence_event_logger_ignores_flicker(tmp_path: Path):
    log_path = tmp_path / "presence.log"
    logger = PresenceEventLogger(log_path, sustain_seconds=0.0, label_sustain={"phone": 10.0})
    when = datetime(2026, 7, 11, 7, 17, 0)

    assert logger.update({"phone"}, when=when) == []
    assert logger.update(set(), when=when) == []
    assert log_path.read_text(encoding="utf-8").strip() == "# Synapse presence log (sustained episodes)"


def test_presence_event_logger_hourly_phone_total(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "presence.log"
    logger = PresenceEventLogger(log_path, sustain_seconds=0.0, label_sustain={"phone": 10.0})
    clock = iter([0.0, 5.0, 10.0, 18.0, 25.0, 25.0])
    monkeypatch.setattr("src.monitoring.presence_logger.monotonic", lambda: next(clock))
    when_7 = datetime(2026, 7, 11, 7, 45, 0)
    when_8 = datetime(2026, 7, 11, 8, 0, 0)

    assert logger.update({"phone"}, when=when_7) == []
    assert logger.update(set(), when=when_7) == []
    assert logger.update({"phone"}, when=when_7) == []
    assert logger.update(set(), when=when_7) == []
    lines = logger.update(set(), when=when_8)

    assert lines == ["8:00 AM | Phone this hour (7 AM\u20138 AM): 13s total"]
    assert logger.finalize(when=when_8) == []


def test_presence_event_logger_logs_sustained_phone(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "presence.log"
    logger = PresenceEventLogger(log_path, sustain_seconds=0.0, label_sustain={"phone": 10.0})
    clock = iter([0.0, 11.0])
    monkeypatch.setattr("src.monitoring.presence_logger.monotonic", lambda: next(clock))
    when = datetime(2026, 7, 11, 7, 17, 0)

    assert logger.update({"phone"}, when=when) == []
    assert logger.update({"phone"}, when=when) == ["7:17 AM | Phone"]


def test_presence_event_logger_writes_end_line(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "presence.log"
    logger = PresenceEventLogger(log_path, sustain_seconds=0.0, label_sustain={})
    clock = iter([0.0, 10.0, 70.0])
    monkeypatch.setattr("src.monitoring.presence_logger.monotonic", lambda: next(clock))
    when = datetime(2026, 7, 11, 7, 17, 0)

    assert logger.update({"phone"}, when=when) == ["7:17 AM | Phone"]
    assert logger.update({"phone"}, when=when) == []
    assert logger.update(set(), when=when) == ["7:17 AM | Phone ended (1m 10s)"]
