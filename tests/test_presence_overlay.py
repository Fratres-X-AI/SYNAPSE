import numpy as np

from src.perception.presence_detector import PresenceBox, PresenceFrame, display_label
from src.visualization.presence_overlay import draw_presence_overlay, presence_hud_note


def test_draw_presence_overlay_skips_coco_person_on_user():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    presence = PresenceFrame(
        people=(PresenceBox("user", 0.2, 0.2, 0.5, 0.7, 0.9, is_primary=True),),
        objects=(PresenceBox("person", 0.15, 0.15, 0.55, 0.85, 0.72),),
        primary_index=0,
    )

    result = draw_presence_overlay(frame, presence)

    assert result is frame
    assert np.array_equal(result, original)


def test_draw_presence_overlay_skips_user_and_hand_boxes():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    presence = PresenceFrame(
        people=(
            PresenceBox("user", 0.2, 0.2, 0.5, 0.7, 0.9, is_primary=True),
            PresenceBox("person", 0.22, 0.22, 0.48, 0.68, 0.75),
            PresenceBox("person", 0.55, 0.2, 0.8, 0.6, 0.8),
        ),
        objects=(
            PresenceBox("phone", 0.42, 0.45, 0.5, 0.62, 0.7),
            PresenceBox("hand", 0.30, 0.50, 0.40, 0.65, 0.7),
        ),
        primary_index=0,
    )

    result = draw_presence_overlay(frame, presence)

    assert result is frame
    assert not np.array_equal(result, original)
    # Primary user, duplicate face, and hand interiors stay unmarked.
    assert result[200, 200].sum() == 0
    assert result[220, 210].sum() == 0
    assert result[276, 224].sum() == 0
    # Real visitor and phone boxes are still drawn.
    assert result[96, 384].sum() != 0
    assert result[216, 288].sum() != 0


def test_presence_hud_note_lists_identified_labels():
    presence = PresenceFrame(
        people=(
            PresenceBox("user", 0.2, 0.2, 0.5, 0.7, 0.9, is_primary=True),
            PresenceBox("person", 0.55, 0.2, 0.8, 0.6, 0.8),
        ),
        objects=(PresenceBox("phone", 0.42, 0.45, 0.5, 0.62, 0.7),),
        primary_index=0,
    )

    note = presence_hud_note(presence)

    assert "User" in note
    assert "Person" in note
    assert "Phone" in note


def test_presence_hud_note_monitor_mode_hides_user_and_shows_phone_only():
    presence = PresenceFrame(
        people=(PresenceBox("user", 0.2, 0.2, 0.5, 0.7, 0.9, is_primary=True),),
        objects=(PresenceBox("phone", 0.42, 0.45, 0.5, 0.62, 0.7),),
        events=("smoking",),
        primary_index=0,
    )

    note = presence_hud_note(presence, monitor=True)

    assert note == "Phone | Smoking"
    assert "User" not in note
