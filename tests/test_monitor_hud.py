from src.perception.presence_detector import PresenceBox, PresenceFrame
from src.visualization.monitor_hud import (
    RuleBannerTracker,
    SmokingBannerTracker,
    VisitorBannerTracker,
    build_monitor_subtitle,
    compose_monitor_banner,
)


def test_visitor_banner_holds_after_event():
    tracker = VisitorBannerTracker(hold_frames=3)
    assert tracker.tick("visitor") == "VISITOR IN FRAME"
    assert tracker.tick("") == "VISITOR IN FRAME"
    assert tracker.tick("") == "VISITOR IN FRAME"
    assert tracker.tick("") == ""


def test_compose_monitor_prioritizes_visitor():
    visitor = VisitorBannerTracker(hold_frames=5)
    rules = RuleBannerTracker()
    smoking = SmokingBannerTracker()
    banner = compose_monitor_banner(
        active_alerts="low_engagement",
        presence_event="visitor",
        state_message="DISTRACTED",
        visitor_tracker=visitor,
        rule_tracker=rules,
        smoking_tracker=smoking,
        new_rule_messages=[],
    )
    assert banner == "VISITOR IN FRAME"


def test_compose_monitor_shows_smoking_banner():
    visitor = VisitorBannerTracker()
    rules = RuleBannerTracker()
    smoking = SmokingBannerTracker(hold_frames=3)
    presence = PresenceFrame(events=("smoking",))
    banner = compose_monitor_banner(
        active_alerts="",
        presence_event="",
        state_message="",
        visitor_tracker=visitor,
        rule_tracker=rules,
        smoking_tracker=smoking,
        new_rule_messages=[],
        presence=presence,
    )
    assert banner == "SMOKING"


def test_rule_banner_from_new_alerts():
    visitor = VisitorBannerTracker()
    rules = RuleBannerTracker(hold_frames=2)
    smoking = SmokingBannerTracker()
    banner = compose_monitor_banner(
        active_alerts="",
        presence_event="",
        state_message="",
        visitor_tracker=visitor,
        rule_tracker=rules,
        smoking_tracker=smoking,
        new_rule_messages=["Fatigue spike detected"],
    )
    assert banner == "FATIGUE SPIKE DETECTED"


def test_compose_monitor_does_not_preview_unsustained_alerts():
    visitor = VisitorBannerTracker()
    rules = RuleBannerTracker()
    smoking = SmokingBannerTracker()
    banner = compose_monitor_banner(
        active_alerts="fatigue_spike",
        presence_event="",
        state_message="",
        visitor_tracker=visitor,
        rule_tracker=rules,
        smoking_tracker=smoking,
        new_rule_messages=[],
    )
    assert banner == ""


def test_build_monitor_subtitle_is_compact_for_pilot():
    presence = PresenceFrame(
        people=(PresenceBox("user", 0.2, 0.2, 0.5, 0.7, 0.9, is_primary=True),),
        objects=(PresenceBox("phone", 0.42, 0.45, 0.5, 0.62, 0.7),),
        primary_index=0,
    )

    subtitle = build_monitor_subtitle(presence, None, None, 0.4)

    assert subtitle == "Phone"
    assert "PHASE" not in subtitle
    assert "AUT" not in subtitle
