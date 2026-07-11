from src.visualization.monitor_hud import (
    RuleBannerTracker,
    VisitorBannerTracker,
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
    banner = compose_monitor_banner(
        active_alerts="low_engagement",
        presence_event="visitor",
        state_message="DISTRACTED",
        visitor_tracker=visitor,
        rule_tracker=rules,
        new_rule_messages=[],
    )
    assert banner == "VISITOR IN FRAME"


def test_rule_banner_from_new_alerts():
    visitor = VisitorBannerTracker()
    rules = RuleBannerTracker(hold_frames=2)
    banner = compose_monitor_banner(
        active_alerts="",
        presence_event="",
        state_message="",
        visitor_tracker=visitor,
        rule_tracker=rules,
        new_rule_messages=["Fatigue spike detected"],
    )
    assert banner == "FATIGUE SPIKE DETECTED"
