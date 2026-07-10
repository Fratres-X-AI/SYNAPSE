from src.monitoring import alert_rules
from src.monitoring.alert_rules import AlertRule, MonitorAlertEngine


def test_alert_engine_waits_for_sustain_window(monkeypatch):
    now = 100.0
    monkeypatch.setattr(alert_rules, "monotonic", lambda: now)
    engine = MonitorAlertEngine(rules=(AlertRule("low_engagement", "Low engagement", 5.0),))

    active, alerts = engine.evaluate(
        elapsed_sec=0,
        engagement=0.20,
        fatigue=0.10,
        tension=0.10,
        distraction=0,
        timestamp_iso="2026-01-01T00:00:00",
    )

    assert active == "low_engagement"
    assert alerts == []

    now = 106.0
    active, alerts = engine.evaluate(
        elapsed_sec=6,
        engagement=0.20,
        fatigue=0.10,
        tension=0.10,
        distraction=0,
        timestamp_iso="2026-01-01T00:00:06",
    )

    assert active == "low_engagement"
    assert [alert.rule_id for alert in alerts] == ["low_engagement"]
    assert engine.summary_flags() == ["Low engagement at 6s"]


def test_alert_engine_does_not_fire_same_rule_twice(monkeypatch):
    monkeypatch.setattr(alert_rules, "monotonic", lambda: 50.0)
    engine = MonitorAlertEngine(rules=(AlertRule("high_tension", "High tension", 0.0),))

    first_active, first_alerts = engine.evaluate(1, 0.8, 0.1, 0.9, 0, "t1")
    second_active, second_alerts = engine.evaluate(2, 0.8, 0.1, 0.9, 0, "t2")

    assert first_active == "high_tension"
    assert [alert.rule_id for alert in first_alerts] == ["high_tension"]
    assert second_active == "high_tension"
    assert second_alerts == []
    assert len(engine.triggered) == 1


def test_alert_engine_resets_condition_timer_when_signal_recovers(monkeypatch):
    times = iter([10.0, 12.0, 20.0])
    monkeypatch.setattr(alert_rules, "monotonic", lambda: next(times))
    engine = MonitorAlertEngine(rules=(AlertRule("fatigue_spike", "Fatigue", 5.0),))

    engine.evaluate(0, 0.8, 0.9, 0.1, 0, "t0")
    active, recovered_alerts = engine.evaluate(2, 0.8, 0.2, 0.1, 0, "t2")
    active_again, new_alerts = engine.evaluate(10, 0.8, 0.9, 0.1, 0, "t10")

    assert active == ""
    assert recovered_alerts == []
    assert active_again == "fatigue_spike"
    assert new_alerts == []
