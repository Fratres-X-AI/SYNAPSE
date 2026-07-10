from dataclasses import dataclass, field
from time import monotonic

from src.cognition.cognitive_state import State


@dataclass(frozen=True)
class AlertRule:
    rule_id: str
    message: str
    sustain_seconds: float


DEFAULT_RULES = (
    AlertRule("low_engagement", "Low engagement sustained", 120.0),
    AlertRule("high_distraction", "High distraction sustained", 120.0),
    AlertRule("fatigue_spike", "Fatigue spike detected", 60.0),
    AlertRule("high_tension", "Elevated tension sustained", 90.0),
)


@dataclass
class TriggeredAlert:
    rule_id: str
    message: str
    elapsed_sec: float
    timestamp_iso: str


@dataclass
class MonitorAlertEngine:
    engagement_threshold: float = 0.40
    distraction_threshold: int = 70
    fatigue_threshold: float = 0.55
    tension_threshold: float = 0.50
    rules: tuple[AlertRule, ...] = DEFAULT_RULES
    _active_since: dict[str, float] = field(default_factory=dict)
    _fired: set[str] = field(default_factory=set)
    triggered: list[TriggeredAlert] = field(default_factory=list)

    def evaluate(
        self,
        elapsed_sec: float,
        engagement: float,
        fatigue: float,
        tension: float,
        distraction: int,
        timestamp_iso: str,
    ) -> tuple[str, list[TriggeredAlert]]:
        now = monotonic()
        conditions = {
            "low_engagement": engagement < self.engagement_threshold,
            "high_distraction": distraction > self.distraction_threshold,
            "fatigue_spike": fatigue > self.fatigue_threshold,
            "high_tension": tension > self.tension_threshold,
        }

        new_alerts: list[TriggeredAlert] = []
        active_labels: list[str] = []

        for rule in self.rules:
            if not conditions.get(rule.rule_id, False):
                self._active_since.pop(rule.rule_id, None)
                continue

            if rule.rule_id not in self._active_since:
                self._active_since[rule.rule_id] = now

            active_for = now - self._active_since[rule.rule_id]
            active_labels.append(rule.rule_id)

            if active_for >= rule.sustain_seconds and rule.rule_id not in self._fired:
                alert = TriggeredAlert(
                    rule_id=rule.rule_id,
                    message=rule.message,
                    elapsed_sec=elapsed_sec,
                    timestamp_iso=timestamp_iso,
                )
                self.triggered.append(alert)
                self._fired.add(rule.rule_id)
                new_alerts.append(alert)

        return ";".join(active_labels), new_alerts

    def summary_flags(self) -> list[str]:
        return [f"{alert.message} at {alert.elapsed_sec:.0f}s" for alert in self.triggered]
