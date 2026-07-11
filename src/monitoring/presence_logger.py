"""Timestamped presence event log — sustained episodes + hourly phone totals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic

from src.perception.presence_detector import display_label

LOGGABLE_LABELS = frozenset({"phone", "smoking", "visitor", "person"})

DEFAULT_LABEL_SUSTAIN = {
    "phone": 10.0,
    "smoking": 2.0,
    "visitor": 3.0,
    "person": 5.0,
}


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def _hour_key(when: datetime) -> tuple[int, int, int, int]:
    return (when.year, when.month, when.day, when.hour)


def _hour_range_label(hour_key: tuple[int, int, int, int]) -> str:
    year, month, day, hour = hour_key
    start = datetime(year, month, day, hour)
    end = start + timedelta(hours=1)
    start_text = start.strftime("%I %p").lstrip("0").replace(" 0", " ")
    end_text = end.strftime("%I %p").lstrip("0").replace(" 0", " ")
    return f"{start_text}\u2013{end_text}"


@dataclass
class PresenceEventLogger:
    """Log sustained presence episodes and hourly phone visibility totals."""

    log_path: Path
    sustain_seconds: float = 3.0
    label_sustain: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LABEL_SUSTAIN))
    _pending_since: dict[str, float] = field(default_factory=dict)
    _episode_started: dict[str, float] = field(default_factory=dict)
    _logged_active: set[str] = field(default_factory=set)
    _phone_hour_seconds: dict[tuple[int, int, int, int], float] = field(default_factory=dict)
    _active_hour_key: tuple[int, int, int, int] | None = None
    _phone_was_visible: bool = False
    _last_tick_at: float | None = None
    _last_tick_when: datetime | None = None

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text("# Synapse presence log (sustained episodes)\n", encoding="utf-8")

    def _threshold(self, label: str) -> float:
        return self.label_sustain.get(label, self.sustain_seconds)

    def update(self, labels: set[str], when: datetime | None = None) -> list[str]:
        now = monotonic()
        when = when or datetime.now()
        raw_visible = {label.strip().lower() for label in labels if label}
        visible = {label for label in raw_visible if label in LOGGABLE_LABELS}

        lines: list[str] = []
        lines.extend(self._track_phone_time("phone" in raw_visible, now, when))

        dropped = set(self._pending_since) - visible
        for label in sorted(dropped):
            self._pending_since.pop(label, None)

        for label in visible:
            if label not in self._pending_since:
                self._pending_since[label] = now

        sustained = {
            label
            for label, started in self._pending_since.items()
            if now - started >= self._threshold(label)
        }

        for label in sorted(sustained - self._logged_active):
            self._logged_active.add(label)
            self._episode_started[label] = self._pending_since.get(label, now)
            lines.append(self._write_line(when, display_label(label)))

        ended = self._logged_active - sustained
        for label in sorted(ended):
            started = self._episode_started.pop(label, now)
            duration = _format_duration(now - started)
            lines.append(self._write_line(when, f"{display_label(label)} ended ({duration})"))
            self._logged_active.discard(label)

        return lines

    def finalize(self, when: datetime | None = None) -> list[str]:
        """Flush the active hour's phone total when a session ends."""
        when = when or datetime.now()
        lines = self._track_phone_time(False, monotonic(), when, flush_active_hour=True)
        self._pending_since.clear()
        self._logged_active.clear()
        self._episode_started.clear()
        return lines

    def _track_phone_time(
        self,
        phone_visible: bool,
        now: float,
        when: datetime,
        *,
        flush_active_hour: bool = False,
    ) -> list[str]:
        lines: list[str] = []
        if self._last_tick_at is not None and self._phone_was_visible and self._last_tick_when is not None:
            elapsed = now - self._last_tick_at
            hour_key = _hour_key(self._last_tick_when)
            self._phone_hour_seconds[hour_key] = self._phone_hour_seconds.get(hour_key, 0.0) + elapsed

        current_hour = _hour_key(when)
        if self._active_hour_key is not None and current_hour != self._active_hour_key:
            lines.extend(self._flush_hour_summary(self._active_hour_key, when))

        if flush_active_hour and self._active_hour_key is not None:
            lines.extend(self._flush_hour_summary(self._active_hour_key, when))
            self._active_hour_key = None
        else:
            self._active_hour_key = current_hour

        self._phone_was_visible = phone_visible
        self._last_tick_at = now
        self._last_tick_when = when
        return lines

    def _flush_hour_summary(
        self,
        hour_key: tuple[int, int, int, int],
        when: datetime,
    ) -> list[str]:
        seconds = self._phone_hour_seconds.pop(hour_key, 0.0)
        if seconds < 1.0:
            return []
        hour_label = _hour_range_label(hour_key)
        message = f"Phone this hour ({hour_label}): {_format_duration(seconds)} total"
        return [self._write_line(when, message)]

    def _write_line(self, when: datetime, message: str) -> str:
        time_text = when.strftime("%I:%M %p").lstrip("0").replace(" 0", " ")
        line = f"{time_text} | {message}"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return line
