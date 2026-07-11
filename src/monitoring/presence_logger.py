"""Timestamped presence event log — e.g. `7:17 AM | User | Vape`."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic

from src.perception.presence_detector import display_label


@dataclass
class PresenceEventLogger:
    """Write human-readable presence lines when sustained labels change."""

    log_path: Path
    sustain_seconds: float = 1.5
    _pending_since: dict[str, float] = field(default_factory=dict)
    _sustained: set[str] = field(default_factory=set)
    _last_snapshot: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text("# Synapse presence log\n", encoding="utf-8")

    def update(self, labels: set[str], when: datetime | None = None) -> list[str]:
        now = monotonic()
        when = when or datetime.now()
        visible = {label.strip().lower() for label in labels if label}

        for label in list(self._pending_since):
            if label not in visible:
                self._pending_since.pop(label, None)

        for label in visible:
            if label not in self._pending_since:
                self._pending_since[label] = now

        sustained = {
            label
            for label, started in self._pending_since.items()
            if now - started >= self.sustain_seconds
        }
        self._sustained = sustained

        snapshot = tuple(sorted(display_label(label) for label in sustained))
        if not snapshot or snapshot == self._last_snapshot:
            return []

        self._last_snapshot = snapshot
        time_text = when.strftime("%I:%M %p").lstrip("0").replace(" 0", " ")
        line = f"{time_text} | {' | '.join(snapshot)}"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return [line]
