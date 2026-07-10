import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.cognition.cognitive_state import CognitiveState, State

SESSION_DIR = Path("sessions")


@dataclass(frozen=True)
class SessionRecord:
    timestamp: datetime
    state: State
    confidence: float
    signals: dict
    distraction: int
    autonomy: float

    def to_cognitive_state(self) -> CognitiveState:
        return CognitiveState(
            state=self.state,
            confidence=self.confidence,
            signals=self.signals,
        )


def list_sessions(directory: Path = SESSION_DIR) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("silent_track_*.csv"), key=lambda path: path.stat().st_mtime)


def latest_session(directory: Path = SESSION_DIR) -> Path | None:
    sessions = list_sessions(directory)
    return sessions[-1] if sessions else None


def load_session(path: Path) -> list[SessionRecord]:
    records: list[SessionRecord] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            records.append(
                SessionRecord(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    state=State(row["state"]),
                    confidence=float(row["confidence"]),
                    signals={
                        "ear": float(row["ear"]),
                        "mean_ear": float(row["ear"]),
                        "is_blinking": False,
                        "blink_rate": float(row["blink_rate"]),
                        "blink_counter": int(row["blink_count"]),
                        "head_yaw": float(row["head_yaw"]),
                        "head_pitch": float(row["head_pitch"]),
                        "gaze_direction": row["gaze_direction"],
                        "gaze_x": float(row["gaze_x"]),
                        "gaze_y": float(row["gaze_y"]),
                    },
                    distraction=int(row["distraction"]),
                    autonomy=float(row["autonomy"]),
                )
            )
    return records
