import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.perception.state_estimator import StateEstimator
from src.visualization.dashboard import render_dashboard
from src.visualization.display_adapter import create_display_adapter
from utils.config import Config
from utils.session_log import latest_session, load_session
from utils.session_summary import summarize_latest_or_path, write_summary_report

FRAME_SIZE = (960, 540)
PLAYBACK_FPS = 24


def build_ear_history(records, index: int) -> deque[float]:
    history: deque[float] = deque(maxlen=180)
    start = max(0, index - 179)
    for record in records[start : index + 1]:
        history.append(record.signals["ear"])
    return history


def draw_replay_overlay(frame, record, index: int, total: int, paused: bool):
    status = "PAUSED" if paused else "PLAYING"
    lines = [
        f"REPLAY {index + 1}/{total} | {status}",
        f"Time: {record.timestamp.strftime('%H:%M:%S')}",
        f"Autonomy: {record.autonomy:.2f}",
    ]
    y = frame.shape[0] - 90
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)
        y += 24

    progress = (index + 1) / max(total, 1)
    bar_width = frame.shape[1] - 32
    cv2.rectangle(
        frame,
        (16, frame.shape[0] - 24),
        (16 + bar_width, frame.shape[0] - 10),
        (40, 40, 40),
        -1,
    )
    cv2.rectangle(
        frame,
        (16, frame.shape[0] - 24),
        (16 + int(bar_width * progress), frame.shape[0] - 10),
        (0, 200, 255),
        -1,
    )
    return frame


def replay_session(session_path: Path) -> None:
    records = load_session(session_path)
    if not records:
        print(f"No records to replay in {session_path}")
        return

    report = write_summary_report(session_path)
    print(report)
    print(f"\nSummary saved to {session_path.with_suffix('.summary.txt')}")

    config = Config()
    estimator = StateEstimator(**config.estimator_kwargs())
    display = create_display_adapter(config.display_mode, "Synapse - Session Replay")
    index = 0
    paused = False
    last_step = time.perf_counter()

    print("\nReplay controls: SPACE pause/resume | LEFT/RIGHT step | q quit")

    while True:
        frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        record = records[index]
        cognitive_state = record.to_cognitive_state()
        ear_history = build_ear_history(records, index)

        frame = render_dashboard(frame, cognitive_state, ear_history, estimator, flash=False)
        frame = draw_replay_overlay(frame, record, index, len(records), paused)
        display.show(frame, cognitive_state, record.autonomy)

        key = display.read_key()
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == 81 or key == 2:
            index = max(0, index - 1)
        if key == 83 or key == 3:
            index = min(len(records) - 1, index + 1)

        if not paused:
            now = time.perf_counter()
            if now - last_step >= 1.0 / PLAYBACK_FPS:
                index = min(len(records) - 1, index + 1)
                last_step = now
                if index >= len(records) - 1:
                    paused = True

    display.close()
    print("Replay ended.")


def main() -> None:
    session_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_session()
    if session_path is None or not session_path.exists():
        print("No session file found. Run test_silent_track.py first.")
        raise SystemExit(1)
    replay_session(session_path)


if __name__ == "__main__":
    main()
