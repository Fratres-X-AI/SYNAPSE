import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from time import monotonic

from src.adaptation.adaptive_agent import AdaptiveAgent
from src.perception.capture import CameraCapture
from src.perception.state_estimator import StateEstimator
from src.visualization.alerts import StateAlertTracker
from src.visualization.dashboard import render_dashboard
from src.visualization.display_adapter import create_display_adapter
from utils.config import Config
from utils.session_summary import write_summary_report

SESSION_DIR = Path("sessions")
SUMMARY_EVERY_SECONDS = 30


def main() -> None:
    config = Config()
    SESSION_DIR.mkdir(exist_ok=True)
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = SESSION_DIR / f"silent_track_{session_name}.csv"

    camera = CameraCapture(camera_index=config.camera_index)
    estimator = StateEstimator(**config.estimator_kwargs())
    agent = AdaptiveAgent()
    alerts = StateAlertTracker(beep=False)
    display = create_display_adapter(config.display_mode, "Synapse - Silent Track")
    ear_history: deque[float] = deque(maxlen=180)

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "timestamp",
                "state",
                "confidence",
                "ear",
                "blink_count",
                "blink_rate",
                "head_yaw",
                "head_pitch",
                "gaze_direction",
                "gaze_x",
                "gaze_y",
                "distraction",
                "autonomy",
            ]
        )

        print(f"Silent tracking started. Logging to {log_path}")
        print("Act naturally. Press 'q' in the window to stop.")

        started_at = monotonic()
        last_summary_at = started_at
        previous_state = None

        try:
            while True:
                frame, landmarks = camera.get_frame_and_landmarks()
                if frame is None:
                    continue

                now = monotonic()
                cognitive_state = None

                if landmarks is not None:
                    cognitive_state = estimator.update(landmarks)
                    agent.adapt(cognitive_state)
                    ear_history.append(cognitive_state.signals["ear"])
                    signals = cognitive_state.signals

                    if cognitive_state.state != previous_state:
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"{previous_state.value if previous_state else 'start'} -> "
                            f"{cognitive_state.state.value} | "
                            f"autonomy {agent.autonomy_level:.2f}"
                        )
                        previous_state = cognitive_state.state

                    writer.writerow(
                        [
                            datetime.now().isoformat(timespec="seconds"),
                            cognitive_state.state.value,
                            f"{cognitive_state.confidence:.2f}",
                            f"{signals['ear']:.3f}",
                            signals["blink_counter"],
                            f"{signals['blink_rate']:.1f}",
                            f"{signals['head_yaw']:+.1f}",
                            f"{signals['head_pitch']:+.1f}",
                            signals["gaze_direction"],
                            f"{signals['gaze_x']:+.3f}",
                            f"{signals['gaze_y']:+.3f}",
                            estimator.distraction_score(signals),
                            f"{agent.autonomy_level:.2f}",
                        ]
                    )
                    log_file.flush()

                flash, alert_message = alerts.update(
                    cognitive_state.state if cognitive_state else None,
                    agent.autonomy_level if cognitive_state else None,
                )

                frame = render_dashboard(
                    frame,
                    cognitive_state,
                    ear_history,
                    estimator,
                    flash=flash,
                    alert_message="",
                )
                display.show(frame, cognitive_state, agent.autonomy_level)

                if now - last_summary_at >= SUMMARY_EVERY_SECONDS:
                    if cognitive_state is not None:
                        signals = cognitive_state.signals
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"tracking: {cognitive_state.state.value} | "
                            f"blinks {signals['blink_counter']} | "
                            f"gaze {signals['gaze_direction']} | "
                            f"distraction {estimator.distraction_score(signals)}%"
                        )
                    last_summary_at = now

                if display.poll_quit():
                    break
        finally:
            camera.release()
            display.close()
            elapsed = monotonic() - started_at
            print(f"\nSilent tracking ended after {elapsed:.0f}s")
            print(f"Session saved to {log_path}")
            if log_path.stat().st_size > 120:
                report = write_summary_report(log_path)
                print("\n" + report)
                print(f"\nSummary saved to {log_path.with_suffix('.summary.txt')}")
                print(f"Replay with: python replay_session.py \"{log_path}\"")


if __name__ == "__main__":
    main()
