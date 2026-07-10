import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from time import monotonic

import cv2

from src.adaptation.adaptive_agent import AdaptiveAgent
from src.cognition.fusion_state import FusionState
from src.cognition.profile_matcher import match_profile
from src.cognition.soft_scores import compute_soft_scores
from src.monitoring.alert_rules import MonitorAlertEngine
from src.perception.capture import CameraCapture
from src.perception.emotion_estimator import EmotionEstimator
from src.perception.state_estimator import StateEstimator
from src.visualization.alerts import StateAlertTracker
from src.visualization.dashboard import render_fusion_dashboard
from src.visualization.display_adapter import create_display_adapter
from utils.config import Config
from utils.emotion_profile import EmotionProfile, load_emotion_profile
from utils.manager_report import write_manager_report

SESSION_DIR = Path("sessions")
SUMMARY_EVERY_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synapse production monitor")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen (F to toggle)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(fullscreen=args.fullscreen)
    SESSION_DIR.mkdir(exist_ok=True)
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = SESSION_DIR / f"monitor_{session_name}.csv"
    alert_log_path = SESSION_DIR / f"monitor_{session_name}.alerts.csv"

    profile = load_emotion_profile(config.emotion_profile_path) or EmotionProfile()
    camera = CameraCapture(camera_index=config.camera_index)
    estimator = StateEstimator(**config.estimator_kwargs())
    emotion_estimator = EmotionEstimator(calibration_frames=0)
    if profile.has_neutral():
        emotion_estimator.set_neutral_baseline(
            profile.neutral.get("smile_score", 0.0),
            profile.neutral.get("cheek_raise", 0.0),
        )

    agent = AdaptiveAgent()
    alerts = StateAlertTracker(beep=False, quiet=True)
    monitor_alerts = MonitorAlertEngine()
    display = create_display_adapter(
        config.display_mode,
        "Synapse - Monitor",
        fullscreen=config.fullscreen,
    )
    ear_history: deque[float] = deque(maxlen=180)
    started_at = monotonic()
    last_summary_at = started_at

    print("Synapse Monitor — production mode")
    print(f"Logging to {log_path}")
    if profile.has_neutral():
        print("Loaded emotion profile: N / H / S/S / Mad")
    else:
        print("No profile — run: python test_onboard.py")
    print("Alert rules: low engagement 2m | high distraction 2m | fatigue 1m | tension 90s")
    print("Act naturally. Press 'q' to stop, 'f' for fullscreen.")

    with log_path.open("w", newline="", encoding="utf-8") as log_file, alert_log_path.open(
        "w", newline="", encoding="utf-8"
    ) as alert_file:
        writer = csv.writer(log_file)
        alert_writer = csv.writer(alert_file)
        writer.writerow(
            [
                "timestamp",
                "elapsed_sec",
                "state",
                "confidence",
                "profile_phase",
                "profile_confidence",
                "engagement",
                "fatigue",
                "tension",
                "positivity",
                "distraction",
                "autonomy",
                "active_alerts",
                "ear",
                "blink_count",
                "blink_rate",
                "head_yaw",
                "head_pitch",
                "gaze_direction",
                "gaze_x",
                "gaze_y",
                "profile_neutral",
                "profile_happy",
                "profile_sad",
                "profile_mad",
            ]
        )
        alert_writer.writerow(["timestamp", "elapsed_sec", "rule_id", "message"])

        try:
            while True:
                frame, landmarks = camera.get_frame_and_landmarks()
                if frame is None:
                    continue

                now = monotonic()
                fusion = None
                active_alerts = ""

                if landmarks is not None:
                    cognitive = estimator.update(landmarks)
                    emotion = emotion_estimator.update(landmarks, ear=cognitive.signals.get("ear"))
                    profile_phase, profile_scores, profile_confidence = match_profile(
                        EmotionEstimator.snapshot_signals(emotion.signals),
                        profile,
                    )
                    distraction = estimator.distraction_score(cognitive.signals)
                    soft = compute_soft_scores(
                        cognitive,
                        emotion.signals,
                        distraction,
                        profile,
                    )
                    agent.adapt(cognitive)
                    ear_history.append(cognitive.signals["ear"])
                    fusion = FusionState.build(
                        cognitive,
                        emotion,
                        soft,
                        estimator,
                        profile_phase=profile_phase or "",
                        profile_scores=profile_scores,
                        profile_confidence=profile_confidence,
                    )

                    elapsed = monotonic() - started_at
                    timestamp = datetime.now().isoformat(timespec="seconds")
                    active_alerts, new_alerts = monitor_alerts.evaluate(
                        elapsed,
                        soft.engagement,
                        soft.fatigue,
                        soft.tension,
                        distraction,
                        timestamp,
                    )
                    for alert in new_alerts:
                        print(f"[ALERT] {alert.message} at {alert.elapsed_sec:.0f}s")
                        alert_writer.writerow(
                            [alert.timestamp_iso, f"{alert.elapsed_sec:.2f}", alert.rule_id, alert.message]
                        )
                        alert_file.flush()

                    cog = cognitive.signals
                    writer.writerow(
                        [
                            timestamp,
                            f"{elapsed:.2f}",
                            cognitive.state.value,
                            f"{cognitive.confidence:.2f}",
                            profile_phase or "",
                            f"{profile_confidence:.2f}",
                            f"{soft.engagement:.3f}",
                            f"{soft.fatigue:.3f}",
                            f"{soft.tension:.3f}",
                            f"{soft.positivity:.3f}",
                            distraction,
                            f"{agent.autonomy_level:.2f}",
                            active_alerts,
                            f"{cog['ear']:.4f}",
                            cog["blink_counter"],
                            f"{cog['blink_rate']:.1f}",
                            f"{cog['head_yaw']:+.2f}",
                            f"{cog['head_pitch']:+.2f}",
                            cog["gaze_direction"],
                            f"{cog['gaze_x']:+.3f}",
                            f"{cog['gaze_y']:+.3f}",
                            f"{profile_scores.get('neutral', 0.0):.3f}",
                            f"{profile_scores.get('happy', 0.0):.3f}",
                            f"{profile_scores.get('sad', 0.0):.3f}",
                            f"{profile_scores.get('mad', 0.0):.3f}",
                        ]
                    )
                    log_file.flush()

                flash, _ = alerts.update(
                    fusion.cognitive.state if fusion else None,
                    agent.autonomy_level if fusion else None,
                )
                frame = render_fusion_dashboard(
                    frame,
                    fusion,
                    ear_history,
                    estimator,
                    flash=flash,
                )
                if fusion and active_alerts:
                    cv2.putText(
                        frame,
                        f"ALERT WATCH: {active_alerts[:60]}",
                        (16, frame.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 140, 255),
                        1,
                    )
                display.show(frame, fusion.cognitive if fusion else None, agent.autonomy_level)

                if fusion and now - last_summary_at >= SUMMARY_EVERY_SECONDS:
                    soft = fusion.soft
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"eng {soft.engagement:.0%} | fat {soft.fatigue:.0%} | "
                        f"ten {soft.tension:.0%} | match {fusion.profile_phase or '—'} | "
                        f"state {fusion.cognitive.state.value}"
                    )
                    last_summary_at = now

                if display.poll_quit():
                    break
        finally:
            camera.release()
            display.close()
            elapsed = monotonic() - started_at
            print(f"\nMonitor ended after {elapsed:.0f}s")
            print(f"Session saved to {log_path}")
            if log_path.stat().st_size > 120:
                report = write_manager_report(
                    log_path,
                    alert_flags=monitor_alerts.summary_flags(),
                    export_desktop=True,
                )
                print("\n" + report)
                print(f"\nReport saved to {log_path.with_suffix('.report.txt')}")
                desktop = Path.home() / "Desktop"
                print(f"Copy exported to Desktop: Synapse_Report_*.txt")


if __name__ == "__main__":
    main()
