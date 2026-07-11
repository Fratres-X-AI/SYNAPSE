import argparse
import csv
from collections import deque
from datetime import datetime
from time import monotonic

import cv2

from src.adaptation.adaptive_agent import AdaptiveAgent
from src.cognition.fusion_state import FusionState
from src.cognition.profile_matcher import match_profile
from src.cognition.soft_scores import compute_soft_scores
from src.monitoring.alert_rules import MonitorAlertEngine
from src.monitoring.presence_logger import PresenceEventLogger
from src.perception.capture import CameraCapture
from src.perception.emotion_estimator import EmotionEstimator
from src.perception.frame_quality import assess_frame_quality, draw_quality_pill
from src.perception.presence_detector import PresenceTracker, display_label
from src.perception.shoulder_tracker import draw_shoulder_markers
from src.perception.state_estimator import StateEstimator
from src.visualization.alerts import StateAlertTracker
from src.visualization.dashboard import draw_profile_match_bars, render_fusion_dashboard
from src.visualization.debrief import run_session_debrief
from src.visualization.display_adapter import create_display_adapter
from src.visualization.landmark_overlay import draw_all_tracking_overlays
from src.visualization.monitor_hud import (
    RuleBannerTracker,
    VisitorBannerTracker,
    build_monitor_subtitle,
    compose_monitor_banner,
)
from src.visualization.presence_overlay import draw_presence_overlay
from src.visualization.timeline import draw_live_session_strip
from utils.config import Config
from utils.app_paths import cleanup_old_data
from utils.emotion_profile import EmotionProfile, load_emotion_profile
from utils.fps_tracker import FpsTracker
from utils.manager_report import write_manager_report
from utils.privacy import ensure_privacy_consent
from utils.settings import load_settings

SUMMARY_EVERY_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synapse production monitor")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen (F to toggle)")
    parser.add_argument("--skip-debrief", action="store_true", help="Close without post-session debrief")
    parser.add_argument("--debrief", action="store_true", help="Show post-session debrief on exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(fullscreen=args.fullscreen)
    settings = load_settings()
    if not ensure_privacy_consent():
        return

    config.session_dir.mkdir(parents=True, exist_ok=True)
    deleted = cleanup_old_data(settings.retention_days)
    if deleted:
        print(f"Cleaned up {len(deleted)} old local data file(s).")
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = config.session_dir / f"monitor_{session_name}.csv"
    alert_log_path = config.session_dir / f"monitor_{session_name}.alerts.csv"
    presence_log_path = config.session_dir / f"monitor_{session_name}.presence.log"

    profile = load_emotion_profile(config.emotion_profile_path) or EmotionProfile()
    try:
        camera = CameraCapture(camera_index=config.camera_index, detect_presence=True)
    except RuntimeError as error:
        print(f"Camera error: {error}")
        return
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
    fps_tracker = FpsTracker()
    presence_tracker = PresenceTracker()
    presence_event_logger = PresenceEventLogger(presence_log_path)
    visitor_banner = VisitorBannerTracker()
    rule_banner = RuleBannerTracker()
    live_rows: deque[dict] = deque(maxlen=240)
    show_landmarks = False
    started_at = monotonic()
    last_summary_at = started_at

    print("Synapse Monitor — production mode")
    print(f"Logging to {log_path}")
    print(f"Presence log: {presence_log_path}")
    if profile.has_neutral():
        print("Loaded emotion profile: N / H / S/S / Mad")
    else:
        print("No profile - run: python synapse_launcher.py onboard")
    print("Alert rules: low engagement 2m | high distraction 2m | fatigue 1m | tension 90s")
    print("Act naturally. Press 'q' to stop, 'f' for fullscreen, 'm' for landmark shell.")

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
                "face_count",
                "extra_people",
                "presence_labels",
                "presence_event",
            ]
        )
        alert_writer.writerow(["timestamp", "elapsed_sec", "rule_id", "message"])

        try:
            while True:
                frame, landmarks, presence = camera.get_frame_landmarks_presence()
                if frame is None:
                    if camera.failed_reads in (1, 30, 120):
                        print(f"Camera read failed ({camera.failed_reads} consecutive frames).")
                    continue

                now = monotonic()
                fusion = None
                active_alerts = ""
                presence_event = ""
                face_count = 0
                extra_people = 0
                presence_labels = ""
                new_rule_messages: list[str] = []
                face_detected = landmarks is not None
                if presence is not None:
                    face_count = presence.face_count
                    extra_people = presence.extra_people
                    presence_labels = ", ".join(
                        display_label(label) for label in sorted(presence.active_labels())
                    )
                    presence_event = presence_tracker.update(extra_people, now)
                    for line in presence_event_logger.update(presence.active_labels()):
                        print(f"[PRESENCE] {line}")
                    if presence_event == "visitor":
                        timestamp = datetime.now().isoformat(timespec="seconds")
                        elapsed = now - started_at
                        alert_writer.writerow(
                            [
                                timestamp,
                                f"{elapsed:.2f}",
                                "visitor_detected",
                                "Additional person in frame",
                            ]
                        )
                        alert_file.flush()
                        print(f"[PRESENCE] Additional person in frame at {elapsed:.0f}s")

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
                        new_rule_messages.append(alert.message)
                        alert_writer.writerow(
                            [alert.timestamp_iso, f"{alert.elapsed_sec:.2f}", alert.rule_id, alert.message]
                        )
                        alert_file.flush()

                    live_rows.append(
                        {
                            "state": cognitive.state.value,
                            "profile_phase": profile_phase or "",
                            "engagement": f"{soft.engagement:.3f}",
                        }
                    )

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
                            face_count,
                            extra_people,
                            presence_labels,
                            presence_event,
                        ]
                    )
                    log_file.flush()

                flash, _state_alert_message = alerts.update(
                    fusion.cognitive.state if fusion else None,
                    agent.autonomy_level if fusion else None,
                )
                draw_presence_overlay(frame, presence)
                draw_shoulder_markers(frame, camera.last_pose_landmarks, camera.last_shoulder_sample)
                if show_landmarks and landmarks is not None:
                    draw_all_tracking_overlays(frame, landmarks)
                banner = compose_monitor_banner(
                    active_alerts=active_alerts,
                    presence_event=presence_event,
                    state_message="",
                    visitor_tracker=visitor_banner,
                    rule_tracker=rule_banner,
                    new_rule_messages=new_rule_messages,
                )
                subtitle = build_monitor_subtitle(
                    presence,
                    camera.last_shoulder_sample,
                    fusion,
                    agent.autonomy_level if fusion else None,
                )
                frame = render_fusion_dashboard(
                    frame,
                    fusion,
                    ear_history,
                    estimator,
                    flash=flash,
                    alert_message=banner,
                    subtitle=subtitle,
                    fps=fps_tracker.tick(),
                )
                quality_message, quality_color = assess_frame_quality(frame, face_detected=face_detected)
                if fusion is not None:
                    height, width = frame.shape[:2]
                    draw_profile_match_bars(frame, fusion, (width - 168, 42))
                    if len(live_rows) > 3:
                        draw_live_session_strip(
                            frame,
                            list(live_rows),
                            (16, height - 22),
                            (min(220, width - 32), 14),
                            mode="profile",
                        )
                draw_quality_pill(frame, quality_message, quality_color)
                if landmarks is None:
                    cv2.putText(
                        frame,
                        "No face detected - center yourself and improve lighting",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 180, 255),
                        2,
                        cv2.LINE_AA,
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

                key = display.read_key(1)
                if key in (ord("m"), ord("M")):
                    show_landmarks = not show_landmarks
                    print(f"Landmark shell {'ON' if show_landmarks else 'OFF'}")
                elif display.try_handle_display_key(key):
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
                    export_desktop=settings.export_reports_to_desktop,
                )
                print("\n" + report)
                print(f"\nReport saved to {log_path.with_suffix('.report.txt')}")
                if settings.export_reports_to_desktop:
                    print("Copy exported to Desktop: Synapse_Report_*.txt")
            if args.debrief and not args.skip_debrief and log_path.stat().st_size > 120:
                run_session_debrief(
                    log_path,
                    alert_csv=alert_log_path,
                    presence_log=presence_log_path,
                )


if __name__ == "__main__":
    main()
