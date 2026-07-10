import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from time import monotonic

from src.adaptation.adaptive_agent import AdaptiveAgent
from src.cognition.fusion_state import FusionState
from src.cognition.profile_matcher import match_profile
from src.cognition.soft_scores import compute_soft_scores
from src.perception.capture import CameraCapture
from src.perception.emotion_estimator import EmotionEstimator
from src.perception.state_estimator import StateEstimator
from src.visualization.alerts import StateAlertTracker
from src.visualization.dashboard import render_fusion_dashboard
from src.visualization.display_adapter import create_display_adapter
from utils.config import Config
from utils.emotion_profile import (
    DEFAULT_EMOTION_PROFILE_PATH,
    EmotionProfile,
    load_emotion_profile,
    save_emotion_profile,
)
from utils.fusion_summary import write_fusion_summary

SESSION_DIR = Path("sessions")

PHASE_KEYS = {
    ord("n"): "neutral",
    ord("N"): "neutral",
    ord("h"): "happy",
    ord("H"): "happy",
    ord("s"): "sad",
    ord("S"): "sad",
    ord("m"): "mad",
    ord("M"): "mad",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synapse fusion track")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen (F to toggle)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(fullscreen=args.fullscreen)
    SESSION_DIR.mkdir(exist_ok=True)
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = SESSION_DIR / f"fusion_track_{session_name}.csv"

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
    display = create_display_adapter(
        config.display_mode,
        "Synapse - Fusion Track",
        fullscreen=config.fullscreen,
    )
    ear_history: deque[float] = deque(maxlen=180)
    labeled_phase = ""
    last_signals: dict | None = None
    started_at = monotonic()

    print("Synapse Fusion Track started.")
    print(f"Logging to {log_path}")
    print("Keys: N=neutral H=happy S=sad/stressed M=mad | Q=quit")
    print("Soft scores: engagement, fatigue, tension, positivity")

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "timestamp",
                "elapsed_sec",
                "labeled_phase",
                "state",
                "confidence",
                "emotion",
                "emotion_confidence",
                "profile_phase",
                "profile_confidence",
                "profile_neutral",
                "profile_happy",
                "profile_sad",
                "profile_mad",
                "engagement",
                "fatigue",
                "tension",
                "positivity",
                "distraction",
                "autonomy",
                "ear",
                "blink_count",
                "blink_rate",
                "head_yaw",
                "head_pitch",
                "gaze_direction",
                "gaze_x",
                "gaze_y",
                "smile_score",
                "smile_delta",
                "cheek_raise",
                "cheek_delta",
                "brow_raise",
                "brow_furrow",
                "brow_inner_pinch",
                "mouth_open",
            ]
        )

        try:
            while True:
                frame, landmarks = camera.get_frame_and_landmarks()
                if frame is None:
                    continue

                fusion = None
                if landmarks is not None:
                    cognitive = estimator.update(landmarks)
                    emotion = emotion_estimator.update(landmarks, ear=cognitive.signals.get("ear"))
                    last_signals = emotion.signals
                    profile_phase, profile_scores, profile_confidence = match_profile(
                        EmotionEstimator.snapshot_signals(last_signals),
                        profile,
                    )
                    soft = compute_soft_scores(
                        cognitive,
                        emotion.signals,
                        estimator.distraction_score(cognitive.signals),
                        profile,
                    )
                    agent.adapt(cognitive)
                    ear_history.append(cognitive.signals["ear"])
                    fusion = FusionState.build(
                        cognitive,
                        emotion,
                        soft,
                        estimator,
                        labeled_phase=labeled_phase,
                        profile_phase=profile_phase or "",
                        profile_scores=profile_scores,
                        profile_confidence=profile_confidence,
                    )

                    elapsed = monotonic() - started_at
                    sig = emotion.signals
                    cog = cognitive.signals
                    writer.writerow(
                        [
                            datetime.now().isoformat(timespec="seconds"),
                            f"{elapsed:.2f}",
                            labeled_phase,
                            cognitive.state.value,
                            f"{cognitive.confidence:.2f}",
                            emotion.emotion.value,
                            f"{emotion.confidence:.2f}",
                            profile_phase or "",
                            f"{profile_confidence:.2f}",
                            f"{profile_scores.get('neutral', 0.0):.3f}",
                            f"{profile_scores.get('happy', 0.0):.3f}",
                            f"{profile_scores.get('sad', 0.0):.3f}",
                            f"{profile_scores.get('mad', 0.0):.3f}",
                            f"{soft.engagement:.3f}",
                            f"{soft.fatigue:.3f}",
                            f"{soft.tension:.3f}",
                            f"{soft.positivity:.3f}",
                            estimator.distraction_score(cog),
                            f"{agent.autonomy_level:.2f}",
                            f"{cog['ear']:.4f}",
                            cog["blink_counter"],
                            f"{cog['blink_rate']:.1f}",
                            f"{cog['head_yaw']:+.2f}",
                            f"{cog['head_pitch']:+.2f}",
                            cog["gaze_direction"],
                            f"{cog['gaze_x']:+.3f}",
                            f"{cog['gaze_y']:+.3f}",
                            f"{sig['smile_score']:.4f}",
                            f"{sig.get('smile_delta', 0.0):.4f}",
                            f"{sig['cheek_raise']:.4f}",
                            f"{sig.get('cheek_delta', 0.0):.4f}",
                            f"{sig['brow_raise']:.4f}",
                            f"{sig['brow_furrow']:.4f}",
                            f"{sig['brow_inner_pinch']:.4f}",
                            f"{sig['mouth_open']:.4f}",
                        ]
                    )
                    log_file.flush()

                flash, alert_message = alerts.update(
                    fusion.cognitive.state if fusion else None,
                    agent.autonomy_level if fusion else None,
                )
                frame = render_fusion_dashboard(
                    frame,
                    fusion,
                    ear_history,
                    estimator,
                    flash=flash,
                    alert_message="",
                )
                display.show(frame, fusion.cognitive if fusion else None, agent.autonomy_level)

                key = display.read_key()
                if key in PHASE_KEYS and last_signals is not None:
                    labeled_phase = PHASE_KEYS[key]
                    snapshot = EmotionEstimator.snapshot_signals(last_signals)
                    profile.set_phase(labeled_phase, snapshot)
                    save_emotion_profile(profile, config.emotion_profile_path)
                    if labeled_phase == "neutral":
                        emotion_estimator.set_neutral_baseline(
                            last_signals["smile_score"],
                            last_signals["cheek_raise"],
                        )
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Saved {labeled_phase} "
                        f"to {config.emotion_profile_path}"
                    )
                elif key in PHASE_KEYS:
                    print("Face not detected - cannot capture phase yet.")

                if display.try_handle_display_key(key):
                    break
        finally:
            camera.release()
            display.close()
            elapsed = monotonic() - started_at
            print(f"\nFusion track ended after {elapsed:.0f}s")
            if log_path.stat().st_size > 120:
                report = write_fusion_summary(log_path)
                print("\n" + report)


if __name__ == "__main__":
    main()
