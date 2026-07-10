from collections import deque

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
from utils.emotion_profile import EmotionProfile, load_emotion_profile


def main() -> None:
    config = Config()
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
    alerts = StateAlertTracker()
    display = create_display_adapter(config.display_mode, config.window_name)
    ear_history: deque[float] = deque(maxlen=180)

    if config.estimator_kwargs():
        print("Loaded personal calibration.")
    if profile.has_neutral():
        print("Loaded emotion profile.")

    print("Starting Synapse fusion loop... Press 'q' in the window to quit.")

    try:
        while True:
            frame, landmarks = camera.get_frame_and_landmarks()
            if frame is None:
                continue

            fusion = None
            if landmarks is not None:
                cognitive = estimator.update(landmarks)
                emotion = emotion_estimator.update(landmarks, ear=cognitive.signals.get("ear"))
                profile_phase, profile_scores, profile_confidence = match_profile(
                    EmotionEstimator.snapshot_signals(emotion.signals),
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
                    profile_phase=profile_phase or "",
                    profile_scores=profile_scores,
                    profile_confidence=profile_confidence,
                )

                signals = cognitive.signals
                print(
                    f"Match {profile_phase or '—'} ({profile_confidence:.0%}) | "
                    f"Eng {soft.engagement:.0%} | Ten {soft.tension:.0%} | "
                    f"State: {cognitive.state.value} | "
                    f"Distraction: {estimator.distraction_score(signals)}%",
                    end="\r",
                )

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
                alert_message=alert_message,
            )
            display.show(frame, fusion.cognitive if fusion else None, agent.autonomy_level)
            if display.poll_quit():
                break
    finally:
        camera.release()
        display.close()
        print("\nSynapse fusion loop ended.")


if __name__ == "__main__":
    main()
