"""Showcase mode: full monitor stack with every landmark overlay enabled."""

from __future__ import annotations

import argparse
from collections import deque

import cv2

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
from src.visualization.landmark_overlay import draw_all_tracking_overlays, showcase_subtitle
from src.visualization.hud_text import HUD_LABEL, draw_hud_text
from utils.config import Config
from utils.emotion_profile import EmotionProfile, load_emotion_profile
from utils.fps_tracker import FpsTracker
from utils.privacy import ensure_privacy_consent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synapse full-feature showcase")
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Start windowed (default is fullscreen for demos)",
    )
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen (F to toggle)")
    return parser.parse_args()


def _resolve_fullscreen(args: argparse.Namespace) -> bool:
    return not args.windowed


def main() -> None:
    args = parse_args()
    if not ensure_privacy_consent():
        return

    config = Config(fullscreen=_resolve_fullscreen(args))
    profile = load_emotion_profile(config.emotion_profile_path) or EmotionProfile()
    estimator = StateEstimator(**config.estimator_kwargs())
    emotion_estimator = EmotionEstimator(calibration_frames=0)
    agent = AdaptiveAgent()
    alerts = StateAlertTracker(beep=False, quiet=True)
    camera = CameraCapture(config.camera_index)
    display = create_display_adapter(
        config.display_mode,
        "Synapse - Showcase (All Features)",
        fullscreen=config.fullscreen,
    )
    ear_history: deque[float] = deque(maxlen=180)
    fps_tracker = FpsTracker()

    print("Synapse showcase running - all landmarks + flight instrument HUD.")
    print("Press Q to quit, F for fullscreen.")

    try:
        while True:
            frame, landmarks = camera.get_frame_and_landmarks()
            if frame is None:
                continue

            fusion = None
            if landmarks is not None:
                draw_all_tracking_overlays(frame, landmarks)
                cognitive = estimator.update(landmarks)
                emotion = emotion_estimator.update(landmarks, ear=cognitive.signals.get("ear"))
                profile_phase, profile_scores, profile_confidence = match_profile(
                    EmotionEstimator.snapshot_signals(emotion.signals),
                    profile,
                )
                distraction = estimator.distraction_score(cognitive.signals)
                soft = compute_soft_scores(cognitive, emotion.signals, distraction, profile)
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
            landmark_note = (
                showcase_subtitle(len(landmarks), cognitive.signals)
                if landmarks is not None
                else ""
            )
            flash, _alert_message = alerts.update(
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
                subtitle=landmark_note,
                fps=fps_tracker.tick(),
            )

            if landmarks is None:
                draw_hud_text(frame, "Center in frame", (16, 40), size=12, color=HUD_LABEL)

            display.show(frame, fusion.cognitive if fusion else None, agent.autonomy_level)
            if display.poll_quit():
                break
    finally:
        camera.release()
        display.close()
        print("Showcase ended.")


if __name__ == "__main__":
    main()
