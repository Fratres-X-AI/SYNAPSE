from collections import deque

from src.adaptation.adaptive_agent import AdaptiveAgent
from src.perception.capture import CameraCapture
from src.perception.state_estimator import StateEstimator
from src.visualization.alerts import StateAlertTracker
from src.visualization.dashboard import render_dashboard
from src.visualization.display_adapter import create_display_adapter
from utils.config import Config


def main() -> None:
    config = Config()
    camera = CameraCapture(camera_index=config.camera_index)
    estimator = StateEstimator(**config.estimator_kwargs())
    agent = AdaptiveAgent()
    alerts = StateAlertTracker()
    display = create_display_adapter(config.display_mode, config.window_name)
    ear_history: deque[float] = deque(maxlen=180)

    if config.estimator_kwargs():
        print("Loaded personal calibration.")

    print("Starting Synapse closed loop... Press 'q' in the window to quit.")
    print("Watch for border flashes and center alerts when your state changes.")

    try:
        while True:
            frame, landmarks = camera.get_frame_and_landmarks()
            if frame is None:
                continue

            cognitive_state = None
            if landmarks is not None:
                cognitive_state = estimator.update(landmarks)
                agent.adapt(cognitive_state)
                ear_history.append(cognitive_state.signals["ear"])
                signals = cognitive_state.signals
                blink_label = "BLINKING" if signals["is_blinking"] else "open"

                print(
                    f"State: {cognitive_state.state.value} "
                    f"({cognitive_state.confidence:.0%}) | "
                    f"EAR: {signals['ear']:.3f} | "
                    f"Gaze: {signals['gaze_direction']} | "
                    f"Distraction: {estimator.distraction_score(signals)}% | "
                    f"Autonomy: {agent.autonomy_level:.2f} | "
                    f"Eyes: {blink_label}",
                    end="\r",
                )

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
                alert_message=alert_message,
            )
            display.show(frame, cognitive_state, agent.autonomy_level)
            if display.poll_quit():
                break
    finally:
        camera.release()
        display.close()
        print("\nSynapse closed loop ended.")


if __name__ == "__main__":
    main()
