import cv2

from src.adaptation.adaptive_agent import AdaptiveAgent
from src.perception.capture import CameraCapture
from src.perception.state_estimator import StateEstimator
from src.visualization.display import render_status
from utils.config import Config


def main() -> None:
    config = Config()
    camera = CameraCapture(camera_index=config.camera_index)
    estimator = StateEstimator()
    agent = AdaptiveAgent()

    print("Starting Synapse closed loop... Press 'q' in the window to quit.")
    print("Signals: EAR, blinks, head pose, gaze -> cognitive state -> agent autonomy")

    try:
        while True:
            frame, landmarks = camera.get_frame_and_landmarks()
            if frame is None:
                continue

            cognitive_state = None
            behavior = None
            if landmarks is not None:
                cognitive_state = estimator.update(landmarks)
                agent.adapt(cognitive_state)
                behavior = agent.get_behavior()
                signals = cognitive_state.signals
                blink_label = "BLINKING" if signals["is_blinking"] else "open"

                print(
                    f"State: {cognitive_state.state.value} "
                    f"({cognitive_state.confidence:.0%}) | "
                    f"EAR: {signals['ear']:.3f} | "
                    f"Blinks: {signals['blink_counter']} ({signals['blink_rate']:.1f}/min) | "
                    f"Yaw: {signals['head_yaw']:+.1f} Pitch: {signals['head_pitch']:+.1f} | "
                    f"Gaze: {signals['gaze_direction']} | "
                    f"Autonomy: {agent.autonomy_level:.2f} | "
                    f"Eyes: {blink_label}",
                    end="\r",
                )

            frame = render_status(frame, cognitive_state, behavior)
            cv2.imshow(config.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        print("\nSynapse closed loop ended.")


if __name__ == "__main__":
    main()
