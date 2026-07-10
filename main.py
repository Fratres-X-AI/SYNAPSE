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

                print(
                    f"State: {cognitive_state.state.value} | "
                    f"Confidence: {cognitive_state.confidence:.2f} | "
                    f"EAR: {cognitive_state.signals['ear']:.3f} | "
                    f"Blink rate: {cognitive_state.signals['blink_rate']:.1f}/min | "
                    f"Yaw: {cognitive_state.signals['head_yaw']:.1f} | "
                    f"{behavior}"
                )

            frame = render_status(frame, cognitive_state, behavior)
            cv2.imshow(config.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()


if __name__ == "__main__":
    main()
