import cv2
import mediapipe as mp
import numpy as np

from src.perception.state_estimator import StateEstimator
from utils.calibration import load_calibration

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LEFT_EYE = StateEstimator.LEFT_EYE
RIGHT_EYE = StateEstimator.RIGHT_EYE
EYE_INDICES = set(LEFT_EYE + RIGHT_EYE)

estimator_kwargs = {}
profile = load_calibration()
if profile is not None:
    estimator_kwargs = profile.to_estimator_kwargs()
    print("Loaded calibration from calibration.json")

estimator = StateEstimator(**estimator_kwargs)
cap = cv2.VideoCapture(0)

print("Starting EAR test... Press 'q' to quit.")
print("Try blinking and holding your eyes open/closed to see EAR change.")


def draw_eye_landmarks(frame, landmarks, eye_indices, color):
    height, width = frame.shape[:2]
    points = []
    for index in eye_indices:
        landmark = landmarks[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 2, color, -1)

    cv2.polylines(frame, [np.array(points[:4], dtype=np.int32)], False, color, 1)
    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        cognitive_state = estimator.update(face_landmarks)
        signals = cognitive_state.signals

        draw_eye_landmarks(frame, face_landmarks, LEFT_EYE, (0, 255, 255))
        draw_eye_landmarks(frame, face_landmarks, RIGHT_EYE, (0, 255, 255))

        blink_label = "BLINKING" if signals["is_blinking"] else "eyes open"
        overlay_lines = [
            f"EAR: {signals['ear']:.3f}",
            f"Mean EAR: {signals['mean_ear']:.3f}",
            f"Blink rate: {signals['blink_rate']:.1f}/min",
            f"Blink count: {signals['blink_counter']}",
            f"Close thresh: {estimator._dynamic_close_threshold():.3f}",
            f"Status: {blink_label}",
        ]

        for index, line in enumerate(overlay_lines):
            cv2.putText(
                frame,
                line,
                (16, 32 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        print(
            f"EAR: {signals['ear']:.3f} | "
            f"Blinks: {signals['blink_counter']} | "
            f"Rate: {signals['blink_rate']:.1f}/min | "
            f"{blink_label}",
            end="\r",
        )
    else:
        cv2.putText(
            frame,
            "No face detected",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Synapse - EAR Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("\nEAR test ended.")
