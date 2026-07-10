import cv2
import mediapipe as mp
import numpy as np

from src.perception.state_estimator import StateEstimator

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

estimator = StateEstimator()
cap = cv2.VideoCapture(0)

print("Starting gaze test... Press 'q' to quit.")
print("Keep your head still and move only your eyes through the checklist.")


def draw_gaze(frame, landmarks):
    height, width = frame.shape[:2]

    for eye_indices, iris_index, color in (
        (StateEstimator.LEFT_EYE, StateEstimator.LEFT_IRIS, (0, 255, 255)),
        (StateEstimator.RIGHT_EYE, StateEstimator.RIGHT_IRIS, (0, 255, 255)),
    ):
        outer = landmarks[eye_indices[0]]
        inner = landmarks[eye_indices[3]]
        iris = landmarks[iris_index]

        outer_pt = (int(outer.x * width), int(outer.y * height))
        inner_pt = (int(inner.x * width), int(inner.y * height))
        iris_pt = (int(iris.x * width), int(iris.y * height))

        cv2.line(frame, outer_pt, inner_pt, color, 1)
        cv2.circle(frame, iris_pt, 5, (0, 0, 255), -1)

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

        draw_gaze(frame, face_landmarks)

        overlay_lines = [
            f"Gaze X: {signals['gaze_x']:+.3f}",
            f"Gaze Y: {signals['gaze_y']:+.3f}",
            f"Direction: {signals['gaze_direction']}",
            f"State: {cognitive_state.state.value}",
            "Checklist: center -> left -> right -> up -> down",
            "Tip: keep head still, move eyes only",
        ]

        for index, line in enumerate(overlay_lines):
            color = (0, 255, 0)
            if cognitive_state.state.value == "distracted":
                color = (0, 140, 255)
            cv2.putText(
                frame,
                line,
                (16, 32 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        print(
            f"Gaze: {signals['gaze_direction']} "
            f"({signals['gaze_x']:+.3f}, {signals['gaze_y']:+.3f}) | "
            f"State: {cognitive_state.state.value}",
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

    cv2.imshow("Synapse - Gaze Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("\nGaze test ended.")
