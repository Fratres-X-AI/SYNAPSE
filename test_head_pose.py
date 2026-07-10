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

POSE_LANDMARKS = (
    StateEstimator.NOSE_TIP,
    StateEstimator.FOREHEAD,
    StateEstimator.CHIN,
    StateEstimator.LEFT_FACE,
    StateEstimator.RIGHT_FACE,
)

estimator = StateEstimator()
cap = cv2.VideoCapture(0)

print("Starting head pose test... Press 'q' to quit.")
print("Run through the pose checklist shown on screen.")


def draw_pose_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = []
    for index in POSE_LANDMARKS:
        landmark = landmarks[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, (255, 180, 0), -1)

    nose, forehead, chin, left_face, right_face = points
    cv2.line(frame, forehead, chin, (255, 180, 0), 1)
    cv2.line(frame, left_face, right_face, (255, 180, 0), 1)
    cv2.circle(frame, nose, 6, (0, 255, 255), -1)
    return frame


def pose_hint(yaw: float, pitch: float, state_value: str) -> str:
    if state_value == "distracted":
        return "DISTRACTED - head turned away"
    if abs(yaw) < 8 and abs(pitch) < 8:
        return "CENTERED - facing camera"
    if yaw <= -12:
        return "Turning LEFT"
    if yaw >= 12:
        return "Turning RIGHT"
    if pitch <= -8:
        return "Looking UP"
    if pitch >= 8:
        return "Looking DOWN"
    return "ADJUSTING"


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
        hint = pose_hint(
            signals["head_yaw"],
            signals["head_pitch"],
            cognitive_state.state.value,
        )

        draw_pose_landmarks(frame, face_landmarks)

        overlay_lines = [
            f"Yaw: {signals['head_yaw']:+.1f} deg",
            f"Pitch: {signals['head_pitch']:+.1f} deg",
            f"State: {cognitive_state.state.value}",
            f"Hint: {hint}",
            "Checklist: center -> left -> right -> up -> down",
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
            f"Yaw: {signals['head_yaw']:+.1f} | "
            f"Pitch: {signals['head_pitch']:+.1f} | "
            f"State: {cognitive_state.state.value} | "
            f"{hint}",
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

    cv2.imshow("Synapse - Head Pose Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("\nHead pose test ended.")
