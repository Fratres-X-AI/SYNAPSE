from collections import deque

import cv2
import mediapipe as mp

from src.perception.state_estimator import StateEstimator
from src.visualization.alerts import StateAlertTracker
from src.visualization.dashboard import render_dashboard
from utils.calibration import load_calibration

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

estimator_kwargs = {}
profile = load_calibration()
if profile is not None:
    estimator_kwargs = profile.to_estimator_kwargs()
    print("Loaded calibration from calibration.json")

estimator = StateEstimator(**estimator_kwargs)
alerts = StateAlertTracker()
cap = cv2.VideoCapture(0)
ear_history: deque[float] = deque(maxlen=180)

print("Starting Synapse dashboard... Press 'q' to quit.")
print("Watch for border flashes and center alerts when your state changes.")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    cognitive_state = None
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        cognitive_state = estimator.update(face_landmarks)
        ear_history.append(cognitive_state.signals["ear"])

    flash, alert_message = alerts.update(
        cognitive_state.state if cognitive_state else None
    )

    frame = render_dashboard(
        frame,
        cognitive_state,
        ear_history,
        estimator,
        flash=flash,
        alert_message=alert_message,
    )
    cv2.imshow("Synapse - Dashboard", frame)

    if cognitive_state is not None:
        signals = cognitive_state.signals
        print(
            f"State: {cognitive_state.state.value} | "
            f"EAR: {signals['ear']:.3f} | "
            f"Gaze: {signals['gaze_direction']} | "
            f"Distraction: {estimator.distraction_score(signals)}%",
            end="\r",
        )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("\nDashboard ended.")
