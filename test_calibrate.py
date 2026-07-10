from dataclasses import dataclass
from time import monotonic

import cv2
import mediapipe as mp
import numpy as np

from src.perception.state_estimator import StateEstimator
from utils.calibration import build_profile, save_calibration

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

estimator = StateEstimator()
cap = cv2.VideoCapture(0)


@dataclass
class Step:
    step_id: str
    prompt: str
    duration: float | None = None


STEPS = [
    Step("center", "Step 1/5: Look at the center of your screen", 4.0),
    Step("blink", "Step 2/5: Blink naturally", 5.0),
    Step("head_turn", "Step 3/5: Turn head LEFT and RIGHT, then press SPACE", None),
    Step("corners", "Step 4/5: Glance at all 4 screen corners, then press SPACE", None),
    Step("away", "Step 5/5: Look UP off-screen, then press SPACE", None),
]

samples = {
    "center_ear": 0.0,
    "blink_ear": 1.0,
    "head_yaw": 0.0,
    "head_pitch": 0.0,
    "head_pitch_up": 0.0,
    "corner_gaze": 0.0,
    "away_gaze": 0.0,
    "gaze_up": 0.0,
}
center_ears: list[float] = []
step_index = 0
step_started_at = monotonic()
awaiting_space = False

print("Starting calibration wizard... Press 'q' to quit early.")


def current_step() -> Step:
    return STEPS[step_index]


def advance_step():
    global step_index, step_started_at, awaiting_space
    step_index += 1
    step_started_at = monotonic()
    awaiting_space = False
    if step_index < len(STEPS):
        print(f"\n-> {STEPS[step_index].prompt}")


def collect_signals(landmarks):
    ear = estimator._calculate_ear(landmarks)
    yaw = estimator._estimate_head_yaw(landmarks)
    pitch = estimator._estimate_head_pitch(landmarks)
    gaze_x, gaze_y = estimator._estimate_gaze(landmarks)
    gaze_mag = max(abs(gaze_x), abs(gaze_y))
    return ear, yaw, pitch, gaze_mag, gaze_y


def update_samples(
    step_id: str,
    ear: float,
    yaw: float,
    pitch: float,
    gaze_mag: float,
    gaze_y: float,
):
    if step_id == "center":
        center_ears.append(ear)
        samples["center_ear"] = float(np.mean(center_ears))
    elif step_id == "blink":
        samples["blink_ear"] = min(samples["blink_ear"], ear)
    elif step_id == "head_turn":
        samples["head_yaw"] = max(samples["head_yaw"], abs(yaw))
        samples["head_pitch"] = max(samples["head_pitch"], abs(pitch))
        if pitch < 0:
            samples["head_pitch_up"] = max(samples["head_pitch_up"], abs(pitch))
    elif step_id == "corners":
        samples["corner_gaze"] = max(samples["corner_gaze"], gaze_mag)
    elif step_id == "away":
        samples["away_gaze"] = max(samples["away_gaze"], gaze_mag)
        if gaze_y < 0:
            samples["gaze_up"] = max(samples["gaze_up"], abs(gaze_y))


def draw_overlay(frame, step: Step, progress: float, ear: float | None):
    lines = [
        "SYNAPSE CALIBRATION",
        step.prompt,
        f"Progress: {int(progress * 100)}%",
    ]
    if ear is not None:
        lines.append(f"Live EAR: {ear:.3f}")

    if step.duration is None:
        lines.append("Press SPACE when finished with this step")

    y = 32
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        y += 30

    bar_width = frame.shape[1] - 32
    filled = int(bar_width * progress)
    cv2.rectangle(frame, (16, y), (16 + bar_width, y + 18), (50, 50, 50), -1)
    cv2.rectangle(frame, (16, y), (16 + filled, y + 18), (0, 220, 120), -1)
    return frame


while step_index < len(STEPS):
    ret, frame = cap.read()
    if not ret:
        break

    step = current_step()
    elapsed = monotonic() - step_started_at
    ear_value = None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        ear_value, yaw, pitch, gaze_mag, gaze_y = collect_signals(landmarks)
        update_samples(step.step_id, ear_value, yaw, pitch, gaze_mag, gaze_y)

    if step.duration is None:
        progress = min(1.0, elapsed / 8.0)
        awaiting_space = True
        if elapsed >= 8.0:
            progress = 1.0
    else:
        progress = min(1.0, elapsed / step.duration)
        if elapsed >= step.duration:
            advance_step()
            continue

    frame = draw_overlay(frame, step, progress, ear_value)
    cv2.imshow("Synapse - Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" ") and awaiting_space and progress >= 0.35:
        advance_step()
        if step_index >= len(STEPS):
            break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()

if step_index >= len(STEPS):
    profile = build_profile(samples)
    save_calibration(profile)
    print("\nCalibration saved to calibration.json")
    print(profile)
else:
    print("\nCalibration cancelled before completion.")
