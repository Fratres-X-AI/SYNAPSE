"""Unified onboarding: attention calibration + emotion profile capture."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import cv2
import mediapipe as mp
import numpy as np

from src.perception.capture import CameraCapture
from src.perception.emotion_estimator import EmotionEstimator
from src.perception.state_estimator import StateEstimator
from utils.calibration import build_profile, save_calibration
from utils.config import Config
from utils.emotion_profile import (
    ACTIVE_PHASES,
    EmotionProfile,
    PHASE_LABELS,
    save_emotion_profile,
)

mp_face_mesh = mp.solutions.face_mesh

# --- Part 1: Attention calibration (from test_calibrate.py) ---


@dataclass
class CalStep:
    step_id: str
    prompt: str
    duration: float | None = None


CALIBRATION_STEPS = [
    CalStep("center", "Step 1/5: Look at the center of your screen", 4.0),
    CalStep("blink", "Step 2/5: Blink naturally", 5.0),
    CalStep("head_turn", "Step 3/5: Turn head LEFT and RIGHT, then press SPACE", None),
    CalStep("corners", "Step 4/5: Glance at all 4 screen corners, then press SPACE", None),
    CalStep("away", "Step 5/5: Look UP off-screen, then press SPACE", None),
]

# --- Part 2: Expression profile capture ---


@dataclass
class ExpressionStep:
    phase: str
    key: str
    prompt: str


EXPRESSION_STEPS = [
    ExpressionStep("neutral", "N", "Relax your face (neutral), then press N"),
    ExpressionStep("happy", "H", "Smile naturally (happy), then press H"),
    ExpressionStep("sad", "S", "Show sad or stressed expression, then press S"),
    ExpressionStep("mad", "M", "Show anger or frustration (mad), then press M"),
]

PHASE_KEYS = {
    ord("n"): "neutral",
    ord("N"): "neutral",
    ord("h"): "happy",
    ord("H"): "happy",
    ord("s"): "sad",
    ord("S"): "sad",
    ord("m"): "mad",
    ord("M"): "mad",
}


def draw_text_block(frame, lines: list[str], y_start: int = 32, color=(0, 255, 255)):
    y = y_start
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 30
    return frame


def draw_calibration_overlay(frame, step: CalStep, progress: float, ear: float | None):
    lines = [
        "SYNAPSE ONBOARDING — Part 1: Attention Calibration",
        step.prompt,
        f"Progress: {int(progress * 100)}%",
    ]
    if ear is not None:
        lines.append(f"Live EAR: {ear:.3f}")
    if step.duration is None:
        lines.append("Press SPACE when finished with this step")
    lines.append("Press Q to quit")
    frame = draw_text_block(frame, lines)

    bar_width = frame.shape[1] - 32
    filled = int(bar_width * progress)
    y = 32 + len(lines) * 30
    cv2.rectangle(frame, (16, y), (16 + bar_width, y + 18), (50, 50, 50), -1)
    cv2.rectangle(frame, (16, y), (16 + filled, y + 18), (0, 220, 120), -1)
    return frame


def run_calibration() -> dict | None:
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    estimator = StateEstimator()
    cap = cv2.VideoCapture(0)

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

    print("SYNAPSE ONBOARDING — Part 1: Attention Calibration")
    print(f"-> {CALIBRATION_STEPS[0].prompt}")
    print("Press Q in the window to quit early.\n")

    def collect_signals(landmarks):
        ear = estimator._calculate_ear(landmarks)
        yaw = estimator._estimate_head_yaw(landmarks)
        pitch = estimator._estimate_head_pitch(landmarks)
        gaze_x, gaze_y = estimator._estimate_gaze(landmarks)
        gaze_mag = max(abs(gaze_x), abs(gaze_y))
        return ear, yaw, pitch, gaze_mag, gaze_y

    def update_samples(step_id, ear, yaw, pitch, gaze_mag, gaze_y):
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

    completed = False

    while step_index < len(CALIBRATION_STEPS):
        ret, frame = cap.read()
        if not ret:
            break

        step = CALIBRATION_STEPS[step_index]
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
                step_index += 1
                step_started_at = monotonic()
                awaiting_space = False
                if step_index < len(CALIBRATION_STEPS):
                    print(f"-> {CALIBRATION_STEPS[step_index].prompt}")
                continue

        frame = draw_calibration_overlay(frame, step, progress, ear_value)
        cv2.imshow("Synapse - Onboarding", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and awaiting_space and progress >= 0.35:
            step_index += 1
            step_started_at = monotonic()
            awaiting_space = False
            if step_index < len(CALIBRATION_STEPS):
                print(f"-> {CALIBRATION_STEPS[step_index].prompt}")
            else:
                completed = True
                break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

    if completed or step_index >= len(CALIBRATION_STEPS):
        print("\nPart 1 complete.\n")
        return samples

    print("\nCalibration cancelled before completion.")
    return None


def draw_expression_overlay(
    frame,
    step: ExpressionStep,
    captured: set[str],
    last_signals: dict | None,
):
    lines = [
        "SYNAPSE ONBOARDING — Part 2: Expression Profile",
        step.prompt,
        "",
        "Captured:",
    ]
    for phase in ACTIVE_PHASES:
        mark = "[x]" if phase in captured else "[ ]"
        key = {"neutral": "N", "happy": "H", "sad": "S", "mad": "M"}[phase]
        label = PHASE_LABELS[phase]
        lines.append(f"  {mark} {key} = {label}")

    lines.append("")
    if last_signals is not None:
        lines.append(f"Smile: {last_signals['smile_score']:.3f} | Brow: {last_signals['brow_furrow']:.3f}")
    else:
        lines.append("Face not detected — center yourself in frame")
    lines.append("Press Q to quit (progress is lost)")
    return draw_text_block(frame, lines, color=(180, 255, 180))


def run_expression_capture(config: Config) -> EmotionProfile | None:
    camera = CameraCapture(camera_index=config.camera_index)
    emotion_estimator = EmotionEstimator(calibration_frames=0)
    profile = EmotionProfile()
    captured: set[str] = set()
    step_index = 0
    last_signals: dict | None = None

    print("SYNAPSE ONBOARDING — Part 2: Expression Profile")
    print(f"-> {EXPRESSION_STEPS[0].prompt}")
    print("Hold each expression, then press the matching key.\n")

    try:
        while step_index < len(EXPRESSION_STEPS):
            frame, landmarks = camera.get_frame_and_landmarks()
            if frame is None:
                continue

            if landmarks is not None:
                emotion = emotion_estimator.update(landmarks)
                last_signals = emotion.signals

            step = EXPRESSION_STEPS[step_index]
            frame = draw_expression_overlay(frame, step, captured, last_signals)
            cv2.imshow("Synapse - Onboarding", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None

            expected_key = ord(step.key.lower())
            if key in (expected_key, ord(step.key)):
                if last_signals is None:
                    print("Face not detected — cannot capture yet.")
                    continue

                snapshot = EmotionEstimator.snapshot_signals(last_signals)
                profile.set_phase(step.phase, snapshot)
                captured.add(step.phase)

                if step.phase == "neutral":
                    emotion_estimator.set_neutral_baseline(
                        last_signals["smile_score"],
                        last_signals["cheek_raise"],
                    )

                print(f"Captured {PHASE_LABELS[step.phase]} ({step.key})")
                step_index += 1
                if step_index < len(EXPRESSION_STEPS):
                    print(f"-> {EXPRESSION_STEPS[step_index].prompt}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(captured) == len(EXPRESSION_STEPS):
        print("\nPart 2 complete.\n")
        return profile

    print("\nExpression capture cancelled before completion.")
    return None


def main() -> None:
    config = Config()

    print("=" * 56)
    print("  SYNAPSE ONBOARDING WIZARD")
    print("  Part 1: Attention calibration (5 steps)")
    print("  Part 2: Expression profile (N / H / S / M)")
    print("=" * 56)
    print()

    samples = run_calibration()
    if samples is None:
        return

    profile = run_expression_capture(config)
    if profile is None:
        return

    calibration = build_profile(samples)
    save_calibration(calibration, config.calibration_path)
    save_emotion_profile(profile, config.emotion_profile_path)

    print("Onboarding complete!")
    print(f"  calibration.json  -> {config.calibration_path.resolve()}")
    print(f"  emotion_profile.json -> {config.emotion_profile_path.resolve()}")
    print()
    print("Next: python synapse_launcher.py monitor")


if __name__ == "__main__":
    main()
