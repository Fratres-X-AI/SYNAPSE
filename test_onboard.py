"""Unified onboarding: attention calibration + emotion profile capture."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

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
from synapse_launcher import relaunch_home, wants_return_home
from utils.user_profiles import get_active_user_display_name
from utils.privacy import ensure_privacy_consent
from utils.onboarding_progress import (
    clear_progress,
    load_progress,
    restore_profile,
    save_progress,
)
from src.ui.theme import HUD_BGR

try:
    from src.ui.dialogs import show_camera_error
except ImportError:
    show_camera_error = None

mp_face_mesh = mp.solutions.face_mesh

TOTAL_STEPS = 9
CALIBRATION_COUNT = 5
EXPRESSION_COUNT = 4


@dataclass
class CalStep:
    step_id: str
    prompt: str


@dataclass
class ExpressionStep:
    phase: str
    key: str
    prompt: str


CALIBRATION_STEPS = [
    CalStep("center", "Look at the center of your screen"),
    CalStep("blink", "Blink naturally"),
    CalStep("head_turn", "Turn head LEFT and RIGHT"),
    CalStep("corners", "Glance at all 4 screen corners"),
    CalStep("away", "Look UP off-screen"),
]

EXPRESSION_STEPS = [
    ExpressionStep("neutral", "N", "Relax your face (neutral)"),
    ExpressionStep("happy", "H", "Smile naturally (happy)"),
    ExpressionStep("sad", "S", "Show sad or stressed expression"),
    ExpressionStep("mad", "M", "Show anger or frustration (mad)"),
]

ONBOARDING_FAQ = [
    "SYNAPSE ONBOARDING — Quick guide",
    "",
    "One continuous setup — 9 steps, one webcam session.",
    "Processing stays on this computer — no video is saved.",
    "",
    "Steps 1-5 — Attention calibration",
    "  Follow each prompt, then press SPACE when finished.",
    "",
    "Steps 6-9 — Expression snapshots",
    "  Press N / H / S / M to start each face, then SPACE to capture.",
    "",
    "Take your time. Press SPACE to begin step 1.",
    "Quit anytime with Q — progress is saved for this profile.",
    "Press Q to quit",
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


def draw_text_block(frame, lines: list[str], y_start: int = 32, color=None):
    if color is None:
        color = HUD_BGR["accent"]
    y = y_start
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 30
    return frame


def _read_window_key() -> int:
    key = cv2.waitKey(30)
    if key == -1:
        return 0
    return key & 0xFF


def frame_quality(frame, face_detected: bool) -> tuple[str, tuple[int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    if not face_detected:
        return "Face not detected - center yourself in frame", (0, 140, 255)
    if brightness < 65:
        return "Lighting low - add light in front of your face", (0, 140, 255)
    if brightness > 215:
        return "Lighting too bright - reduce glare", (0, 140, 255)
    if contrast < 22:
        return "Image is flat - improve lighting or camera angle", (0, 140, 255)
    return "Webcam quality good", (80, 220, 100)


def draw_faq_overlay(
    frame,
    quality_message: str,
    quality_color: tuple[int, int, int],
):
    lines = list(ONBOARDING_FAQ)
    lines.insert(1, quality_message)
    frame = draw_text_block(frame, lines, color=HUD_BGR["ink"])
    cv2.circle(frame, (frame.shape[1] - 32, 32), 10, quality_color, -1)
    return frame


def run_onboarding_faq(cap, face_mesh) -> bool:
    print("SYNAPSE ONBOARDING — Quick guide")
    for line in ONBOARDING_FAQ:
        if line.strip():
            print(line)
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        face_detected = bool(results.multi_face_landmarks)
        quality_message, quality_color = frame_quality(frame, face_detected)

        frame = draw_faq_overlay(frame, quality_message, quality_color)
        cv2.imshow("Synapse - Onboarding", frame)

        key = _read_window_key()
        if key == ord("q"):
            return False
        if key == ord(" "):
            return True


def draw_calibration_overlay(
    frame,
    step: CalStep,
    step_number: int,
    ear: float | None,
    quality_message: str,
    quality_color: tuple[int, int, int],
):
    lines = [
        f"SYNAPSE ONBOARDING — Step {step_number}/{TOTAL_STEPS}",
        f"Step {step_number}/{TOTAL_STEPS}: {step.prompt}",
        quality_message,
        "Press SPACE when finished with this step",
        "Press Q to quit",
    ]
    if ear is not None:
        lines.insert(3, f"Live EAR: {ear:.3f}")
    frame = draw_text_block(frame, lines)
    cv2.circle(frame, (frame.shape[1] - 32, 32), 10, quality_color, -1)
    return frame


def draw_expression_overlay(
    frame,
    step: ExpressionStep,
    step_number: int,
    captured: set[str],
    last_signals: dict | None,
    *,
    armed: bool,
    status_message: str = "",
):
    lines = [
        f"SYNAPSE ONBOARDING — Step {step_number}/{TOTAL_STEPS}",
        f"Step {step_number}/{TOTAL_STEPS}: {step.prompt}",
        f"Press {step.key}, then SPACE to capture",
        "",
        "Captured:",
    ]
    for phase in ACTIVE_PHASES:
        mark = "[x]" if phase in captured else "[ ]"
        key = {"neutral": "N", "happy": "H", "sad": "S", "mad": "M"}[phase]
        label = PHASE_LABELS[phase]
        lines.append(f"  {mark} {key} = {label}")

    lines.append("")
    if status_message:
        lines.append(status_message)
    elif armed:
        lines.append(f"Ready — hold {PHASE_LABELS[step.phase]}, press SPACE to capture")
    elif last_signals is not None:
        lines.append(f"Smile: {last_signals['smile_score']:.3f} | Brow: {last_signals['brow_furrow']:.3f}")
    else:
        lines.append("Face not detected — center yourself in frame")
    lines.append("Press Q to quit (progress is saved — resume later)")
    return draw_text_block(frame, lines, color=HUD_BGR["accent"])


def _expression_arm_key(step: ExpressionStep, key: int) -> bool:
    return key in (ord(step.key.lower()), ord(step.key))


def _expression_confirm_key(key: int) -> bool:
    return key == ord(" ")


def run_hold_screen(cap, lines: list[str], *, title: str = "Synapse - Onboarding") -> bool:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    failed_reads = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            failed_reads += 1
            if failed_reads > 90:
                return False
            key = _read_window_key()
            if key == ord("q"):
                return False
            continue
        failed_reads = 0

        frame = draw_text_block(frame, [*lines, "", "Press SPACE to continue", "Press Q to quit"], color=HUD_BGR["accent"])
        cv2.imshow(title, frame)

        key = _read_window_key()
        if key == ord("q"):
            return False
        if key == ord(" "):
            return True


def _persist_progress(
    *,
    cal_index: int,
    expr_index: int,
    captured: set[str],
    expr_armed: bool,
    samples: dict,
    center_ears: list[float],
    profile: EmotionProfile,
) -> None:
    save_progress(
        cal_index=cal_index,
        expr_index=expr_index,
        captured=sorted(captured),
        expr_armed=expr_armed,
        samples=samples,
        center_ears=center_ears,
        profile=profile,
    )


def run_onboarding(
    config: Config,
) -> tuple[dict | None, EmotionProfile | None, cv2.VideoCapture | None]:
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    state_estimator = StateEstimator()
    emotion_estimator = EmotionEstimator(calibration_frames=0)
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        message = f"Could not open webcam index {config.camera_index}."
        print(message)
        if show_camera_error is not None:
            show_camera_error(message)
        face_mesh.close()
        return None, None, None

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
    profile = EmotionProfile()
    captured: set[str] = set()
    cal_index = 0
    expr_index = 0
    expr_armed = False
    last_signals: dict | None = None
    status_message = ""
    cancelled = False

    cv2.namedWindow("Synapse - Onboarding", cv2.WINDOW_NORMAL)

    def cleanup() -> None:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()

    def collect_signals(landmarks):
        ear = state_estimator._calculate_ear(landmarks)
        yaw = state_estimator._estimate_head_yaw(landmarks)
        pitch = state_estimator._estimate_head_pitch(landmarks)
        gaze_x, gaze_y = state_estimator._estimate_gaze(landmarks)
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

    if not run_onboarding_faq(cap, face_mesh):
        cleanup()
        print("\nOnboarding cancelled.")
        return None, None, None

    saved = load_progress()
    if saved is not None:
        cal_index = int(saved.get("cal_index", 0))
        expr_index = int(saved.get("expr_index", 0))
        captured = set(saved.get("captured") or [])
        expr_armed = bool(saved.get("expr_armed", False))
        samples.update(saved.get("samples") or {})
        center_ears.extend(float(value) for value in (saved.get("center_ears") or []))
        profile = restore_profile(saved)
        print(f"Resuming onboarding from step {cal_index + expr_index + 1}/{TOTAL_STEPS}.")
    else:
        print("SYNAPSE ONBOARDING — 9 steps (calibration + expressions)")

    if cal_index < len(CALIBRATION_STEPS):
        print(f"-> Step {cal_index + 1}/{TOTAL_STEPS}: {CALIBRATION_STEPS[cal_index].prompt}")
    elif expr_index < len(EXPRESSION_STEPS):
        print(
            f"-> Step {CALIBRATION_COUNT + expr_index + 1}/{TOTAL_STEPS}: "
            f"{EXPRESSION_STEPS[expr_index].prompt}"
        )
    print("Calibration: press SPACE when finished. Expressions: letter then SPACE.\n")

    while cal_index < len(CALIBRATION_STEPS) or expr_index < len(EXPRESSION_STEPS):
        ret, frame = cap.read()
        if not ret:
            cancelled = True
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        face_detected = bool(results.multi_face_landmarks)
        landmarks = results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else None
        quality_message, quality_color = frame_quality(frame, face_detected)

        if cal_index < len(CALIBRATION_STEPS):
            step = CALIBRATION_STEPS[cal_index]
            ear_value = None
            if landmarks is not None:
                ear_value, yaw, pitch, gaze_mag, gaze_y = collect_signals(landmarks)
                update_samples(step.step_id, ear_value, yaw, pitch, gaze_mag, gaze_y)

            frame = draw_calibration_overlay(
                frame,
                step,
                cal_index + 1,
                ear_value,
                quality_message,
                quality_color,
            )
            cv2.imshow("Synapse - Onboarding", frame)

            key = _read_window_key()
            if key == ord("q"):
                cancelled = True
                break
            if key == ord(" "):
                cal_index += 1
                _persist_progress(
                    cal_index=cal_index,
                    expr_index=expr_index,
                    captured=captured,
                    expr_armed=expr_armed,
                    samples=samples,
                    center_ears=center_ears,
                    profile=profile,
                )
                if cal_index < len(CALIBRATION_STEPS):
                    print(f"-> Step {cal_index + 1}/{TOTAL_STEPS}: {CALIBRATION_STEPS[cal_index].prompt}")
                elif expr_index < len(EXPRESSION_STEPS):
                    print(f"-> Step {CALIBRATION_COUNT + 1}/{TOTAL_STEPS}: {EXPRESSION_STEPS[0].prompt}")
            continue

        step = EXPRESSION_STEPS[expr_index]
        step_number = CALIBRATION_COUNT + expr_index + 1

        if landmarks is not None:
            emotion = emotion_estimator.update(landmarks)
            last_signals = emotion.signals
            if status_message.startswith("Face not detected"):
                status_message = ""

        frame = draw_expression_overlay(
            frame,
            step,
            step_number,
            captured,
            last_signals,
            armed=expr_armed,
            status_message=status_message,
        )
        cv2.imshow("Synapse - Onboarding", frame)

        key = _read_window_key()
        if key == ord("q"):
            cancelled = True
            break
        if key == 0:
            continue

        if not expr_armed:
            if _expression_arm_key(step, key):
                expr_armed = True
                status_message = ""
                print(f"Armed {PHASE_LABELS[step.phase]} — press SPACE to capture")
            elif key in PHASE_KEYS:
                status_message = f"Press {step.key}, then SPACE for {PHASE_LABELS[step.phase]}"
            continue

        if _expression_confirm_key(key):
            if last_signals is None:
                status_message = "Face not detected — hold expression and try again"
                print("Face not detected — cannot capture yet.")
                continue

            snapshot = EmotionEstimator.snapshot_signals(last_signals)
            profile.set_phase(step.phase, snapshot)
            captured.add(step.phase)
            status_message = ""
            expr_armed = False

            if step.phase == "neutral":
                emotion_estimator.set_neutral_baseline(
                    last_signals["smile_score"],
                    last_signals["cheek_raise"],
                )

            print(f"Captured {PHASE_LABELS[step.phase]} ({step.key})")
            expr_index += 1
            _persist_progress(
                cal_index=cal_index,
                expr_index=expr_index,
                captured=captured,
                expr_armed=expr_armed,
                samples=samples,
                center_ears=center_ears,
                profile=profile,
            )
            if expr_index < len(EXPRESSION_STEPS):
                print(f"-> Step {CALIBRATION_COUNT + expr_index + 1}/{TOTAL_STEPS}: {EXPRESSION_STEPS[expr_index].prompt}")
            continue

        if _expression_arm_key(step, key):
            continue

        if key in PHASE_KEYS:
            status_message = f"Press {step.key}, then SPACE for {PHASE_LABELS[step.phase]}"

    if cancelled or cal_index < len(CALIBRATION_STEPS) or expr_index < len(EXPRESSION_STEPS):
        _persist_progress(
            cal_index=cal_index,
            expr_index=expr_index,
            captured=captured,
            expr_armed=expr_armed,
            samples=samples,
            center_ears=center_ears,
            profile=profile,
        )
        cleanup()
        if cancelled:
            print("\nOnboarding paused — run setup again to resume.")
        else:
            print("\nOnboarding incomplete — run setup again to resume.")
        return None, None, None

    print("\nAll 9 onboarding steps complete.\n")
    return samples, profile, cap


def main() -> int:
    config = Config()
    if not ensure_privacy_consent():
        return 1

    print("=" * 56)
    print("  SYNAPSE ONBOARDING WIZARD")
    print("  One session: 5 calibration steps + 4 expressions (letter, SPACE)")
    print("=" * 56)
    print()

    samples, profile, cap = run_onboarding(config)
    if samples is None or profile is None or cap is None:
        return 1

    calibration = build_profile(samples)
    save_calibration(calibration, config.calibration_path)
    save_emotion_profile(profile, config.emotion_profile_path)
    clear_progress()

    return_home = wants_return_home()
    profile_name = get_active_user_display_name()
    completion_lines = [
        "Onboarding complete!",
        f"Saved for profile: {profile_name}",
        "All 9 steps finished in one session.",
        f"Calibration -> {config.calibration_path.name}",
        f"Emotion profile -> {config.emotion_profile_path.name}",
    ]
    if return_home:
        completion_lines.append("Press SPACE to return to the home menu.")
    else:
        completion_lines.append("Next: launch Monitor from the Synapse home screen.")

    try:
        run_hold_screen(cap, completion_lines)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Onboarding complete!")
    print(f"  calibration.json  -> {config.calibration_path.resolve()}")
    print(f"  emotion_profile.json -> {config.emotion_profile_path.resolve()}")

    if return_home:
        relaunch_home()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
