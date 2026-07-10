import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from time import monotonic

import cv2
import mediapipe as mp
import numpy as np

from src.cognition.emotion_state import Emotion
from src.perception.emotion_estimator import EmotionEstimator
from src.perception.state_estimator import StateEstimator

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

SESSION_DIR = Path("sessions")

EMOTION_COLORS = {
    Emotion.NEUTRAL: (200, 200, 200),
    Emotion.HAPPY: (0, 220, 0),
    Emotion.SAD: (255, 120, 0),
    Emotion.SURPRISED: (0, 220, 255),
    Emotion.STRESSED: (0, 0, 255),
}

PHASE_COLORS = {
    "": (90, 90, 90),
    "neutral": (200, 200, 200),
    "happy": (0, 220, 0),
    "sad": (255, 120, 0),
    "mad": (0, 0, 255),
}

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

BROW_POINT_COLORS = ((0, 220, 0), (0, 220, 255), (255, 0, 220))
BROW_POINT_LABELS = ("O", "M", "I")

MESH_STYLE = mp_drawing.DrawingSpec(color=(70, 70, 70), thickness=1, circle_radius=1)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(110, 110, 110), thickness=1, circle_radius=1)


def landmark_xy(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def emotion_highlight_indices() -> set[int]:
    indices: set[int] = set()
    for region_indices in EmotionEstimator.tracking_regions().values():
        indices.update(region_indices)
    return indices


def draw_full_face_mesh(frame, face_landmarks):
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=MESH_STYLE,
        connection_drawing_spec=MESH_STYLE,
    )
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=CONTOUR_STYLE,
    )
    return frame


def draw_region_points(frame, landmarks, indices, color, radius=4, closed=False):
    height, width = frame.shape[:2]
    points = [landmark_xy(landmarks[index], width, height) for index in indices]
    for point in points:
        cv2.circle(frame, point, radius, color, -1)
        cv2.circle(frame, point, radius + 2, color, 1)
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], closed, color, 2)
    return frame


def draw_eye_ring(frame, landmarks, ring_indices, color):
    height, width = frame.shape[:2]
    points = np.array(
        [landmark_xy(landmarks[index], width, height) for index in ring_indices],
        dtype=np.float32,
    )
    if len(points) < 5:
        return frame

    cv2.polylines(frame, [points.astype(np.int32)], True, color, 2)
    for point in points.astype(np.int32):
        cv2.circle(frame, tuple(point), 3, color, -1)

    ellipse = cv2.fitEllipse(points)
    cv2.ellipse(frame, ellipse, color, 2)
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    cv2.circle(frame, center, 2, (0, 0, 255), -1)
    return frame


def draw_expression_overlay(frame, landmarks):
    regions = EmotionEstimator.tracking_regions()
    for region_name, indices in regions.items():
        if region_name.endswith("_eye"):
            continue
        color = EmotionEstimator.REGION_COLORS[region_name]
        frame = draw_region_points(frame, landmarks, indices, color, radius=4, closed=False)

    frame = draw_eye_ring(frame, landmarks, regions["left_eye"], EmotionEstimator.REGION_COLORS["left_eye"])
    frame = draw_eye_ring(frame, landmarks, regions["right_eye"], EmotionEstimator.REGION_COLORS["right_eye"])

    height, width = frame.shape[:2]
    for brow_name, side in (("left_brow", "L"), ("right_brow", "R")):
        brow_indices = regions[brow_name]
        for index, point_color, label in zip(
            brow_indices, BROW_POINT_COLORS, BROW_POINT_LABELS
        ):
            point = landmark_xy(landmarks[index], width, height)
            cv2.circle(frame, point, 7, point_color, -1)
            cv2.putText(
                frame,
                f"{side}{label}",
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                point_color,
                1,
            )

    return frame


def draw_meter(frame, x, y, width, value, label, color, max_value=0.05):
    cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.rectangle(frame, (x, y), (x + width, y + 14), (60, 60, 60), -1)
    fill = int(min(max(value / max_value, 0.0), 1.0) * width)
    cv2.rectangle(frame, (x, y), (x + fill, y + 14), color, -1)


def region_center(landmarks, indices):
    xs = [landmarks[index].x for index in indices]
    ys = [landmarks[index].y for index in indices]
    return float(np.mean(xs)), float(np.mean(ys))


def print_summary(
    log_path: Path,
    emotion_counts: Counter[str],
    phase_counts: Counter[str],
    phase_refs: dict[str, dict[str, float]],
    elapsed: float,
) -> None:
    total = sum(emotion_counts.values()) or 1
    print(f"\nSession saved to {log_path}")
    refs_path = log_path.with_suffix(".refs.json")
    refs_path.write_text(json.dumps(phase_refs, indent=2), encoding="utf-8")
    print(f"Phase references saved to {refs_path}")
    print(f"Duration: {elapsed:.0f}s | Frames logged: {total}")
    print("Labeled phases (key presses):")
    for phase, count in phase_counts.most_common():
        if phase:
            print(f"  {phase}: {count} frames")
    print("Auto-detected emotions:")
    for emotion, count in emotion_counts.most_common():
        print(f"  {emotion}: {count / total:.0%} ({count} frames)")


def main() -> None:
    SESSION_DIR.mkdir(exist_ok=True)
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = SESSION_DIR / f"emotion_test_{session_name}.csv"

    emotion_estimator = EmotionEstimator(calibration_frames=0)
    state_estimator = StateEstimator()
    cap = cv2.VideoCapture(0)
    emotion_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    phase_refs: dict[str, dict[str, float]] = {}
    labeled_phase = ""
    last_signals: dict | None = None
    started_at = monotonic()

    print("Starting emotion test... Press 'q' to quit.")
    print(f"Logging to {log_path}")
    print("Full 478-point mesh + highlighted emotion regions.")
    print("Press keys while holding each expression:")
    print("  N = capture neutral baseline")
    print("  H = label happy")
    print("  S = label sad")
    print("  S = sad/stressed   M = mad")

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "timestamp",
                "elapsed_sec",
                "landmark_count",
                "labeled_phase",
                "emotion",
                "emotion_confidence",
                "calibrating",
                "smile_score",
                "smile_delta",
                "cheek_raise",
                "cheek_delta",
                "brow_raise",
                "brow_furrow",
                "brow_inner_pinch",
                "mouth_open",
                "ear",
                "head_yaw",
                "head_pitch",
                "left_brow_x",
                "left_brow_y",
                "right_brow_x",
                "right_brow_y",
                "left_eye_x",
                "left_eye_y",
                "right_eye_x",
                "right_eye_y",
                "left_cheek_x",
                "left_cheek_y",
                "right_cheek_x",
                "right_cheek_y",
                "mouth_x",
                "mouth_y",
            ]
        )

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark
                frame = draw_full_face_mesh(frame, face_landmarks)
                frame = draw_expression_overlay(frame, landmarks)

                cognitive = state_estimator.update(landmarks)
                emotion = emotion_estimator.update(landmarks, ear=cognitive.signals.get("ear"))
                cog = cognitive.signals
                sig = emotion.signals
                last_signals = sig
                regions = EmotionEstimator.tracking_regions()

                left_brow = region_center(landmarks, regions["left_brow"])
                right_brow = region_center(landmarks, regions["right_brow"])
                left_eye = region_center(landmarks, regions["left_eye"])
                right_eye = region_center(landmarks, regions["right_eye"])
                left_cheek = region_center(landmarks, regions["left_cheek"])
                right_cheek = region_center(landmarks, regions["right_cheek"])
                mouth = region_center(landmarks, regions["mouth"])

                elapsed = monotonic() - started_at
                if not sig.get("calibrating"):
                    emotion_counts[emotion.emotion.value] += 1
                if labeled_phase:
                    phase_counts[labeled_phase] += 1

                writer.writerow(
                    [
                        datetime.now().isoformat(timespec="seconds"),
                        f"{elapsed:.2f}",
                        len(landmarks),
                        labeled_phase,
                        emotion.emotion.value,
                        f"{emotion.confidence:.2f}",
                        int(bool(sig.get("calibrating"))),
                        f"{sig['smile_score']:.4f}",
                        f"{sig.get('smile_delta', 0.0):.4f}",
                        f"{sig['cheek_raise']:.4f}",
                        f"{sig.get('cheek_delta', 0.0):.4f}",
                        f"{sig['brow_raise']:.4f}",
                        f"{sig['brow_furrow']:.4f}",
                        f"{sig['brow_inner_pinch']:.4f}",
                        f"{sig['mouth_open']:.4f}",
                        f"{cog['ear']:.4f}",
                        f"{cog['head_yaw']:+.2f}",
                        f"{cog['head_pitch']:+.2f}",
                        f"{left_brow[0]:.4f}",
                        f"{left_brow[1]:.4f}",
                        f"{right_brow[0]:.4f}",
                        f"{right_brow[1]:.4f}",
                        f"{left_eye[0]:.4f}",
                        f"{left_eye[1]:.4f}",
                        f"{right_eye[0]:.4f}",
                        f"{right_eye[1]:.4f}",
                        f"{left_cheek[0]:.4f}",
                        f"{left_cheek[1]:.4f}",
                        f"{right_cheek[0]:.4f}",
                        f"{right_cheek[1]:.4f}",
                        f"{mouth[0]:.4f}",
                        f"{mouth[1]:.4f}",
                    ]
                )
                log_file.flush()

                border_color = PHASE_COLORS.get(labeled_phase, PHASE_COLORS[""])
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), border_color, 8)
                cv2.putText(
                    frame,
                    f"AUTO: {emotion.emotion.value.upper()} ({emotion.confidence:.0%})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    EMOTION_COLORS[emotion.emotion],
                    2,
                )
                phase_text = labeled_phase.upper() if labeled_phase else "PRESS N/H/S/M"
                cv2.putText(
                    frame,
                    f"LABEL: {phase_text}",
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    border_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"Mesh: {len(landmarks)} pts | Log: {log_path.name}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                if sig.get("calibrating"):
                    cv2.putText(
                        frame,
                        "Hold neutral face, press N to set baseline",
                        (20, 128),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 220, 255),
                        2,
                    )

                meter_y = 150
                draw_meter(frame, 20, meter_y, 180, sig.get("smile_delta", 0.0), "Smile", (0, 220, 0))
                draw_meter(
                    frame,
                    20,
                    meter_y + 30,
                    180,
                    max(0.0, -sig.get("smile_delta", 0.0)),
                    "Frown",
                    (255, 120, 0),
                )
                draw_meter(
                    frame,
                    20,
                    meter_y + 60,
                    180,
                    sig.get("cheek_delta", 0.0),
                    "Cheek raise",
                    (180, 0, 255),
                    0.03,
                )
                draw_meter(frame, 20, meter_y + 90, 180, sig["brow_raise"], "Brow raise", (0, 220, 255))
                draw_meter(frame, 20, meter_y + 120, 180, sig["mouth_open"], "Mouth open", (255, 180, 0), 0.6)
                draw_meter(frame, 20, meter_y + 150, 180, sig["brow_furrow"], "Brow furrow", (0, 0, 255))
                draw_meter(
                    frame,
                    20,
                    meter_y + 180,
                    180,
                    0.5 - sig["brow_inner_pinch"],
                    "Brow pinch",
                    (180, 0, 255),
                    0.2,
                )
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            key = cv2.waitKey(1) & 0xFF
            if key in PHASE_KEYS and last_signals is not None:
                labeled_phase = PHASE_KEYS[key]
                snapshot = EmotionEstimator.snapshot_signals(last_signals)
                phase_refs[labeled_phase] = snapshot
                if labeled_phase == "neutral":
                    emotion_estimator.set_neutral_baseline(
                        last_signals["smile_score"],
                        last_signals["cheek_raise"],
                    )
                    last_signals = {
                        **last_signals,
                        "smile_delta": 0.0,
                        "cheek_delta": 0.0,
                        "calibrating": False,
                    }
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Captured {labeled_phase}: {snapshot}")
            elif key in PHASE_KEYS:
                print("Face not detected - cannot capture phase yet.")
            if key == ord("q"):
                break

            cv2.imshow("Synapse Emotion Test", frame)

    cap.release()
    cv2.destroyAllWindows()
    print_summary(log_path, emotion_counts, phase_counts, phase_refs, monotonic() - started_at)


if __name__ == "__main__":
    main()
