"""Capture phone + bare hand so we can measure the real difference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.perception.capture import CameraCapture
from src.perception.presence_detector import (
    PresenceBox,
    _looks_like_phone,
    _qualifies_as_phone,
    normalize_object_label,
)
from utils.presence_model import object_model_bytes


def _box_metrics(box: PresenceBox) -> dict:
    aspect = box.width / max(box.height, 1e-6)
    return {
        "label": box.label,
        "conf": round(box.confidence, 3),
        "cx": round(box.center[0], 3),
        "cy": round(box.center[1], 3),
        "w": round(box.width, 3),
        "h": round(box.height, 3),
        "area": round(box.area, 4),
        "aspect": round(aspect, 3),
    }


def _raw_objects(detector, rgb_frame) -> list[dict]:
    """Bypass refine — dump every COCO hit the model returns."""
    from mediapipe.tasks.python import vision

    # Use the detector already on PresenceDetector.
    objects = detector._detect_objects(rgb_frame)
    return [_box_metrics(obj) for obj in objects]


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture phone vs bare hand for calibration")
    parser.add_argument("--seconds", type=float, default=30.0, help="Recording length (default 30)")
    parser.add_argument(
        "--mode",
        choices=("combo", "phone"),
        default="combo",
        help="combo=phone+bare hand; phone=phone visible, hands out of frame",
    )
    args = parser.parse_args()
    duration = max(5.0, args.seconds)
    phone_only = args.mode == "phone"

    out_dir = Path.home() / "AppData" / "Local" / "Synapse" / "hand_phone_capture"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):
        old.unlink()

    print("Opening camera...")
    camera = CameraCapture(detect_presence=True)
    detector = camera._presence_detector
    assert detector is not None

    if phone_only:
        print("Get ready: show your PHONE clearly — keep hands out of the way as best you can.")
    else:
        print("Get ready: PHONE in one hand, BARE HAND in the other.")
    print("Countdown...")
    for sec in (3, 2, 1):
        print(f"  {sec}...")
        time.sleep(1.0)
    if phone_only:
        print(f"RECORDING {int(duration)} seconds — phone visible, hands clear of the device.")
    else:
        print(f"RECORDING {int(duration)} seconds — move both hands a bit.")

    started = time.monotonic()
    frame_i = 0
    samples: list[dict] = []
    preview_frames: list[np.ndarray] = []

    while time.monotonic() - started < duration:
        frame, landmarks, presence = camera.get_frame_landmarks_presence()
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = detector._detect_hands(rgb)
        raw = detector._detect_objects(rgb)

        hand_metrics = [_box_metrics(h) for h in hands]
        raw_metrics = [_box_metrics(o) for o in raw]
        presence_labels = sorted(presence.active_labels()) if presence else []
        presence_objs = (
            [_box_metrics(o) for o in presence.objects]
            if presence is not None
            else []
        )

        # Per-hand: what raw detections overlap each hand?
        hand_hits: list[dict] = []
        for hi, hand in enumerate(hands):
            overlaps = []
            for obj in raw:
                if not (
                    not (obj.x_max < hand.x_min or hand.x_max < obj.x_min
                         or obj.y_max < hand.y_min or hand.y_max < obj.y_min)
                ):
                    continue
                ratio = obj.area / max(hand.area, 1e-6)
                overlaps.append({
                    **_box_metrics(obj),
                    "area_ratio": round(ratio, 3),
                    "looks_phone": _looks_like_phone(obj),
                    "qualifies_phone": _qualifies_as_phone(obj, [hand]),
                })
            hand_hits.append({"hand_index": hi, "hand": hand_metrics[hi], "overlaps": overlaps})

        sample = {
            "t": round(time.monotonic() - started, 2),
            "hands": hand_metrics,
            "raw_objects": raw_metrics,
            "presence_labels": presence_labels,
            "presence_objects": presence_objs,
            "hand_hits": hand_hits,
        }
        samples.append(sample)

        # Draw overlay for review
        vis = frame.copy()
        h, w = vis.shape[:2]
        for hand in hands:
            cv2.rectangle(
                vis,
                (int(hand.x_min * w), int(hand.y_min * h)),
                (int(hand.x_max * w), int(hand.y_max * h)),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                "HAND",
                (int(hand.x_min * w), max(20, int(hand.y_min * h) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        for obj in raw:
            color = (0, 180, 255) if obj.label == "phone" else (180, 180, 180)
            cv2.rectangle(
                vis,
                (int(obj.x_min * w), int(obj.y_min * h)),
                (int(obj.x_max * w), int(obj.y_max * h)),
                color,
                2,
            )
            cv2.putText(
                vis,
                f"{obj.label} {obj.confidence:.2f}",
                (int(obj.x_min * w), max(20, int(obj.y_min * h) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        elapsed = time.monotonic() - started
        prompt = (
            f"RECORDING {elapsed:.1f}s / {duration:.0f}s  |  phone only, hands clear"
            if phone_only
            else f"RECORDING {elapsed:.1f}s / {duration:.0f}s  |  phone + bare hand"
        )
        cv2.putText(
            vis,
            prompt,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Synapse hand vs phone capture", vis)
        if frame_i % 5 == 0:
            preview_frames.append(vis.copy())
            cv2.imwrite(str(out_dir / f"frame_{frame_i:04d}.jpg"), vis)
        frame_i += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    log_path = out_dir / "samples.json"
    log_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")

    # Summary: how often bare-hand vs phone-hand get labeled
    label_counts: dict[str, int] = {}
    phone_on_hand = 0
    hand_as_phone = 0
    frames_with_2_hands = 0
    area_ratios: list[float] = []
    for s in samples:
        for label in s["presence_labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1
        if len(s["hands"]) >= 2:
            frames_with_2_hands += 1
        for hit in s["hand_hits"]:
            for ov in hit["overlaps"]:
                if ov["label"] == "phone":
                    phone_on_hand += 1
                    area_ratios.append(ov["area_ratio"])
                    if ov["area_ratio"] > 0.55:
                        hand_as_phone += 1

    summary = {
        "mode": args.mode,
        "frames": len(samples),
        "frames_with_2_hands": frames_with_2_hands,
        "presence_label_counts": label_counts,
        "phone_overlaps_on_hands": phone_on_hand,
        "hand_sized_phone_hits": hand_as_phone,
        "phone_area_ratio_stats": {
            "n": len(area_ratios),
            "mean": round(float(np.mean(area_ratios)), 3) if area_ratios else None,
            "min": round(float(np.min(area_ratios)), 3) if area_ratios else None,
            "max": round(float(np.max(area_ratios)), 3) if area_ratios else None,
            "p25": round(float(np.percentile(area_ratios, 25)), 3) if area_ratios else None,
            "p75": round(float(np.percentile(area_ratios, 75)), 3) if area_ratios else None,
        },
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== CAPTURE DONE ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
