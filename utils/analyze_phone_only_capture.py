"""Analyze phone-only capture."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path


def main() -> None:
    p = Path.home() / "AppData/Local/Synapse/hand_phone_capture/samples.json"
    samples = json.loads(p.read_text(encoding="utf-8"))

    raw_labels: Counter[str] = Counter()
    raw_phone: list[dict] = []
    presence_counts: Counter[str] = Counter()

    for s in samples:
        for label in s["presence_labels"]:
            presence_counts[label] += 1
        for obj in s["raw_objects"]:
            raw_labels[obj["label"]] += 1
            if obj["label"] == "phone":
                raw_phone.append(obj)

    print("presence labels:", dict(presence_counts))
    print("raw COCO labels:", dict(raw_labels))
    print(f"raw phone hits: {len(raw_phone)}")
    if raw_phone:
        areas = [o["area"] for o in raw_phone]
        confs = [o["conf"] for o in raw_phone]
        aspects = [o["aspect"] for o in raw_phone]
        print(
            f"  area mean={statistics.mean(areas):.4f} "
            f"conf mean={statistics.mean(confs):.3f} "
            f"aspect mean={statistics.mean(aspects):.3f}"
        )
        print("  sample:", raw_phone[:3])

    # frames with any raw phone
    phone_frames = [s for s in samples if any(o["label"] == "phone" for o in s["raw_objects"])]
    print(f"frames with raw phone: {len(phone_frames)}/{len(samples)}")
    if phone_frames:
        s = phone_frames[0]
        print("example frame raw:", s["raw_objects"])
        print("example presence:", s["presence_labels"], s["presence_objects"])


if __name__ == "__main__":
    main()
