"""Analyze hand_phone_capture samples.json."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def main() -> None:
    p = Path.home() / "AppData/Local/Synapse/hand_phone_capture/samples.json"
    samples = json.loads(p.read_text(encoding="utf-8"))

    two_hand = [s for s in samples if len(s["hands"]) >= 2]
    print(f"frames={len(samples)} two_hands={len(two_hand)}")

    # Per-hand-index: collect phone overlap stats when 2 hands visible
    by_hand: dict[int, list[float]] = {0: [], 1: []}
    by_hand_conf: dict[int, list[float]] = {0: [], 1: []}
    by_hand_aspect: dict[int, list[float]] = {0: [], 1: []}

    for s in two_hand:
        for hit in s["hand_hits"]:
            idx = hit["hand_index"]
            if idx > 1:
                continue
            for ov in hit["overlaps"]:
                if ov["label"] != "phone":
                    continue
                by_hand[idx].append(ov["area_ratio"])
                by_hand_conf[idx].append(ov["conf"])
                by_hand_aspect[idx].append(ov["aspect"])

    for idx in (0, 1):
        ratios = by_hand[idx]
        if not ratios:
            print(f"hand[{idx}]: no phone overlaps")
            continue
        print(
            f"hand[{idx}] phone overlaps n={len(ratios)} "
            f"ratio mean={statistics.mean(ratios):.3f} "
            f"median={statistics.median(ratios):.3f} "
            f"min={min(ratios):.3f} max={max(ratios):.3f}"
        )
        print(
            f"  conf mean={statistics.mean(by_hand_conf[idx]):.3f} "
            f"aspect mean={statistics.mean(by_hand_aspect[idx]):.3f}"
        )

    # Guess: lower median area_ratio hand = phone hand (device smaller than palm)
    if by_hand[0] and by_hand[1]:
        m0, m1 = statistics.median(by_hand[0]), statistics.median(by_hand[1])
        phone_idx = 0 if m0 < m1 else 1
        bare_idx = 1 - phone_idx
        print(f"\nLikely PHONE hand = index {phone_idx} (lower area_ratio)")
        print(f"Likely BARE hand = index {bare_idx}")
        print(f"  phone hand median ratio: {min(m0,m1):.3f}")
        print(f"  bare hand median ratio: {max(m0,m1):.3f}")
        print(f"  suggested cutoff area_ratio: {(min(m0,m1)+max(m0,m1))/2:.3f}")

    # False positive presence labels
    pen_frames = sum(1 for s in samples if "pen" in s["presence_labels"])
    phone_frames = sum(1 for s in samples if "phone" in s["presence_labels"])
    print(f"\npresence: phone in {phone_frames} frames, pen in {pen_frames} frames")


if __name__ == "__main__":
    main()
