import csv
from pathlib import Path

import cv2
import numpy as np

STATE_COLORS = {
    "high_attention": (80, 220, 100),
    "moderate": (0, 220, 255),
    "fatigued": (0, 140, 255),
    "distracted": (60, 60, 255),
}

PROFILE_COLORS = {
    "neutral": (200, 200, 200),
    "happy": (0, 220, 0),
    "sad": (255, 120, 0),
    "mad": (0, 0, 255),
}

ALERT_COLORS = {
    "low_engagement": (0, 120, 255),
    "high_distraction": (0, 0, 255),
    "fatigue_spike": (0, 180, 255),
    "high_tension": (180, 0, 255),
}


def load_session_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _segment_color(row: dict, mode: str) -> tuple[int, int, int]:
    if mode == "profile":
        return PROFILE_COLORS.get(row.get("profile_phase", ""), (90, 90, 90))
    if mode == "engagement":
        value = float(row.get("engagement", 0.0))
        return (0, int(180 + value * 75), int(80 + (1 - value) * 80))
    return STATE_COLORS.get(row.get("state", ""), (90, 90, 90))


def draw_timeline_bar(
    frame,
    rows: list[dict],
    current_index: int,
    origin: tuple[int, int],
    size: tuple[int, int],
    mode: str = "state",
    alert_rows: list[dict] | None = None,
):
    x, y = origin
    width, height = size
    panel = frame[y : y + height, x : x + width].copy()
    panel[:] = (15, 15, 15)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (60, 60, 60), 1)
    cv2.putText(panel, f"TIMELINE ({mode})", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    if not rows:
        frame[y : y + height, x : x + width] = panel
        return frame

    bar_top = 24
    bar_bottom = height - 18
    bar_left = 8
    bar_right = width - 8
    bar_width = bar_right - bar_left
    total = len(rows)

    for index, row in enumerate(rows):
        x1 = bar_left + int(index * bar_width / total)
        x2 = bar_left + int((index + 1) * bar_width / total)
        color = _segment_color(row, mode)
        cv2.rectangle(panel, (x1, bar_top), (max(x2, x1 + 1), bar_bottom), color, -1)

    cursor_x = bar_left + int(current_index * bar_width / max(total - 1, 1))
    cv2.line(panel, (cursor_x, bar_top - 2), (cursor_x, bar_bottom + 2), (255, 255, 255), 2)

    if alert_rows:
        duration = float(rows[-1]["elapsed_sec"]) - float(rows[0]["elapsed_sec"])
        duration = max(duration, 1.0)
        for alert in alert_rows:
            rule = alert.get("rule_id", "")
            color = ALERT_COLORS.get(rule, (255, 255, 255))
            at = float(alert.get("elapsed_sec", 0.0)) - float(rows[0]["elapsed_sec"])
            mark_x = bar_left + int((at / duration) * bar_width)
            cv2.line(panel, (mark_x, bar_top - 4), (mark_x, bar_bottom + 4), color, 2)

    progress = (current_index + 1) / total
    cv2.rectangle(panel, (bar_left, height - 12), (bar_right, height - 4), (40, 40, 40), -1)
    cv2.rectangle(
        panel,
        (bar_left, height - 12),
        (bar_left + int((bar_right - bar_left) * progress), height - 4),
        (0, 200, 255),
        -1,
    )
    frame[y : y + height, x : x + width] = panel
    return frame


def load_alert_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))
