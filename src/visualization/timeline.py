import csv
from pathlib import Path

import cv2

from src.visualization.hud_text import HUD_ACCENT, HUD_DIM, HUD_LABEL, draw_hud_text

STATE_COLORS = {
    "high_attention": (62, 128, 52),
    "moderate": (168, 128, 58),
    "fatigued": (48, 158, 210),
    "distracted": (42, 52, 198),
}

PROFILE_COLORS = {
    "neutral": (170, 165, 158),
    "happy": (62, 128, 52),
    "sad": (48, 158, 210),
    "mad": (42, 52, 198),
}

ALERT_COLORS = {
    "low_engagement": (48, 158, 210),
    "high_distraction": (42, 52, 198),
    "fatigue_spike": (50, 140, 210),
    "high_tension": (100, 80, 170),
}


def load_session_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _segment_color(row: dict, mode: str) -> tuple[int, int, int]:
    if mode == "profile":
        return PROFILE_COLORS.get(row.get("profile_phase", ""), (190, 186, 176))
    if mode == "engagement":
        value = float(row.get("engagement", 0.0))
        return (
            int(70 + (1 - value) * 60),
            int(120 + value * 80),
            int(50 + value * 40),
        )
    return STATE_COLORS.get(row.get("state", ""), (190, 186, 176))


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
    draw_hud_text(frame, f"TL {mode}", (x + 8, y + 2), size=10, label=True)

    if not rows:
        return frame

    bar_top = y + 16
    bar_bottom = y + height - 10
    bar_left = x + 8
    bar_right = x + width - 8
    bar_width = bar_right - bar_left
    total = len(rows)

    cv2.line(frame, (bar_left, bar_top), (bar_right, bar_top), HUD_DIM, 1, cv2.LINE_AA)
    cv2.line(frame, (bar_left, bar_bottom), (bar_right, bar_bottom), HUD_DIM, 1, cv2.LINE_AA)
    for index, row in enumerate(rows):
        x1 = bar_left + int(index * bar_width / total)
        x2 = bar_left + int((index + 1) * bar_width / total)
        color = _segment_color(row, mode)
        cv2.rectangle(frame, (x1, bar_top + 1), (max(x2, x1 + 1), bar_bottom - 1), color, -1)

    cursor_x = bar_left + int(current_index * bar_width / max(total - 1, 1))
    cv2.line(frame, (cursor_x, bar_top - 2), (cursor_x, bar_bottom + 2), HUD_ACCENT, 2, cv2.LINE_AA)

    if alert_rows:
        duration = float(rows[-1]["elapsed_sec"]) - float(rows[0]["elapsed_sec"])
        duration = max(duration, 1.0)
        for alert in alert_rows:
            rule = alert.get("rule_id", "")
            color = ALERT_COLORS.get(rule, HUD_ACCENT)
            at = float(alert.get("elapsed_sec", 0.0)) - float(rows[0]["elapsed_sec"])
            mark_x = bar_left + int((at / duration) * bar_width)
            cv2.line(frame, (mark_x, bar_top - 4), (mark_x, bar_bottom + 4), color, 2, cv2.LINE_AA)

    progress = (current_index + 1) / total
    py = y + height - 4
    cv2.line(frame, (bar_left, py), (bar_right, py), HUD_DIM, 1, cv2.LINE_AA)
    cv2.line(
        frame,
        (bar_left, py),
        (bar_left + int((bar_right - bar_left) * progress), py),
        HUD_ACCENT,
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_live_session_strip(
    frame,
    rows: list[dict],
    origin: tuple[int, int],
    size: tuple[int, int],
    mode: str = "profile",
):
    if len(rows) < 2:
        return frame
    x, y = origin
    width, height = size
    draw_hud_text(frame, "LIVE", (x, y), size=9, color=HUD_LABEL, label=True)
    bar_top = y + 12
    bar_bottom = y + height - 2
    bar_left = x + 40
    bar_right = x + width
    bar_width = max(bar_right - bar_left, 1)
    total = len(rows)
    for index, row in enumerate(rows):
        x1 = bar_left + int(index * bar_width / total)
        x2 = bar_left + int((index + 1) * bar_width / total)
        color = _segment_color(row, mode)
        cv2.rectangle(frame, (x1, bar_top), (max(x2, x1 + 1), bar_bottom), color, -1)
    cursor_x = bar_right - 1
    cv2.line(frame, (cursor_x, bar_top - 1), (cursor_x, bar_bottom + 1), HUD_ACCENT, 2, cv2.LINE_AA)
    return frame


def load_alert_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))
