"""Post-session debrief — timeline, alerts, and presence summary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2

from src.visualization.hud_text import HUD_ACCENT, HUD_LABEL, draw_hud_text
from src.visualization.timeline import draw_timeline_bar, load_alert_log, load_session_rows
from utils.fusion_summary import summarize_fusion_csv
from utils.manager_report import build_manager_report


def _presence_tail(path: Path, limit: int = 8) -> list[str]:
    if not path.exists():
        return []
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return lines[-limit:]


def run_session_debrief(
    session_csv: Path,
    *,
    alert_csv: Path | None = None,
    presence_log: Path | None = None,
    window_title: str = "Synapse - Session Debrief",
) -> None:
    rows = load_session_rows(session_csv) if session_csv.exists() else []
    alerts = load_alert_log(alert_csv) if alert_csv else []
    presence_lines = _presence_tail(presence_log) if presence_log else []
    summary = summarize_fusion_csv(session_csv) if session_csv.exists() else {}
    alert_flags = [f"{row.get('message', '')} at {row.get('elapsed_sec', '?')}s" for row in alerts]
    report_lines = build_manager_report(summary, session_csv.name, alert_flags=alert_flags).splitlines()[:14]

    index = max(0, len(rows) - 1)
    paused = False
    print("Debrief: SPACE pause | LEFT/RIGHT step | R replay | Q close")

    while True:
        frame = _blank_debrief_frame()
        y = 28
        draw_hud_text(frame, "SESSION DEBRIEF", (24, y), size=16, color=HUD_ACCENT, label=True)
        y += 28
        for line in report_lines[:6]:
            draw_hud_text(frame, line[:92], (24, y), size=11, color=HUD_LABEL)
            y += 16

        if presence_lines:
            draw_hud_text(frame, "PRESENCE", (24, y + 8), size=11, color=HUD_LABEL, label=True)
            y += 24
            for line in presence_lines[-4:]:
                draw_hud_text(frame, line[:92], (24, y), size=10, color=HUD_ACCENT)
                y += 14

        if rows:
            draw_timeline_bar(
                frame,
                rows,
                index,
                (24, frame.shape[0] - 110),
                (frame.shape[1] - 48, 72),
                mode="state",
                alert_rows=alerts,
            )

        draw_hud_text(
            frame,
            "Q close  |  R replay  |  arrows step",
            (24, frame.shape[0] - 18),
            size=10,
            color=HUD_LABEL,
        )
        cv2.imshow(window_title, frame)
        delay = 0 if paused else 40
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            _launch_replay(session_csv)
        elif key == 81 and index > 0:
            index -= 1
        elif key == 83 and index < len(rows) - 1:
            index += 1

    cv2.destroyWindow(window_title)


def _blank_debrief_frame(width: int = 960, height: int = 540):
    import numpy as np

    frame = np.full((height, width, 3), (12, 14, 18), dtype=np.uint8)
    return frame


def _launch_replay(session_csv: Path) -> None:
    replay_script = Path(__file__).resolve().parents[2] / "replay_monitor.py"
    if not replay_script.exists():
        return
    subprocess.Popen([sys.executable, str(replay_script), str(session_csv)])
