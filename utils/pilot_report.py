"""Sanitized pilot summaries for no-payment team trials."""

from __future__ import annotations

import csv
from pathlib import Path

from utils.app_paths import reports_dir, session_dir
from utils.fusion_summary import summarize_fusion_csv


def _recommendation(summary: dict) -> str:
    if summary.get("avg_fatigue", 0) > 0.55:
        return "Schedule shorter sessions or add recovery breaks."
    if summary.get("avg_distraction", 0) > 60:
        return "Review environment and reduce off-screen interruptions."
    if summary.get("avg_engagement", 0) < 0.45:
        return "Try a shorter focused block and compare the next session."
    return "No major focus/fatigue flag for this session."


def summarize_monitor_sessions(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        summary = summarize_fusion_csv(path)
        if summary.get("record_count", 0) == 0:
            continue
        rows.append(
            {
                "session": path.name,
                "duration": summary.get("duration_label", ""),
                "samples": str(summary.get("record_count", 0)),
                "dominant_state": str(summary.get("dominant_state", "unknown")),
                "avg_engagement_pct": f"{summary.get('avg_engagement', 0):.0%}",
                "avg_fatigue_pct": f"{summary.get('avg_fatigue', 0):.0%}",
                "avg_tension_pct": f"{summary.get('avg_tension', 0):.0%}",
                "avg_distraction_pct": f"{summary.get('avg_distraction', 0):.0f}%",
                "recommendation": _recommendation(summary),
            }
        )
    return rows


def write_pilot_summary(paths: list[Path] | None = None) -> Path:
    reports_dir().mkdir(parents=True, exist_ok=True)
    if paths is None:
        paths = sorted(session_dir().glob("monitor_*.csv"))
    rows = summarize_monitor_sessions(paths)
    target = reports_dir() / "synapse_pilot_summary.csv"
    fieldnames = [
        "session",
        "duration",
        "samples",
        "dominant_state",
        "avg_engagement_pct",
        "avg_fatigue_pct",
        "avg_tension_pct",
        "avg_distraction_pct",
        "recommendation",
    ]
    with target.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target
