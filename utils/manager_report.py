from pathlib import Path
import shutil
from datetime import datetime

from utils.app_paths import reports_dir
from utils.fusion_summary import summarize_fusion_csv


def build_manager_report(summary: dict, session_name: str, alert_flags: list[str] | None = None) -> str:
    if summary.get("record_count", 0) == 0:
        return f"No data in {session_name}."

    flags = []
    if summary.get("avg_engagement", 0) < 0.45:
        flags.append("Low focus signal - frequent gaze-away or reduced attention indicators.")
    if summary.get("avg_fatigue", 0) > 0.55:
        flags.append("Elevated fatigue signal - blink and eye openness patterns suggest a break may help.")
    if summary.get("avg_tension", 0) > 0.40:
        flags.append("Elevated strain signal - facial tension patterns were higher than baseline.")
    if summary.get("avg_distraction", 0) > 60:
        flags.append("High distraction signal - frequent off-screen gaze or head turns.")
    if summary.get("avg_positivity", 0) > 0.65:
        flags.append("Positive expression pattern - optional profile matching skewed toward the happy baseline.")
    if alert_flags:
        flags.extend(alert_flags)
    if not flags:
        flags.append("Baseline session — no significant flags.")

    dominant = summary.get("dominant_state", "unknown")
    profile_breakdown = summary.get("profile_phase_breakdown", {})

    lines = [
        "SYNAPSE FOCUS SESSION REPORT",
        f"Session: {session_name}",
        f"Duration: {summary.get('duration_label', '?')} ({summary.get('duration_seconds', 0)}s)",
        f"Samples: {summary.get('record_count', 0)}",
        "",
        "SESSION SUMMARY",
        f"- Dominant attention state: {dominant}",
        f"- Engagement: {summary.get('avg_engagement', 0):.0%}",
        f"- Fatigue: {summary.get('avg_fatigue', 0):.0%}",
        f"- Tension: {summary.get('avg_tension', 0):.0%}",
        f"- Positivity: {summary.get('avg_positivity', 0):.0%}",
        f"- Distraction: {summary.get('avg_distraction', 0):.0f}%",
        "",
        "OPTIONAL EXPRESSION-PATTERN MATCH",
    ]
    lines.append("- These are personal baseline pattern matches, not emotional diagnoses.")
    for phase, count in sorted(profile_breakdown.items(), key=lambda item: -item[1]):
        label = {
            "neutral": "neutral baseline",
            "happy": "happy baseline",
            "sad": "sad/stressed baseline",
            "mad": "frustration baseline",
        }.get(phase, phase)
        pct = count / summary["record_count"] * 100
        lines.append(f"- {label}: {pct:.0f}% ({count} frames)")

    lines.extend(
        [
            "",
            "INTERPRETATION LIMITS",
            "- Webcam signals are support signals for focus and fatigue awareness.",
            "- Synapse does not save raw video frames.",
            "- Do not use this report as a medical, disciplinary, or emotion-detection record.",
            "",
            "FOCUS FLAGS",
        ]
    )
    lines.extend(f"- {flag}" for flag in flags)
    return "\n".join(lines)


def export_report_to_desktop(report_text: str, session_name: str) -> Path | None:
    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = desktop / f"Synapse_Report_{stamp}.txt"
    header = f"Source session: {session_name}\n\n"
    target.write_text(header + report_text + "\n", encoding="utf-8")
    return target


def write_manager_report(
    csv_path: Path,
    alert_flags: list[str] | None = None,
    export_desktop: bool = True,
) -> str:
    summary = summarize_fusion_csv(csv_path)
    report = build_manager_report(summary, csv_path.name, alert_flags=alert_flags)
    report_path = csv_path.with_suffix(".report.txt")
    report_path.write_text(report + "\n", encoding="utf-8")
    reports_dir().mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, reports_dir() / report_path.name)

    if export_desktop:
        desktop_path = export_report_to_desktop(report, csv_path.name)
        if desktop_path:
            shutil.copy2(report_path, desktop_path)
    return report
