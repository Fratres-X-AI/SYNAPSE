import csv
from collections import Counter
from datetime import timedelta
from pathlib import Path


def summarize_fusion_csv(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        return {"record_count": 0}

    duration = float(rows[-1]["elapsed_sec"]) - float(rows[0]["elapsed_sec"])
    duration = max(duration, 1.0)

    states = Counter(row["state"] for row in rows)
    phases = (
        Counter(row["labeled_phase"] for row in rows if row.get("labeled_phase"))
        if "labeled_phase" in rows[0]
        else Counter()
    )
    emotions = (
        Counter(row["emotion"] for row in rows)
        if "emotion" in rows[0]
        else Counter()
    )
    profile_phases = Counter(row.get("profile_phase") for row in rows if row.get("profile_phase"))

    labeled_rows = [row for row in rows if row.get("labeled_phase")]
    profile_hits = sum(
        1 for row in labeled_rows if row.get("labeled_phase") == row.get("profile_phase")
    )
    profile_accuracy = round(profile_hits / len(labeled_rows) * 100, 1) if labeled_rows else 0.0

    def avg(key: str) -> float:
        return round(sum(float(row[key]) for row in rows) / len(rows), 3)

    return {
        "record_count": len(rows),
        "duration_seconds": round(duration, 1),
        "duration_label": str(timedelta(seconds=int(duration))),
        "dominant_state": states.most_common(1)[0][0],
        "state_breakdown": dict(states),
        "emotion_breakdown": dict(emotions),
        "labeled_phases": dict(phases),
        "profile_phase_breakdown": dict(profile_phases),
        "profile_label_accuracy_pct": profile_accuracy,
        "avg_engagement": avg("engagement"),
        "avg_fatigue": avg("fatigue"),
        "avg_tension": avg("tension"),
        "avg_positivity": avg("positivity"),
        "avg_distraction": avg("distraction"),
    }


def format_fusion_summary(summary: dict, session_name: str) -> str:
    if summary.get("record_count", 0) == 0:
        return f"No records found in {session_name}."

    lines = [
        "SYNAPSE FUSION SESSION SUMMARY",
        f"Session: {session_name}",
        f"Duration: {summary['duration_label']} ({summary['duration_seconds']}s)",
        f"Samples: {summary['record_count']}",
        "",
        "RELIABLE LAYER",
        f"- Dominant state: {summary['dominant_state']}",
    ]
    for state, count in summary["state_breakdown"].items():
        lines.append(f"- {state}: {count} frames")

    lines.extend(
        [
            "",
            "SOFT SCORES (avg)",
            f"- Engagement: {summary['avg_engagement']:.0%}",
            f"- Fatigue: {summary['avg_fatigue']:.0%}",
            f"- Tension: {summary['avg_tension']:.0%}",
            f"- Positivity: {summary['avg_positivity']:.0%}",
            f"- Distraction: {summary['avg_distraction']:.0f}%",
            "",
            "LABELED PHASES",
        ]
    )
    if summary["labeled_phases"]:
        for phase, count in summary["labeled_phases"].items():
            lines.append(f"- {phase}: {count} frames")
    else:
        lines.append("- none")

    if summary.get("profile_label_accuracy_pct"):
        lines.extend(
            [
                "",
                "PROFILE MATCHING",
                f"- Label accuracy: {summary['profile_label_accuracy_pct']}%",
            ]
        )
        for phase, count in summary.get("profile_phase_breakdown", {}).items():
            lines.append(f"- matched {phase}: {count} frames")

    lines.extend(["", "AUTO EMOTIONS"])
    for emotion, count in summary["emotion_breakdown"].items():
        lines.append(f"- {emotion}: {count} frames")

    return "\n".join(lines)


def write_fusion_summary(path: Path) -> str:
    summary = summarize_fusion_csv(path)
    report = format_fusion_summary(summary, path.name)
    path.with_suffix(".summary.txt").write_text(report + "\n", encoding="utf-8")
    return report
