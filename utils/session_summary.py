from collections import Counter
from datetime import timedelta
from pathlib import Path

from src.cognition.cognitive_state import State
from utils.session_log import SessionRecord, latest_session, load_session


def summarize_session(records: list[SessionRecord]) -> dict:
    if not records:
        return {"record_count": 0}

    start = records[0].timestamp
    end = records[-1].timestamp
    duration = max((end - start).total_seconds(), 1.0)

    state_counts = Counter(record.state for record in records)
    gaze_counts = Counter(record.signals["gaze_direction"] for record in records)
    transitions = sum(
        1 for previous, current in zip(records, records[1:]) if previous.state != current.state
    )

    ears = [record.signals["ear"] for record in records]
    distractions = [record.distraction for record in records]
    autonomies = [record.autonomy for record in records]
    blink_rates = [record.signals["blink_rate"] for record in records]

    state_percentages = {
        state.value: round(count / len(records) * 100, 1)
        for state, count in state_counts.items()
    }

    return {
        "record_count": len(records),
        "start": start.isoformat(sep=" "),
        "end": end.isoformat(sep=" "),
        "duration_seconds": round(duration, 1),
        "duration_label": str(timedelta(seconds=int(duration))),
        "state_percentages": state_percentages,
        "dominant_state": state_counts.most_common(1)[0][0].value,
        "state_transitions": transitions,
        "avg_ear": round(sum(ears) / len(ears), 3),
        "min_ear": round(min(ears), 3),
        "max_ear": round(max(ears), 3),
        "final_blink_count": records[-1].signals["blink_counter"],
        "avg_blink_rate": round(sum(blink_rates) / len(blink_rates), 1),
        "avg_distraction": round(sum(distractions) / len(distractions), 1),
        "max_distraction": max(distractions),
        "avg_autonomy": round(sum(autonomies) / len(autonomies), 2),
        "min_autonomy": round(min(autonomies), 2),
        "max_autonomy": round(max(autonomies), 2),
        "top_gaze_directions": gaze_counts.most_common(3),
        "high_attention_pct": state_percentages.get(State.HIGH_ATTENTION.value, 0.0),
        "distracted_pct": state_percentages.get(State.DISTRACTED.value, 0.0),
        "fatigued_pct": state_percentages.get(State.FATIGUED.value, 0.0),
    }


def format_summary(summary: dict, session_name: str) -> str:
    if summary.get("record_count", 0) == 0:
        return f"No records found in {session_name}."

    lines = [
        "SYNAPSE SESSION SUMMARY",
        f"Session: {session_name}",
        f"Window: {summary['start']} -> {summary['end']}",
        f"Duration: {summary['duration_label']} ({summary['duration_seconds']}s)",
        f"Samples: {summary['record_count']}",
        "",
        "STATE BREAKDOWN",
        f"- Dominant: {summary['dominant_state']}",
        f"- Transitions: {summary['state_transitions']}",
    ]

    for state_name, percentage in summary["state_percentages"].items():
        lines.append(f"- {state_name}: {percentage}%")

    lines.extend(
        [
            "",
            "SIGNAL AVERAGES",
            f"- EAR: avg {summary['avg_ear']} | min {summary['min_ear']} | max {summary['max_ear']}",
            f"- Blinks: {summary['final_blink_count']} total | {summary['avg_blink_rate']}/min avg",
            f"- Distraction: avg {summary['avg_distraction']}% | peak {summary['max_distraction']}%",
            f"- Autonomy: avg {summary['avg_autonomy']} | range {summary['min_autonomy']}-{summary['max_autonomy']}",
            "",
            "TOP GAZE DIRECTIONS",
        ]
    )

    for gaze_direction, count in summary["top_gaze_directions"]:
        lines.append(f"- {gaze_direction}: {count} samples")

    lines.extend(
        [
            "",
            "TAKEAWAYS",
            f"- Focused time: {summary['high_attention_pct']}% high_attention",
            f"- Drift time: {summary['distracted_pct']}% distracted",
            f"- Fatigue time: {summary['fatigued_pct']}% fatigued",
        ]
    )
    return "\n".join(lines)


def write_summary_report(session_path: Path, output_path: Path | None = None) -> str:
    records = load_session(session_path)
    summary = summarize_session(records)
    report = format_summary(summary, session_path.name)
    output_path = output_path or session_path.with_suffix(".summary.txt")
    output_path.write_text(report, encoding="utf-8")
    return report


def summarize_latest_or_path(path: str | None = None) -> tuple[Path, str]:
    session_path = Path(path) if path else latest_session()
    if session_path is None or not session_path.exists():
        raise FileNotFoundError("No session CSV found in sessions/")
    report = write_summary_report(session_path)
    return session_path, report
