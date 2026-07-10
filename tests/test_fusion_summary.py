from pathlib import Path

from utils.fusion_summary import format_fusion_summary, summarize_fusion_csv, write_fusion_summary


FIXTURES = Path(__file__).parent / "fixtures"


def test_summarize_fusion_csv_calculates_monitor_session_breakdowns():
    summary = summarize_fusion_csv(FIXTURES / "monitor_fusion_summary.csv")

    assert summary["record_count"] == 4
    assert summary["duration_seconds"] == 15.0
    assert summary["duration_label"] == "0:00:15"
    assert summary["dominant_state"] == "high_attention"
    assert summary["state_breakdown"] == {"high_attention": 2, "moderate": 1, "distracted": 1}
    assert summary["labeled_phases"] == {"neutral": 1, "happy": 2, "sad": 1}
    assert summary["profile_phase_breakdown"] == {"neutral": 1, "happy": 2, "mad": 1}
    assert summary["profile_label_accuracy_pct"] == 75.0
    assert summary["avg_engagement"] == 0.68
    assert summary["avg_fatigue"] == 0.26
    assert summary["avg_tension"] == 0.26
    assert summary["avg_positivity"] == 0.598
    assert summary["avg_distraction"] == 27.0


def test_format_fusion_summary_includes_replay_report_regression_sections():
    summary = summarize_fusion_csv(FIXTURES / "replay_report_session.csv")
    report = format_fusion_summary(summary, "replay_report_session.csv")

    assert "SYNAPSE FUSION SESSION SUMMARY" in report
    assert "Session: replay_report_session.csv" in report
    assert "- Dominant state: moderate" in report
    assert "- none" in report
    assert "- neutral: 2 frames" in report
    assert "- sad: 1 frames" in report


def test_write_fusion_summary_writes_neighbor_text_report(tmp_path):
    session_path = tmp_path / "session.csv"
    session_path.write_text(
        (FIXTURES / "monitor_fusion_summary.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    report = write_fusion_summary(session_path)

    assert report.startswith("SYNAPSE FUSION SESSION SUMMARY")
    assert session_path.with_suffix(".summary.txt").read_text(encoding="utf-8") == report + "\n"


def test_summarize_fusion_csv_handles_empty_csv(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text(
        "elapsed_sec,state,engagement,fatigue,tension,positivity,distraction\n",
        encoding="utf-8",
    )

    assert summarize_fusion_csv(empty_csv) == {"record_count": 0}
    assert format_fusion_summary({"record_count": 0}, "empty.csv") == "No records found in empty.csv."
