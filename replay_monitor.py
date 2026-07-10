import argparse
import sys
from pathlib import Path

from utils.fusion_replay import load_rows, replay_fusion_rows, row_to_fusion_monitor
from utils.manager_report import write_manager_report
from utils.config import Config

SESSION_DIR = Path("sessions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a Synapse monitor session")
    parser.add_argument("session", nargs="?", help="Path to monitor CSV (default: latest)")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen (F to toggle)")
    return parser.parse_args()


def latest_monitor_session(directory: Path = SESSION_DIR) -> Path | None:
    search_dirs = [Config().session_dir, directory]
    sessions = []
    for search_dir in search_dirs:
        if search_dir.exists():
            sessions.extend(search_dir.glob("monitor_*.csv"))
    sessions = sorted(set(sessions), key=lambda p: p.stat().st_mtime)
    return sessions[-1] if sessions else None


def replay_monitor_session(session_path: Path, *, fullscreen: bool = False) -> None:
    rows = load_rows(session_path)
    if not rows:
        print(f"No records to replay in {session_path}")
        return

    if session_path.stat().st_size > 120:
        print(write_manager_report(session_path, export_desktop=False))

    replay_fusion_rows(
        rows,
        row_to_fusion_monitor,
        title="MONITOR REPLAY",
        window_title="Synapse - Monitor Replay",
        session_path=session_path,
        fullscreen=fullscreen,
    )


def main() -> None:
    args = parse_args()
    if args.session:
        path = Path(args.session)
    else:
        path = latest_monitor_session()
        if path is None:
            raise SystemExit("No monitor session CSV found in sessions/")
    replay_monitor_session(path, fullscreen=args.fullscreen)


if __name__ == "__main__":
    main()
