import argparse
from pathlib import Path

from utils.fusion_replay import load_rows, replay_fusion_rows, row_to_fusion
from utils.fusion_summary import write_fusion_summary

SESSION_GLOB = "fusion_track_*.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a Synapse fusion session")
    parser.add_argument("session", nargs="?", help="Path to fusion CSV (default: latest)")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen (F to toggle)")
    return parser.parse_args()


def replay_fusion_session(session_path: Path, *, fullscreen: bool = False) -> None:
    rows = load_rows(session_path)
    if not rows:
        print(f"No records to replay in {session_path}")
        return

    print(write_fusion_summary(session_path))

    replay_fusion_rows(
        rows,
        row_to_fusion,
        title="FUSION REPLAY",
        window_title="Synapse - Fusion Replay",
        session_path=session_path,
        fullscreen=fullscreen,
    )


def main() -> None:
    args = parse_args()
    if args.session:
        path = Path(args.session)
    else:
        sessions = sorted(Path("sessions").glob(SESSION_GLOB))
        if not sessions:
            raise SystemExit("No fusion session CSV found in sessions/")
        path = sessions[-1]
    replay_fusion_session(path, fullscreen=args.fullscreen)

if __name__ == "__main__":
    main()
