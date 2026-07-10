from pathlib import Path
import sys

from utils.fusion_summary import write_fusion_summary


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        sessions = sorted(Path("sessions").glob("fusion_track_*.csv"))
        if not sessions:
            raise SystemExit("No fusion session CSV found in sessions/")
        path = sessions[-1]

    print(write_fusion_summary(path))


if __name__ == "__main__":
    main()
