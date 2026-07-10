"""Create a sanitized local pilot summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.pilot_report import write_pilot_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sanitized Synapse pilot summary")
    parser.add_argument("sessions", nargs="*", help="Optional monitor CSV paths")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(item) for item in args.sessions] if args.sessions else None
    output = write_pilot_summary(paths)
    print(f"Pilot summary exported to {output}")


if __name__ == "__main__":
    main()
