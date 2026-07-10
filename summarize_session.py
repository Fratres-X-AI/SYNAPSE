import sys

from utils.session_summary import summarize_latest_or_path


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        session_path, report = summarize_latest_or_path(path)
    except FileNotFoundError as error:
        print(error)
        raise SystemExit(1) from error

    print(report)
    print(f"\nSaved to {session_path.with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()
