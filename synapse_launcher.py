"""Synapse CLI launcher with optional system tray."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    "onboard": "test_onboard.py",
    "monitor": "test_monitor.py",
    "replay": "replay_monitor.py",
    "fusion": "test_fusion_track.py",
}

DESCRIPTION = """\
Synapse cognitive monitoring launcher.

Commands:
  onboard   Run unified onboarding wizard (calibration + emotion profile)
  monitor   Start production monitor mode
  replay    Replay latest monitor session (or pass a CSV path)
  fusion    Run fusion track mode (live N/H/S/M labeling)

Examples:
  python synapse_launcher.py onboard
  python synapse_launcher.py monitor
  python synapse_launcher.py replay
  python synapse_launcher.py monitor --fullscreen
  python synapse_launcher.py replay --fullscreen
  python synapse_launcher.py --tray
"""


def run_script(script_name: str, extra_args: list[str] | None = None) -> int:
    script_path = ROOT / script_name
    if not script_path.exists():
        print(f"Error: {script_path} not found", file=sys.stderr)
        return 1
    cmd = [sys.executable, str(script_path), *(extra_args or [])]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=ROOT)


def run_command(command: str, extra_args: list[str] | None = None) -> int:
    script = SCRIPTS.get(command)
    if script is None:
        print(f"Unknown command: {command}", file=sys.stderr)
        return 1
    return run_script(script, extra_args)


def command_argv(command: str, extra: list[str], fullscreen: bool) -> list[str]:
    argv = list(extra)
    if fullscreen and "--fullscreen" not in argv:
        argv.append("--fullscreen")
    return argv


def _try_import_tray():
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        return None, None, None
    return pystray, Image, ImageDraw


def _make_tray_icon(Image, ImageDraw):
    size = 64
    image = Image.new("RGB", (size, size), color=(18, 22, 32))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, size - 8, size - 8), fill=(0, 200, 140))
    draw.ellipse((22, 22, size - 22, size - 22), fill=(18, 22, 32))
    return image


def run_tray() -> int:
    pystray, Image, ImageDraw = _try_import_tray()
    if pystray is None:
        print("System tray requires: pip install pystray pillow")
        print("Install with: pip install -r requirements-dev.txt")
        return 1

    icon_image = _make_tray_icon(Image, ImageDraw)

    def launch(command: str):
        def _action(_icon, _item):
            threading.Thread(
                target=run_command,
                args=(command,),
                daemon=True,
            ).start()

        return _action

    menu = pystray.Menu(
        pystray.MenuItem("Onboard", launch("onboard")),
        pystray.MenuItem("Monitor", launch("monitor")),
        pystray.MenuItem("Replay (latest)", launch("replay")),
        pystray.MenuItem("Fusion Track", launch("fusion")),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", lambda _icon, _item: _icon.stop()),
    )

    icon = pystray.Icon("synapse", icon_image, "Synapse", menu)
    print("Synapse tray running. Right-click the tray icon for commands.")
    icon.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synapse_launcher",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(SCRIPTS),
        help="Synapse mode to launch",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Extra arguments (e.g. replay CSV path)",
    )
    parser.add_argument(
        "--tray",
        action="store_true",
        help="Show system tray icon (requires pystray + pillow)",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Start the selected mode in fullscreen (press F to toggle)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.tray and args.command is None:
        return run_tray()

    if args.command:
        return run_command(
            args.command,
            command_argv(args.command, args.extra or [], args.fullscreen),
        )

    if args.tray:
        tray_thread = threading.Thread(target=run_tray, daemon=True)
        tray_thread.start()

    parser.print_help()
    return 0 if args.tray else 1


if __name__ == "__main__":
    raise SystemExit(main())
