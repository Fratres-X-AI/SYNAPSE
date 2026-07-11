"""Synapse CLI launcher with optional system tray."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path

from utils.app_paths import data_inventory, delete_all_user_data
from utils.privacy import ensure_privacy_consent
from utils.product import PRODUCT_NAME, VERSION
from utils.settings import UserSettings, load_settings, save_settings

ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", ROOT))

SCRIPTS = {
    "home": "synapse_home.py",
    "onboard": "synapse_onboard.py",
    "monitor": "synapse_monitor.py",
    "replay": "replay_monitor.py",
    "fusion": "synapse_fusion.py",
    "showcase": "synapse_showcase.py",
    "pilot-summary": "synapse_pilot_summary.py",
}
SCRIPT_MODULES = {
    "synapse_home.py": "synapse_home",
    "synapse_onboard.py": "test_onboard",
    "synapse_monitor.py": "test_monitor",
    "replay_monitor.py": "replay_monitor",
    "synapse_fusion.py": "test_fusion_track",
    "synapse_showcase.py": "test_showcase",
    "synapse_pilot_summary.py": "synapse_pilot_summary",
}
UTILITY_COMMANDS = {"first-run", "privacy", "data", "delete-data", "settings"}
RETURN_HOME_FLAG = "--return-home"


def setup_entry() -> tuple[str, str]:
    """Return tray/home label and launcher command for first-time or repeat setup."""
    from utils.user_profiles import user_is_configured

    if user_is_configured():
        return "Recalibrate", "onboard"
    return "Get Started", "first-run"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def wants_return_home(extra_args: list[str] | None = None) -> bool:
    return RETURN_HOME_FLAG in (extra_args or sys.argv[1:])


def relaunch_home() -> None:
    if is_frozen():
        subprocess.Popen([sys.executable, "home"], cwd=ROOT)
        return

    launcher = ROOT / "synapse_launcher.py"
    if not launcher.exists():
        return
    subprocess.Popen([sys.executable, str(launcher), "home"], cwd=ROOT)


def ensure_cli_console(command: str | None) -> None:
    """Attach a console for frozen windowed builds running CLI subcommands."""
    if not is_frozen() or not command or command == "home":
        return
    if sys.platform != "win32":
        return
    try:
        import ctypes

        if ctypes.windll.kernel32.GetConsoleWindow():
            return
        ctypes.windll.kernel32.AllocConsole()
        sys.stdout = open("CONOUT$", "w", encoding="utf-8")  # noqa: SIM115
        sys.stderr = open("CONOUT$", "w", encoding="utf-8")  # noqa: SIM115
    except OSError:
        pass


def run_frozen_module(module_name: str, argv0: str, extra_args: list[str]) -> int:
    import importlib

    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        print(f"Error: {module_name} has no main()", file=sys.stderr)
        return 1

    previous_argv = sys.argv
    sys.argv = [argv0, *extra_args]
    try:
        result = module.main()
        if isinstance(result, int):
            return result
        return 0
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return int(code) if isinstance(code, int) else 1
    finally:
        sys.argv = previous_argv

DESCRIPTION = f"""\
{PRODUCT_NAME} v{VERSION} — cognitive monitoring launcher.

Commands:
  home      Open graphical launcher shell
  first-run Privacy, onboarding, then home (or monitor from CLI)
  onboard   Run unified onboarding wizard (calibration + emotion profile)
  monitor   Start production monitor mode
  showcase  Demo mode with elite landmark shell + flight HUD
  replay    Replay latest monitor session (or pass a CSV path)
  fusion    Run fusion track mode (live N/H/S/M labeling)
  data      Show local data inventory
  settings  View or update local app settings
  privacy   Show privacy notice and record consent
  delete-data Delete local Synapse user data

Examples:
  python synapse_launcher.py
  python synapse_launcher.py home
  python synapse_launcher.py first-run
  python synapse_launcher.py onboard
  python synapse_launcher.py monitor
  python synapse_launcher.py showcase --fullscreen
  python synapse_launcher.py replay
  python synapse_launcher.py data
  python synapse_launcher.py monitor --fullscreen
  python synapse_launcher.py replay --fullscreen
  python synapse_launcher.py --tray
"""


def run_script(script_name: str, extra_args: list[str] | None = None) -> int:
    argv = extra_args or []
    if is_frozen():
        module_name = SCRIPT_MODULES.get(script_name)
        if module_name is None:
            print(f"Error: no frozen handler for {script_name}", file=sys.stderr)
            return 1
        print(f"Running: Synapse {script_name}")
        return run_frozen_module(module_name, script_name, argv)

    script_path = ROOT / script_name
    if not script_path.exists():
        script_path = RESOURCE_ROOT / script_name
    if not script_path.exists():
        print(f"Error: {script_path} not found", file=sys.stderr)
        return 1
    cmd = [sys.executable, str(script_path), *argv]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=ROOT)


def run_command(command: str, extra_args: list[str] | None = None) -> int:
    if command in UTILITY_COMMANDS:
        return run_utility(command, extra_args or [])
    script = SCRIPTS.get(command)
    if script is None:
        print(f"Unknown command: {command}", file=sys.stderr)
        return 1
    return run_script(script, extra_args)


def run_utility(command: str, extra_args: list[str]) -> int:
    if command == "first-run":
        if not ensure_privacy_consent():
            return 1
        onboard_code = run_command("onboard", extra_args)
        if onboard_code != 0:
            return onboard_code
        if wants_return_home(extra_args):
            return 0
        return run_command("monitor", extra_args)

    if command == "privacy":
        return 0 if ensure_privacy_consent() else 1

    if command == "data":
        inventory = data_inventory()
        print("Synapse local data")
        for key, value in inventory.items():
            print(f"- {key}: {value}")
        return 0

    if command == "delete-data":
        response = input("Delete all local Synapse user data? Type DELETE to confirm: ").strip()
        if response != "DELETE":
            print("Delete cancelled.")
            return 1
        delete_all_user_data()
        print("Deleted local Synapse user data.")
        return 0

    if command == "settings":
        return run_settings_command(extra_args)

    print(f"Unknown utility command: {command}", file=sys.stderr)
    return 1


def run_settings_command(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="synapse_launcher settings")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--fullscreen-default", choices=("on", "off"))
    parser.add_argument("--retention-days", type=int)
    parser.add_argument("--desktop-export", choices=("on", "off"))
    parsed = parser.parse_args(args)

    current = load_settings()
    updated = UserSettings(
        camera_index=parsed.camera_index if parsed.camera_index is not None else current.camera_index,
        fullscreen_default=(
            parsed.fullscreen_default == "on"
            if parsed.fullscreen_default is not None
            else current.fullscreen_default
        ),
        retention_days=(
            parsed.retention_days if parsed.retention_days is not None else current.retention_days
        ),
        export_reports_to_desktop=(
            parsed.desktop_export == "on"
            if parsed.desktop_export is not None
            else current.export_reports_to_desktop
        ),
        privacy_mode=current.privacy_mode,
    )
    if updated != current:
        save_settings(updated)
    print("Synapse settings")
    for key, value in updated.__dict__.items():
        print(f"- {key}: {value}")
    return 0


def command_argv(command: str, extra: list[str], fullscreen: bool) -> list[str]:
    argv = list(extra)
    if command == "showcase":
        if "--windowed" not in argv and "--fullscreen" not in argv:
            argv.append("--fullscreen")
    elif fullscreen and "--fullscreen" not in argv:
        argv.append("--fullscreen")
    return argv


def _try_import_tray():
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        return None, None, None
    return pystray, Image, ImageDraw


def _icon_asset_path(name: str) -> Path | None:
    for base in (ROOT, RESOURCE_ROOT):
        candidate = base / "assets" / name
        if candidate.exists():
            return candidate
    return None


def _make_tray_icon(Image, ImageDraw):
    asset = _icon_asset_path("synapse_icon.png")
    if asset is not None:
        image = Image.open(asset).convert("RGBA")
        return image.resize((64, 64), Image.Resampling.LANCZOS)

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

    setup_label, setup_command = setup_entry()
    menu = pystray.Menu(
        pystray.MenuItem("Home", launch("home")),
        pystray.MenuItem(setup_label, launch(setup_command)),
        pystray.MenuItem("Monitor", launch("monitor")),
        pystray.MenuItem("Showcase", launch("showcase")),
        pystray.MenuItem("Replay (latest)", launch("replay")),
        pystray.MenuItem("Pilot Summary", launch("pilot-summary")),
        pystray.MenuItem("Data Inventory", launch("data")),
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
        choices=sorted(set(SCRIPTS) | UTILITY_COMMANDS),
        help="Synapse mode to launch",
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
    args, extra = parser.parse_known_args(argv)

    if args.tray and args.command is None:
        return run_tray()

    if args.command:
        ensure_cli_console(args.command)
        return run_command(
            args.command,
            command_argv(args.command, extra, args.fullscreen),
        )

    if args.tray:
        tray_thread = threading.Thread(target=run_tray, daemon=True)
        tray_thread.start()
        tray_thread.join()
        return 0

    return run_command("home", command_argv("home", extra, args.fullscreen))


if __name__ == "__main__":
    raise SystemExit(main())
