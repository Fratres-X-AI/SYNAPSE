"""Graphical Synapse home shell — launch monitor, showcase, onboard, replay."""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

ROOT = Path(__file__).resolve().parent
ICON_PATH = ROOT / "assets" / "synapse.ico"


def _set_window_icon(window: tk.Tk) -> None:
    if not ICON_PATH.exists():
        return
    try:
        window.iconbitmap(default=str(ICON_PATH))
    except tk.TclError:
        pass


def _launch(command: str, extra: list[str] | None = None) -> None:
    argv = extra or []
    if getattr(sys, "frozen", False):
        subprocess.Popen([sys.executable, command, *argv], cwd=ROOT)
        return

    launcher = ROOT / "synapse_launcher.py"
    if not launcher.exists():
        print(f"Error: {launcher} not found", file=sys.stderr)
        return
    subprocess.Popen([sys.executable, str(launcher), command, *argv], cwd=ROOT)


def run_home_shell() -> int:
    window = tk.Tk()
    _set_window_icon(window)
    window.title("Synapse")
    window.configure(bg="#0c1018")
    window.geometry("520x420")
    window.minsize(480, 380)

    title_font = tkfont.Font(family="Segoe UI", size=22, weight="bold")
    body_font = tkfont.Font(family="Segoe UI", size=11)
    button_font = tkfont.Font(family="Segoe UI", size=12, weight="bold")

    header = tk.Label(
        window,
        text="SYNAPSE",
        fg="#00c88c",
        bg="#0c1018",
        font=title_font,
    )
    header.pack(pady=(28, 6))
    subtitle = tk.Label(
        window,
        text="Cognitive presence monitoring",
        fg="#9aa3b2",
        bg="#0c1018",
        font=body_font,
    )
    subtitle.pack(pady=(0, 18))

    button_frame = tk.Frame(window, bg="#0c1018")
    button_frame.pack(fill="both", expand=True, padx=36, pady=8)

    commands = [
        ("Monitor", "monitor", "Production session with full HUD"),
        ("Showcase", "showcase", "Demo mode — all overlays"),
        ("Onboard", "onboard", "Calibration + emotion profile"),
        ("Replay", "replay", "Replay latest session"),
        ("First Run", "first-run", "Privacy, onboard, then monitor"),
    ]

    def make_handler(command: str):
        def _handler() -> None:
            window.destroy()
            _launch(command)

        return _handler

    for index, (label, command, hint) in enumerate(commands):
        row = tk.Frame(button_frame, bg="#0c1018")
        row.pack(fill="x", pady=6)
        btn = tk.Button(
            row,
            text=label,
            command=make_handler(command),
            font=button_font,
            fg="#e8edf5",
            bg="#16202f",
            activebackground="#1f2d42",
            activeforeground="#ffffff",
            relief="flat",
            padx=16,
            pady=10,
            cursor="hand2",
        )
        btn.pack(side="left", fill="x", expand=True)
        tip = tk.Label(row, text=hint, fg="#6f7a8c", bg="#0c1018", font=body_font)
        tip.pack(side="right", padx=(12, 0))

    footer = tk.Label(
        window,
        text="Desktop: Synapse shortcut  |  Tray: synapse_launcher.py --tray",
        fg="#5c6678",
        bg="#0c1018",
        font=body_font,
    )
    footer.pack(pady=(8, 20))

    window.mainloop()
    return 0


def main() -> int:
    return run_home_shell()


if __name__ == "__main__":
    raise SystemExit(main())
