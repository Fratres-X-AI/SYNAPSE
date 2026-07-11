"""Tkinter dialogs for consent, errors, settings, and session review."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

from src.ui.theme import COLORS
from utils.product import PRODUCT_NAME, TAGLINE, VERSION
from utils.settings import UserSettings, load_settings, save_settings

ROOT = Path(__file__).resolve().parents[2]
ICON_PATH = ROOT / "assets" / "synapse.ico"


def _apply_icon(window: tk.Tk | tk.Toplevel) -> None:
    if not ICON_PATH.exists():
        return
    try:
        window.iconbitmap(default=str(ICON_PATH))
    except tk.TclError:
        pass


def _base_window(title: str, *, width: int, height: int) -> tk.Tk:
    window = tk.Tk()
    window.title(title)
    window.configure(bg=COLORS["bg"])
    window.geometry(f"{width}x{height}")
    window.minsize(width, height)
    window.resizable(True, True)
    _apply_icon(window)
    return window


def _shell(parent: tk.Tk | tk.Toplevel, *, padx: int = 20, pady: int = 16) -> tk.Frame:
    frame = tk.Frame(parent, bg=COLORS["panel"], highlightbackground="#151d2a", highlightthickness=1)
    frame.pack(fill="both", expand=True, padx=padx, pady=pady)
    return frame


def show_error(title: str, message: str) -> None:
    window = _base_window(title, width=460, height=280)
    shell = _shell(window)
    tk.Label(shell, text=title, fg=COLORS["danger"], bg=COLORS["panel"], font=("Segoe UI", 14, "bold")).pack(
        anchor="w", padx=18, pady=(16, 8)
    )
    tk.Label(
        shell,
        text=message,
        fg=COLORS["text"],
        bg=COLORS["panel"],
        font=("Segoe UI", 10),
        justify="left",
        wraplength=400,
    ).pack(anchor="w", padx=18, pady=(0, 12))
    tk.Button(
        shell,
        text="Close",
        command=window.destroy,
        bg=COLORS["card_hover"],
        fg=COLORS["text"],
        relief="flat",
        padx=16,
        pady=8,
        cursor="hand2",
    ).pack(pady=(0, 16))
    window.mainloop()


def show_camera_error(message: str) -> None:
    show_error(
        "Camera unavailable",
        f"{message}\n\n"
        "Try:\n"
        "• Close other apps using the webcam\n"
        "• Allow camera access in Windows Settings → Privacy → Camera\n"
        "• Change camera index in Home → Settings",
    )


def ask_privacy_consent(consent_text: str) -> bool:
    window = _base_window(f"{PRODUCT_NAME} — Privacy", width=560, height=520)
    accepted = {"value": False}
    shell = _shell(window)

    tk.Label(
        shell,
        text="Privacy notice",
        fg=COLORS["title"],
        bg=COLORS["panel"],
        font=("Segoe UI", 16, "bold"),
    ).pack(anchor="w", padx=18, pady=(16, 8))
    tk.Label(
        shell,
        text="Local processing only. No video is saved.",
        fg=COLORS["muted"],
        bg=COLORS["panel"],
        font=("Segoe UI", 10),
    ).pack(anchor="w", padx=18, pady=(0, 10))

    text_box = scrolledtext.ScrolledText(
        shell,
        wrap="word",
        height=16,
        bg=COLORS["card"],
        fg=COLORS["text"],
        insertbackground=COLORS["text"],
        relief="flat",
        font=("Segoe UI", 10),
    )
    text_box.pack(fill="both", expand=True, padx=18, pady=(0, 12))
    text_box.insert("1.0", consent_text.strip())
    text_box.configure(state="disabled")

    buttons = tk.Frame(shell, bg=COLORS["panel"])
    buttons.pack(fill="x", padx=18, pady=(0, 16))

    def _accept() -> None:
        accepted["value"] = True
        window.destroy()

    tk.Button(
        buttons,
        text="Accept and continue",
        command=_accept,
        bg=COLORS["title"],
        fg=COLORS["bg"],
        relief="flat",
        padx=14,
        pady=8,
        cursor="hand2",
    ).pack(side="left")
    tk.Button(
        buttons,
        text="Decline",
        command=window.destroy,
        bg=COLORS["card_hover"],
        fg=COLORS["text"],
        relief="flat",
        padx=14,
        pady=8,
        cursor="hand2",
    ).pack(side="left", padx=(10, 0))

    window.mainloop()
    return accepted["value"]


def _form_row(parent: tk.Frame, label: str, widget: tk.Widget) -> None:
    line = tk.Frame(parent, bg=COLORS["panel"])
    line.pack(fill="x", pady=6)
    tk.Label(line, text=label, fg=COLORS["muted"], bg=COLORS["panel"], width=18, anchor="w").pack(side="left")
    widget.pack(side="left", fill="x", expand=True)


def show_settings_dialog() -> None:
    settings = load_settings()
    window = _base_window(f"{PRODUCT_NAME} — Settings", width=480, height=420)
    shell = _shell(window)

    tk.Label(shell, text="Settings", fg=COLORS["title"], bg=COLORS["panel"], font=("Segoe UI", 16, "bold")).pack(
        anchor="w", padx=18, pady=(16, 12)
    )

    form = tk.Frame(shell, bg=COLORS["panel"])
    form.pack(fill="x", padx=18)

    camera_var = tk.StringVar(value=str(settings.camera_index))
    fullscreen_var = tk.StringVar(value="on" if settings.fullscreen_default else "off")
    retention_var = tk.StringVar(value=str(settings.retention_days))
    export_var = tk.StringVar(value="on" if settings.export_reports_to_desktop else "off")

    _form_row(form, "Camera index", tk.Entry(form, textvariable=camera_var, bg=COLORS["card"], fg=COLORS["text"]))
    _form_row(
        form,
        "Fullscreen default",
        tk.OptionMenu(form, fullscreen_var, "on", "off"),
    )
    _form_row(form, "Retention days", tk.Entry(form, textvariable=retention_var, bg=COLORS["card"], fg=COLORS["text"]))
    _form_row(
        form,
        "Desktop export",
        tk.OptionMenu(form, export_var, "on", "off"),
    )

    def _save() -> None:
        try:
            updated = UserSettings(
                camera_index=int(camera_var.get().strip()),
                fullscreen_default=fullscreen_var.get() == "on",
                retention_days=max(1, int(retention_var.get().strip())),
                export_reports_to_desktop=export_var.get() == "on",
                privacy_mode=settings.privacy_mode,
                active_user=settings.active_user,
            )
        except ValueError:
            show_error("Invalid settings", "Camera index and retention days must be numbers.")
            return
        save_settings(updated)
        window.destroy()

    tk.Button(
        shell,
        text="Save",
        command=_save,
        bg=COLORS["title"],
        fg=COLORS["bg"],
        relief="flat",
        padx=16,
        pady=8,
        cursor="hand2",
    ).pack(pady=18)


def show_about_dialog() -> None:
    window = _base_window(f"About {PRODUCT_NAME}", width=420, height=300)
    shell = _shell(window)
    tk.Label(shell, text=PRODUCT_NAME, fg=COLORS["title"], bg=COLORS["panel"], font=("Segoe UI", 20, "bold")).pack(
        pady=(20, 4)
    )
    tk.Label(shell, text=f"Version {VERSION}", fg=COLORS["muted"], bg=COLORS["panel"], font=("Segoe UI", 10)).pack()
    tk.Label(shell, text=TAGLINE, fg=COLORS["text"], bg=COLORS["panel"], font=("Segoe UI", 11)).pack(pady=(8, 16))
    tk.Label(
        shell,
        text="Local-first focus support.\nNot medical advice or employee monitoring.",
        fg=COLORS["muted"],
        bg=COLORS["panel"],
        font=("Segoe UI", 9),
        justify="center",
    ).pack(pady=(0, 16))
    tk.Button(
        shell,
        text="Close",
        command=window.destroy,
        bg=COLORS["card_hover"],
        fg=COLORS["text"],
        relief="flat",
        padx=16,
        pady=8,
    ).pack(pady=(0, 16))
    window.mainloop()


def show_session_summary(
    *,
    profile_name: str,
    duration_seconds: float,
    report_text: str,
    report_path: Path | None = None,
    return_home: bool = False,
) -> bool:
    """Show session report. Returns True if user chose to return home."""
    window = _base_window(f"{PRODUCT_NAME} — Session complete", width=620, height=560)
    go_home = {"value": return_home}
    shell = _shell(window)

    minutes = int(duration_seconds // 60)
    tk.Label(
        shell,
        text="Session complete",
        fg=COLORS["title"],
        bg=COLORS["panel"],
        font=("Segoe UI", 16, "bold"),
    ).pack(anchor="w", padx=18, pady=(16, 4))
    tk.Label(
        shell,
        text=f"Profile: {profile_name}  •  Duration: {minutes} min",
        fg=COLORS["muted"],
        bg=COLORS["panel"],
        font=("Segoe UI", 10),
    ).pack(anchor="w", padx=18, pady=(0, 10))

    text_box = scrolledtext.ScrolledText(
        shell,
        wrap="word",
        bg=COLORS["card"],
        fg=COLORS["text"],
        relief="flat",
        font=("Consolas", 10),
    )
    text_box.pack(fill="both", expand=True, padx=18, pady=(0, 10))
    text_box.insert("1.0", report_text.strip() or "No report generated for this session.")
    text_box.configure(state="disabled")

    if report_path is not None:
        tk.Label(
            shell,
            text=f"Saved: {report_path}",
            fg=COLORS["footer"],
            bg=COLORS["panel"],
            font=("Segoe UI", 8),
        ).pack(anchor="w", padx=18)

    buttons = tk.Frame(shell, bg=COLORS["panel"])
    buttons.pack(fill="x", padx=18, pady=(8, 16))

    def _home() -> None:
        go_home["value"] = True
        window.destroy()

    tk.Button(
        buttons,
        text="Return to home",
        command=_home,
        bg=COLORS["title"],
        fg=COLORS["bg"],
        relief="flat",
        padx=14,
        pady=8,
        cursor="hand2",
    ).pack(side="left")
    tk.Button(
        buttons,
        text="Close",
        command=window.destroy,
        bg=COLORS["card_hover"],
        fg=COLORS["text"],
        relief="flat",
        padx=14,
        pady=8,
        cursor="hand2",
    ).pack(side="left", padx=(10, 0))

    window.mainloop()
    return go_home["value"]
