"""Graphical Synapse home shell — launch monitor, showcase, replay, setup."""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, simpledialog

from PIL import Image, ImageDraw, ImageFilter, ImageTk

from synapse_launcher import setup_entry

from utils.user_profiles import (
    create_user_profile,
    get_active_user_display_name,
    get_user_profile,
    list_user_profiles,
    set_active_user,
    user_is_configured,
)
from utils.product import VERSION
from utils.onboarding_progress import load_progress
from src.ui.dialogs import show_about_dialog, show_settings_dialog

ROOT = Path(__file__).resolve().parent
ASSET_ROOT = Path(getattr(sys, "_MEIPASS", ROOT))
ICON_PATH = ASSET_ROOT / "assets" / "synapse.ico"
ICON_IMAGE_PATH = ASSET_ROOT / "assets" / "synapse_icon.png"

CONTENT_WIDTH = 336
MIN_WINDOW_WIDTH = 400
MIN_WINDOW_HEIGHT = 760
WINDOW_WIDTH = 420
WINDOW_HEIGHT = 820
SHELL_PAD = 12

from src.ui.theme import COLORS


def _set_window_icon(window: tk.Tk) -> None:
    if not ICON_PATH.exists():
        return
    try:
        window.iconbitmap(default=str(ICON_PATH))
    except tk.TclError:
        pass


def _launch(command: str, extra: list[str] | None = None) -> None:
    argv = list(extra or [])
    if command in ("onboard", "first-run", "monitor"):
        from synapse_launcher import RETURN_HOME_FLAG

        if RETURN_HOME_FLAG not in argv:
            argv.append(RETURN_HOME_FLAG)
    if getattr(sys, "frozen", False):
        subprocess.Popen([sys.executable, command, *argv], cwd=ROOT)
        return

    launcher = ROOT / "synapse_launcher.py"
    if not launcher.exists():
        print(f"Error: {launcher} not found", file=sys.stderr)
        return
    subprocess.Popen([sys.executable, str(launcher), command, *argv], cwd=ROOT)


def _setup_command() -> tuple[str, str, str]:
    label, command = setup_entry()
    if load_progress() is not None and not user_is_configured():
        return "Resume Setup", "onboard", "Continue your saved onboarding session"
    if command == "onboard":
        return label, command, "Redo attention + expression baseline"
    return label, command, "Privacy notice, setup, then home menu"


def _blend(a: int, b: int, t: float) -> int:
    return int(a * (1 - t) + b * t)


def _gradient_fill(width: int, height: int) -> Image.Image:
    start = (255, 122, 24)
    end = (232, 48, 160)
    image = Image.new("RGBA", (width, height))
    pixels = image.load()
    for x in range(width):
        t = x / max(width - 1, 1)
        color = (
            _blend(start[0], end[0], t),
            _blend(start[1], end[1], t),
            _blend(start[2], end[2], t),
            255,
        )
        for y in range(height):
            pixels[x, y] = color
    return image


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def _apply_rounded(image: Image.Image, radius: int) -> Image.Image:
    mask = _rounded_mask(image.size, radius)
    output = Image.new("RGBA", image.size, (0, 0, 0, 0))
    output.paste(image, (0, 0), mask)
    return output


def _card_surface(
    width: int,
    height: int,
    *,
    fill: str,
    radius: int = 14,
    border: str | None = None,
    sheen: bool = False,
) -> Image.Image:
    base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    body = Image.new("RGBA", (width, height), fill)
    body = _apply_rounded(body, radius)
    base = Image.alpha_composite(base, body)

    if border:
        outline = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(outline)
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, outline=border, width=1)
        base = Image.alpha_composite(base, outline)

    if sheen:
        sheen_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sheen_img)
        draw.rounded_rectangle((1, 1, width - 2, height // 2), radius=radius - 1, fill=(255, 255, 255, 18))
        base = Image.alpha_composite(base, sheen_img)

    return base


def _with_shadow(surface: Image.Image, *, offset_y: int = 4, blur: int = 10) -> Image.Image:
    width, height = surface.size
    canvas = Image.new("RGBA", (width, height + offset_y + blur), (0, 0, 0, 0))
    shadow = Image.new("RGBA", surface.size, (0, 0, 0, 0))
    alpha = surface.split()[3]
    shadow.paste((0, 0, 0, 110), (0, 0), alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    canvas.paste(shadow, (0, offset_y), shadow)
    canvas.paste(surface, (0, 0), surface)
    return canvas


def _window_backdrop(width: int, height: int) -> Image.Image:
    image = Image.new("RGBA", (width, height), COLORS["bg"])
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    draw.ellipse((width // 2 - 180, -40, width // 2 + 180, 220), fill=COLORS["glow_orange"])
    draw.ellipse((width // 2 - 150, 20, width // 2 + 150, 260), fill=COLORS["glow_magenta"])
    glow = glow.filter(ImageFilter.GaussianBlur(42))
    image = Image.alpha_composite(image, glow)

    panel = Image.new("RGBA", (width - 24, height - 24), COLORS["panel"])
    panel = _apply_rounded(panel, 22)
    panel_outline = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    ImageDraw.Draw(panel_outline).rounded_rectangle(
        (0, 0, panel.size[0] - 1, panel.size[1] - 1),
        radius=22,
        outline="#151d2a",
        width=1,
    )
    panel = Image.alpha_composite(panel, panel_outline)
    image.paste(panel, (12, 12), panel)
    return image


def _load_icon_image(size: int = 72) -> ImageTk.PhotoImage | None:
    if not ICON_IMAGE_PATH.exists():
        return None
    try:
        image = Image.open(ICON_IMAGE_PATH).convert("RGBA")
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)
    except OSError:
        return None


class _ActionTile:
    def __init__(
        self,
        parent: tk.Widget,
        *,
        title: str,
        subtitle: str,
        command,
        primary: bool = False,
        fonts: dict[str, tkfont.Font],
    ) -> None:
        self.command = command
        self.primary = primary
        self.title = title
        self.subtitle = subtitle
        self.fonts = fonts
        self.height = 84 if primary else 62
        self.radius = 16 if primary else 14
        self.canvas_height = self.height + (10 if primary else 6)
        self.content_width = CONTENT_WIDTH
        self._hovering = False

        self.frame = tk.Frame(parent, bg=COLORS["panel"])
        self.frame.pack(fill="x", pady=(0, 10 if primary else 8))

        self.canvas = tk.Canvas(
            self.frame,
            width=self.content_width,
            height=self.canvas_height,
            bg=COLORS["panel"],
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="x")
        self._rebuild_surfaces()
        self._bind_interactions()
        self.frame.bind("<Configure>", self._on_resize)

    def _on_resize(self, event) -> None:
        if event.widget is not self.frame:
            return
        width = max(CONTENT_WIDTH, event.width)
        if width == self.content_width:
            return
        self.content_width = width
        self.canvas.configure(width=width, height=self.canvas_height)
        self._rebuild_surfaces()

    def _rebuild_surfaces(self) -> None:
        self._normal = self._render_surface(hover=False)
        self._hover = self._render_surface(hover=True)
        self._photo = ImageTk.PhotoImage(self._normal)
        self._photo_hover = ImageTk.PhotoImage(self._hover)
        photo = self._photo_hover if self._hovering else self._photo
        self._draw(photo)

    def _bind_interactions(self) -> None:
        for widget in (self.frame, self.canvas):
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)
            widget.bind("<Button-1>", self._on_click)
        for item_id in self.canvas.find_withtag("tile"):
            self.canvas.tag_bind(item_id, "<Enter>", self._on_enter)
            self.canvas.tag_bind(item_id, "<Leave>", self._on_leave)
            self.canvas.tag_bind(item_id, "<Button-1>", self._on_click)

    def _set_photo(self, photo: ImageTk.PhotoImage) -> None:
        self.canvas.image = photo
        if self._image_id is not None:
            self.canvas.itemconfig(self._image_id, image=photo)
            return
        self._draw(photo)

    def _on_enter(self, _event=None) -> None:
        self._hovering = True
        self.canvas.configure(cursor="hand2")
        self._set_photo(self._photo_hover)

    def _on_leave(self, _event=None) -> None:
        self._hovering = False
        self.canvas.configure(cursor="")
        self._set_photo(self._photo)

    def _on_click(self, _event=None) -> None:
        self.command()

    def update_content(self, *, title: str, subtitle: str, command) -> None:
        self.title = title
        self.subtitle = subtitle
        self.command = command
        self._draw(self._photo_hover if self._hovering else self._photo)

    def _render_surface(self, *, hover: bool) -> Image.Image:
        width = self.content_width
        if self.primary:
            fill = _gradient_fill(width, self.height)
            fill = _apply_rounded(fill, self.radius)
            if hover:
                brighten = Image.new("RGBA", fill.size, (255, 255, 255, 24))
                fill = Image.alpha_composite(fill, brighten)
            surface = _with_shadow(fill, offset_y=5, blur=12)
            return surface

        fill_color = COLORS["card_hover"] if hover else COLORS["card"]
        border = "#31586f" if hover else COLORS["card_border"]
        card = _card_surface(
            width,
            self.height,
            fill=fill_color,
            radius=self.radius,
            border=border,
            sheen=hover,
        )
        return _with_shadow(card, offset_y=3, blur=8)

    def _draw(self, photo: ImageTk.PhotoImage) -> None:
        self.canvas.delete("all")
        self._image_id = self.canvas.create_image(0, 0, anchor="nw", image=photo, tags="tile")
        self.canvas.image = photo
        width = self.content_width

        title_color = "#ffffff" if self.primary else COLORS["text"]
        subtitle_color = "#f7e9ff" if self.primary else COLORS["muted"]
        title_y = 28 if self.primary else 20
        subtitle_y = 54 if self.primary else 40

        self.canvas.create_text(
            20,
            title_y,
            text=self.title,
            anchor="w",
            fill=title_color,
            font=self.fonts["card_title" if self.primary else "row_title"],
            tags="tile",
        )
        self.canvas.create_text(
            20,
            subtitle_y,
            text=self.subtitle,
            anchor="w",
            fill=subtitle_color,
            font=self.fonts["card_sub" if self.primary else "row_sub"],
            tags="tile",
        )
        if not self.primary:
            self.canvas.create_text(
                width - 18,
                self.height // 2 + 2,
                text="›",
                anchor="e",
                fill=COLORS["chevron"],
                font=self.fonts["chevron"],
                tags="tile",
            )

        self.canvas.create_rectangle(
            0,
            0,
            width,
            self.canvas_height,
            outline="",
            fill="",
            tags="tile",
        )
        self._bind_interactions()


def run_home_shell() -> int:
    window = tk.Tk()
    _set_window_icon(window)
    window.title(f"Synapse {VERSION}")
    window.configure(bg=COLORS["bg"])
    window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    window.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
    window.resizable(True, True)

    outer = tk.Frame(window, bg=COLORS["bg"])
    outer.pack(fill="both", expand=True, padx=SHELL_PAD, pady=SHELL_PAD)

    shell = tk.Frame(
        outer,
        bg=COLORS["panel"],
        highlightbackground="#151d2a",
        highlightthickness=1,
    )
    shell.pack(fill="both", expand=True)

    fonts = {
        "brand": tkfont.Font(family="Segoe UI", size=26, weight="bold"),
        "tagline": tkfont.Font(family="Segoe UI", size=10),
        "section": tkfont.Font(family="Segoe UI", size=8, weight="bold"),
        "card_title": tkfont.Font(family="Segoe UI", size=15, weight="bold"),
        "card_sub": tkfont.Font(family="Segoe UI", size=10),
        "row_title": tkfont.Font(family="Segoe UI", size=12, weight="bold"),
        "row_sub": tkfont.Font(family="Segoe UI", size=9),
        "chevron": tkfont.Font(family="Segoe UI", size=18),
        "footer": tkfont.Font(family="Segoe UI", size=8),
    }

    footer = tk.Label(
        shell,
        text="Tray icon for advanced tools",
        fg=COLORS["footer"],
        bg=COLORS["panel"],
        font=fonts["footer"],
    )
    footer.pack(side="bottom", fill="x", pady=(0, 16))

    header = tk.Frame(shell, bg=COLORS["panel"])
    header.pack(fill="x", pady=(22, 12))

    icon_frame = tk.Frame(header, bg=COLORS["panel"])
    icon_frame.pack()
    icon_image = _load_icon_image(72)
    if icon_image is not None:
        window._icon_image = icon_image
        tk.Label(icon_frame, image=icon_image, bg=COLORS["panel"], bd=0).pack()

    tk.Label(header, text="SYNAPSE", fg=COLORS["title"], bg=COLORS["panel"], font=fonts["brand"]).pack(pady=(10, 2))
    tk.Label(
        header,
        text="Cognitive presence monitoring",
        fg=COLORS["muted"],
        bg=COLORS["panel"],
        font=fonts["tagline"],
    ).pack()

    menu = tk.Frame(shell, bg=COLORS["panel"])
    menu.pack(fill="both", expand=True, padx=18, pady=(0, 8))

    profiles = list_user_profiles()
    profile_lookup = {profile.display_name: profile.user_id for profile in profiles}
    selected_profile = tk.StringVar(value=get_active_user_display_name())

    profile_status = tk.StringVar(value=get_user_profile().status_line)

    def refresh_profile_status() -> None:
        profile_status.set(get_user_profile().status_line)

    profile_section = tk.Frame(menu, bg=COLORS["panel"])
    profile_section.pack(fill="x", pady=(0, 12))
    tk.Label(
        profile_section,
        text="PROFILE",
        fg=COLORS["section"],
        bg=COLORS["panel"],
        font=fonts["section"],
    ).pack(anchor="w", pady=(0, 6))

    profile_row = tk.Frame(profile_section, bg=COLORS["card"], highlightbackground=COLORS["card_border"], highlightthickness=1)
    profile_row.pack(fill="x")

    profile_menu = tk.OptionMenu(
        profile_row,
        selected_profile,
        *[profile.display_name for profile in profiles],
    )
    profile_menu.configure(
        bg=COLORS["card"],
        fg=COLORS["text"],
        activebackground=COLORS["card_hover"],
        activeforeground=COLORS["text"],
        highlightthickness=0,
        borderwidth=0,
        font=fonts["row_title"],
        width=18,
    )
    profile_menu["menu"].configure(bg=COLORS["card"], fg=COLORS["text"])
    profile_menu.pack(side="left", fill="x", expand=True, padx=10, pady=8)

    setup_label, setup_command, setup_hint = _setup_command()
    setup_tile: _ActionTile | None = None

    def refresh_setup_tile() -> None:
        nonlocal setup_label, setup_command, setup_hint
        setup_label, setup_command, setup_hint = _setup_command()
        if setup_tile is not None:
            setup_tile.update_content(
                title=setup_label,
                subtitle=setup_hint,
                command=close_and_launch(setup_command),
            )

    def on_profile_change(*_args) -> None:
        display_name = selected_profile.get()
        user_id = profile_lookup.get(display_name)
        if user_id is None:
            return
        set_active_user(user_id)
        refresh_setup_tile()
        refresh_profile_status()

    selected_profile.trace_add("write", on_profile_change)

    def add_profile() -> None:
        name = simpledialog.askstring("New profile", "Who is using Synapse?", parent=window)
        if not name:
            return
        try:
            profile = create_user_profile(name)
        except ValueError as exc:
            messagebox.showerror("Profile", str(exc), parent=window)
            return
        profile_lookup[profile.display_name] = profile.user_id
        menu_list = profile_menu["menu"]
        menu_list.add_command(
            label=profile.display_name,
            command=lambda value=profile.display_name: selected_profile.set(value),
        )
        selected_profile.set(profile.display_name)
        refresh_setup_tile()
        refresh_profile_status()

    add_button = tk.Button(
        profile_row,
        text="+",
        command=add_profile,
        font=fonts["row_title"],
        fg=COLORS["text"],
        bg=COLORS["card_hover"],
        activebackground=COLORS["card_border"],
        activeforeground=COLORS["text"],
        relief="flat",
        width=3,
        cursor="hand2",
    )
    add_button.pack(side="right", padx=(0, 10), pady=8)

    tk.Label(
        profile_section,
        textvariable=profile_status,
        fg=COLORS["footer"],
        bg=COLORS["panel"],
        font=fonts["footer"],
    ).pack(anchor="w", pady=(4, 0))

    def close_and_launch(command: str):
        def _handler() -> None:
            _launch(command)
            window.destroy()

        return _handler

    _ActionTile(
        menu,
        title="Start Monitor",
        subtitle="Production session with full HUD",
        command=close_and_launch("monitor"),
        primary=True,
        fonts=fonts,
    )

    tk.Label(menu, text="TOOLS", fg=COLORS["section"], bg=COLORS["panel"], font=fonts["section"]).pack(anchor="w", pady=(2, 8))

    for label, command, hint in (
        ("Showcase", "showcase", "Demo only — overlays on, no session saved"),
        ("Replay", "replay", "Replay your latest saved session"),
    ):
        _ActionTile(
            menu,
            title=label,
            subtitle=hint,
            command=close_and_launch(command),
            fonts=fonts,
        )

    tk.Frame(menu, bg="#1a2434", height=1).pack(fill="x", pady=(4, 10))

    tk.Label(menu, text="SETUP", fg=COLORS["section"], bg=COLORS["panel"], font=fonts["section"]).pack(anchor="w", pady=(0, 8))

    setup_tile = _ActionTile(
        menu,
        title=setup_label,
        subtitle=setup_hint,
        command=close_and_launch(setup_command),
        fonts=fonts,
    )

    tk.Frame(menu, bg="#1a2434", height=1).pack(fill="x", pady=(8, 10))

    tk.Label(menu, text="APP", fg=COLORS["section"], bg=COLORS["panel"], font=fonts["section"]).pack(
        anchor="w", pady=(0, 8)
    )

    _ActionTile(
        menu,
        title="Settings",
        subtitle="Camera, retention, desktop export",
        command=lambda: (show_settings_dialog(), refresh_profile_status()),
        fonts=fonts,
    )
    _ActionTile(
        menu,
        title="About",
        subtitle=f"Synapse v{VERSION}",
        command=show_about_dialog,
        fonts=fonts,
    )

    window.mainloop()
    return 0


def main() -> int:
    return run_home_shell()


if __name__ == "__main__":
    raise SystemExit(main())
