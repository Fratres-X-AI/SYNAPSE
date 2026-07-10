"""User settings for the local Synapse desktop app."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.app_paths import ensure_app_dirs, settings_path


@dataclass(frozen=True)
class UserSettings:
    camera_index: int = 0
    fullscreen_default: bool = False
    retention_days: int = 30
    export_reports_to_desktop: bool = True
    privacy_mode: bool = True


def load_settings(path: Path = settings_path()) -> UserSettings:
    ensure_app_dirs()
    if not path.exists():
        return UserSettings()
    data = json.loads(path.read_text(encoding="utf-8"))
    defaults = asdict(UserSettings())
    return UserSettings(**{**defaults, **data})


def save_settings(settings: UserSettings, path: Path = settings_path()) -> None:
    ensure_app_dirs()
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
