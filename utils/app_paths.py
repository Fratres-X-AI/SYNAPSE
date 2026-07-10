"""Local app data locations for Synapse.

Synapse is local-first: webcam processing happens on-device and user data is
stored in a per-user app directory instead of the source tree.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

APP_NAME = "Synapse"


def app_data_dir() -> Path:
    base = os.getenv("LOCALAPPDATA")
    if base:
        return Path(base) / APP_NAME
    return Path.home() / ".synapse"


def config_dir() -> Path:
    return app_data_dir() / "config"


def session_dir() -> Path:
    return app_data_dir() / "sessions"


def reports_dir() -> Path:
    return app_data_dir() / "reports"


def ensure_app_dirs() -> None:
    for path in (config_dir(), session_dir(), reports_dir()):
        path.mkdir(parents=True, exist_ok=True)


def calibration_path() -> Path:
    return config_dir() / "calibration.json"


def emotion_profile_path() -> Path:
    return config_dir() / "emotion_profile.json"


def settings_path() -> Path:
    return config_dir() / "settings.json"


def consent_path() -> Path:
    return config_dir() / "privacy_consent.json"


def migrate_legacy_file(legacy_path: Path, target_path: Path) -> bool:
    """Copy old repo-root data into app data once, without deleting the source."""
    if target_path.exists() or not legacy_path.exists():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_path, target_path)
    return True


def migrate_legacy_data(root: Path | None = None) -> list[str]:
    root = root or Path.cwd()
    moved: list[str] = []
    for legacy, target in (
        (root / "calibration.json", calibration_path()),
        (root / "emotion_profile.json", emotion_profile_path()),
    ):
        if migrate_legacy_file(legacy, target):
            moved.append(f"{legacy.name} -> {target}")

    legacy_sessions = root / "sessions"
    if legacy_sessions.exists():
        session_dir().mkdir(parents=True, exist_ok=True)
        for file_path in legacy_sessions.glob("*"):
            if file_path.is_file():
                target = session_dir() / file_path.name
                if not target.exists():
                    shutil.copy2(file_path, target)
                    moved.append(f"{file_path.name} -> {target}")
    return moved


def delete_all_user_data() -> None:
    root = app_data_dir()
    if root.exists():
        shutil.rmtree(root)


def data_inventory() -> dict[str, str | int]:
    ensure_app_dirs()
    migrate_legacy_data()
    sessions = list(session_dir().glob("*"))
    reports = list(reports_dir().glob("*"))
    return {
        "app_data_dir": str(app_data_dir()),
        "calibration_exists": int(calibration_path().exists()),
        "emotion_profile_exists": int(emotion_profile_path().exists()),
        "session_files": len([path for path in sessions if path.is_file()]),
        "report_files": len([path for path in reports if path.is_file()]),
    }


def cleanup_old_data(retention_days: int) -> list[Path]:
    if retention_days <= 0:
        return []
    cutoff = datetime.now() - timedelta(days=retention_days)
    deleted: list[Path] = []
    for root in (session_dir(), reports_dir()):
        if not root.exists():
            continue
        for path in root.glob("*"):
            if not path.is_file():
                continue
            modified = datetime.fromtimestamp(path.stat().st_mtime)
            if modified < cutoff:
                path.unlink()
                deleted.append(path)
    return deleted
