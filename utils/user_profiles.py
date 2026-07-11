"""Per-user calibration and emotion profile storage."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from utils.app_paths import config_dir, ensure_app_dirs
from utils.settings import UserSettings, load_settings, save_settings

DEFAULT_USER_ID = "default"
DEFAULT_USER_NAME = "Default"
USER_META_FILE = "user.json"


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    display_name: str
    calibration_path: Path
    emotion_profile_path: Path
    last_session_at: str | None = None
    last_session_name: str | None = None

    @property
    def is_configured(self) -> bool:
        return self.calibration_path.exists() and self.emotion_profile_path.exists()

    @property
    def status_line(self) -> str:
        if not self.is_configured:
            return "Setup needed"
        if self.last_session_at:
            return f"Last session {self.last_session_at}"
        return "Ready"


def users_dir() -> Path:
    return config_dir() / "users"


def user_dir(user_id: str) -> Path:
    return users_dir() / user_id


def sanitize_user_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug[:32] or DEFAULT_USER_ID


def unique_user_id(display_name: str) -> str:
    base = sanitize_user_id(display_name)
    candidate = base
    suffix = 2
    while user_dir(candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _read_user_meta(user_id: str) -> dict:
    path = user_dir(user_id) / USER_META_FILE
    if not path.exists():
        return {"display_name": user_id.replace("-", " ").title()}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_user_meta(user_id: str, display_name: str) -> None:
    directory = user_dir(user_id)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_id": user_id,
        "display_name": display_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (directory / USER_META_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def user_calibration_path(user_id: str | None = None) -> Path:
    user_id = user_id or get_active_user_id()
    return user_dir(user_id) / "calibration.json"


def user_emotion_profile_path(user_id: str | None = None) -> Path:
    user_id = user_id or get_active_user_id()
    return user_dir(user_id) / "emotion_profile.json"


def get_active_user_id() -> str:
    settings = load_settings()
    if settings.active_user and user_dir(settings.active_user).exists():
        return settings.active_user
    return DEFAULT_USER_ID


def get_active_user_display_name() -> str:
    return get_user_profile(get_active_user_id()).display_name


def get_user_profile(user_id: str | None = None) -> UserProfile:
    user_id = user_id or get_active_user_id()
    meta = _read_user_meta(user_id)
    display_name = str(meta.get("display_name", user_id.replace("-", " ").title()))
    return UserProfile(
        user_id=user_id,
        display_name=display_name,
        calibration_path=user_calibration_path(user_id),
        emotion_profile_path=user_emotion_profile_path(user_id),
        last_session_at=meta.get("last_session_at"),
        last_session_name=meta.get("last_session_name"),
    )


def list_user_profiles() -> list[UserProfile]:
    ensure_app_dirs()
    migrate_user_profiles()
    profiles: list[UserProfile] = []
    root = users_dir()
    if not root.exists():
        return [get_user_profile(DEFAULT_USER_ID)]
    for path in sorted(root.iterdir()):
        if path.is_dir():
            profiles.append(get_user_profile(path.name))
    if not profiles:
        profiles.append(get_user_profile(DEFAULT_USER_ID))
    return profiles


def set_active_user(user_id: str) -> UserProfile:
    directory = user_dir(user_id)
    if not directory.exists():
        raise ValueError(f"Unknown user profile: {user_id}")
    settings = load_settings()
    updated = UserSettings(
        camera_index=settings.camera_index,
        fullscreen_default=settings.fullscreen_default,
        retention_days=settings.retention_days,
        export_reports_to_desktop=settings.export_reports_to_desktop,
        privacy_mode=settings.privacy_mode,
        active_user=user_id,
    )
    save_settings(updated)
    return get_user_profile(user_id)


def create_user_profile(display_name: str) -> UserProfile:
    name = display_name.strip()
    if not name:
        raise ValueError("Profile name cannot be empty.")
    user_id = unique_user_id(name)
    _write_user_meta(user_id, name)
    user_dir(user_id).mkdir(parents=True, exist_ok=True)
    return set_active_user(user_id)


def user_is_configured(user_id: str | None = None) -> bool:
    return get_user_profile(user_id).is_configured


def record_session(session_name: str, *, duration_seconds: float, user_id: str | None = None) -> None:
    user_id = user_id or get_active_user_id()
    meta = _read_user_meta(user_id)
    meta["last_session_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta["last_session_name"] = session_name
    meta["last_session_seconds"] = round(duration_seconds)
    (user_dir(user_id) / USER_META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def migrate_user_profiles() -> list[str]:
    ensure_app_dirs()
    users_root = users_dir()
    users_root.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []
    default_dir = user_dir(DEFAULT_USER_ID)
    if not default_dir.exists():
        default_dir.mkdir(parents=True, exist_ok=True)
        _write_user_meta(DEFAULT_USER_ID, DEFAULT_USER_NAME)

    settings = load_settings()
    if not settings.active_user:
        updated = UserSettings(
            camera_index=settings.camera_index,
            fullscreen_default=settings.fullscreen_default,
            retention_days=settings.retention_days,
            export_reports_to_desktop=settings.export_reports_to_desktop,
            privacy_mode=settings.privacy_mode,
            active_user=DEFAULT_USER_ID,
        )
        save_settings(updated)

    for legacy_path, target_path in (
        (config_dir() / "calibration.json", user_calibration_path(DEFAULT_USER_ID)),
        (config_dir() / "emotion_profile.json", user_emotion_profile_path(DEFAULT_USER_ID)),
    ):
        if legacy_path.exists() and not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(legacy_path), str(target_path))
            moved.append(f"{legacy_path.name} -> {target_path}")

    return moved
