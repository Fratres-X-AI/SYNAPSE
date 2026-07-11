"""Save and resume partial onboarding for the active user profile."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from utils.emotion_profile import EmotionProfile
from utils.user_profiles import get_active_user_id, user_dir

PROGRESS_FILE = "onboarding_progress.json"


def progress_path(user_id: str | None = None) -> Path:
    return user_dir(user_id or get_active_user_id()) / PROGRESS_FILE


def load_progress() -> dict | None:
    path = progress_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_progress(
    *,
    cal_index: int,
    expr_index: int,
    captured: list[str],
    expr_armed: bool,
    samples: dict[str, float] | None = None,
    center_ears: list[float] | None = None,
    profile: EmotionProfile | None = None,
) -> None:
    path = progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cal_index": cal_index,
        "expr_index": expr_index,
        "captured": captured,
        "expr_armed": expr_armed,
        "samples": samples or {},
        "center_ears": center_ears or [],
        "profile": asdict(profile) if profile is not None else {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_progress() -> None:
    path = progress_path()
    if path.exists():
        path.unlink()


def restore_profile(data: dict) -> EmotionProfile:
    profile_data = data.get("profile") or {}
    return EmotionProfile(
        neutral=profile_data.get("neutral", {}),
        happy=profile_data.get("happy", {}),
        sad=profile_data.get("sad", {}),
        mad=profile_data.get("mad", {}),
    )
