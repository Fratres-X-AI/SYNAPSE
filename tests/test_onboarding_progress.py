from pathlib import Path

import pytest

from utils.emotion_profile import EmotionProfile
from utils.onboarding_progress import (
    clear_progress,
    load_progress,
    progress_path,
    restore_profile,
    save_progress,
)
from utils.settings import UserSettings, save_settings
from utils.user_profiles import DEFAULT_USER_ID, migrate_user_profiles


def _patch_user_dirs(tmp_path, monkeypatch):
    config_root = tmp_path / "config"
    users_root = config_root / "users"
    monkeypatch.setattr("utils.app_paths.config_dir", lambda: config_root)
    monkeypatch.setattr("utils.user_profiles.config_dir", lambda: config_root)
    monkeypatch.setattr("utils.user_profiles.users_dir", lambda: users_root)
    monkeypatch.setattr("utils.user_profiles.user_dir", lambda user_id: users_root / user_id)
    monkeypatch.setattr(
        "utils.onboarding_progress.user_dir",
        lambda user_id=None: users_root / (user_id or DEFAULT_USER_ID),
    )
    save_settings(UserSettings(active_user=DEFAULT_USER_ID))
    migrate_user_profiles()
    return users_root


def test_save_load_and_clear_progress(tmp_path, monkeypatch):
    _patch_user_dirs(tmp_path, monkeypatch)
    profile = EmotionProfile()
    profile.set_phase("neutral", {"smile_score": 0.1, "cheek_raise": 0.2, "ear": 0.3})

    save_progress(
        cal_index=2,
        expr_index=1,
        captured=["neutral"],
        expr_armed=True,
        samples={"center_ear": 0.25},
        center_ears=[0.24, 0.26],
        profile=profile,
    )

    loaded = load_progress()
    assert loaded is not None
    assert loaded["cal_index"] == 2
    assert loaded["expr_index"] == 1
    assert loaded["captured"] == ["neutral"]
    assert loaded["expr_armed"] is True
    assert loaded["samples"]["center_ear"] == pytest.approx(0.25)
    assert loaded["center_ears"] == [pytest.approx(0.24), pytest.approx(0.26)]

    restored = restore_profile(loaded)
    assert restored.neutral["smile_score"] == pytest.approx(0.1)
    assert progress_path().exists()

    clear_progress()
    assert load_progress() is None
    assert not progress_path().exists()
