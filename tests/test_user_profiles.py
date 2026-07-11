from pathlib import Path

import pytest

from utils.settings import UserSettings, load_settings, save_settings
from utils.user_profiles import (
    DEFAULT_USER_ID,
    create_user_profile,
    get_active_user_id,
    get_user_profile,
    list_user_profiles,
    migrate_user_profiles,
    set_active_user,
    user_calibration_path,
    user_emotion_profile_path,
    user_is_configured,
)


def test_migrate_moves_flat_calibration_into_default_user(tmp_path, monkeypatch):
    config_root = tmp_path / "config"
    users_root = config_root / "users"
    monkeypatch.setattr("utils.app_paths.config_dir", lambda: config_root)
    monkeypatch.setattr("utils.user_profiles.config_dir", lambda: config_root)
    monkeypatch.setattr("utils.user_profiles.users_dir", lambda: users_root)
    monkeypatch.setattr("utils.user_profiles.user_dir", lambda user_id: users_root / user_id)
    monkeypatch.setattr(
        "utils.user_profiles.user_calibration_path",
        lambda user_id=None: (users_root / (user_id or DEFAULT_USER_ID)) / "calibration.json",
    )
    monkeypatch.setattr(
        "utils.user_profiles.user_emotion_profile_path",
        lambda user_id=None: (users_root / (user_id or DEFAULT_USER_ID)) / "emotion_profile.json",
    )

    config_root.mkdir(parents=True)
    (config_root / "calibration.json").write_text("{}", encoding="utf-8")
    save_settings(UserSettings(active_user=DEFAULT_USER_ID))

    moved = migrate_user_profiles()

    assert moved
    assert user_calibration_path(DEFAULT_USER_ID).exists()
    assert not (config_root / "calibration.json").exists()


def test_create_and_switch_user_profiles(tmp_path, monkeypatch):
    config_root = tmp_path / "config"
    users_root = config_root / "users"
    monkeypatch.setattr("utils.app_paths.config_dir", lambda: config_root)
    monkeypatch.setattr("utils.user_profiles.config_dir", lambda: config_root)
    monkeypatch.setattr("utils.user_profiles.users_dir", lambda: users_root)
    monkeypatch.setattr("utils.user_profiles.user_dir", lambda user_id: users_root / user_id)
    monkeypatch.setattr(
        "utils.user_profiles.user_calibration_path",
        lambda user_id=None: (users_root / (user_id or DEFAULT_USER_ID)) / "calibration.json",
    )
    monkeypatch.setattr(
        "utils.user_profiles.user_emotion_profile_path",
        lambda user_id=None: (users_root / (user_id or DEFAULT_USER_ID)) / "emotion_profile.json",
    )
    save_settings(UserSettings(active_user=DEFAULT_USER_ID))
    migrate_user_profiles()

    alice = create_user_profile("Alice")
    bob = create_user_profile("Bob")

    assert get_active_user_id() == bob.user_id
    assert alice.user_id != bob.user_id
    assert len(list_user_profiles()) == 3

    set_active_user(alice.user_id)
    assert get_active_user_id() == alice.user_id
    assert get_user_profile().display_name == "Alice"
    assert not user_is_configured(alice.user_id)


def test_user_is_configured_requires_both_files(tmp_path, monkeypatch):
    config_root = tmp_path / "config"
    users_root = config_root / "users"
    monkeypatch.setattr("utils.app_paths.config_dir", lambda: config_root)
    monkeypatch.setattr("utils.user_profiles.config_dir", lambda: config_root)
    monkeypatch.setattr("utils.user_profiles.users_dir", lambda: users_root)
    monkeypatch.setattr("utils.user_profiles.user_dir", lambda user_id: users_root / user_id)
    monkeypatch.setattr(
        "utils.user_profiles.user_calibration_path",
        lambda user_id=None: (users_root / (user_id or DEFAULT_USER_ID)) / "calibration.json",
    )
    monkeypatch.setattr(
        "utils.user_profiles.user_emotion_profile_path",
        lambda user_id=None: (users_root / (user_id or DEFAULT_USER_ID)) / "emotion_profile.json",
    )
    save_settings(UserSettings(active_user=DEFAULT_USER_ID))
    migrate_user_profiles()

    user_calibration_path(DEFAULT_USER_ID).parent.mkdir(parents=True, exist_ok=True)
    user_calibration_path(DEFAULT_USER_ID).write_text("{}", encoding="utf-8")
    assert not user_is_configured(DEFAULT_USER_ID)

    user_emotion_profile_path(DEFAULT_USER_ID).write_text("{}", encoding="utf-8")
    assert user_is_configured(DEFAULT_USER_ID)
