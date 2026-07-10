from dataclasses import dataclass
from pathlib import Path

from utils.app_paths import ensure_app_dirs, migrate_legacy_data, session_dir
from utils.calibration import DEFAULT_CALIBRATION_PATH, load_calibration
from utils.emotion_profile import DEFAULT_EMOTION_PROFILE_PATH
from utils.settings import load_settings


@dataclass(frozen=True)
class Config:
    camera_index: int | None = None
    window_name: str = "Synapse Cognitive State"
    display_mode: str = "opencv"
    fullscreen: bool | None = None
    calibration_path: Path = DEFAULT_CALIBRATION_PATH
    emotion_profile_path: Path = DEFAULT_EMOTION_PROFILE_PATH
    session_dir: Path = session_dir()

    def __post_init__(self) -> None:
        ensure_app_dirs()
        migrate_legacy_data()
        settings = load_settings()
        if self.camera_index is None:
            object.__setattr__(self, "camera_index", settings.camera_index)
        if self.fullscreen is None:
            object.__setattr__(self, "fullscreen", settings.fullscreen_default)

    def estimator_kwargs(self) -> dict:
        profile = load_calibration(self.calibration_path)
        if profile is None:
            return {}
        return profile.to_estimator_kwargs()
