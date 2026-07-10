from dataclasses import dataclass
from pathlib import Path

from utils.calibration import DEFAULT_CALIBRATION_PATH, load_calibration


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    window_name: str = "Synapse Cognitive State"
    display_mode: str = "opencv"
    calibration_path: Path = DEFAULT_CALIBRATION_PATH

    def estimator_kwargs(self) -> dict:
        profile = load_calibration(self.calibration_path)
        if profile is None:
            return {}
        return profile.to_estimator_kwargs()
