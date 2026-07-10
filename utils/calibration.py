import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_CALIBRATION_PATH = Path("calibration.json")


@dataclass
class CalibrationProfile:
    ear_blink_threshold: float = 0.21
    ear_blink_open_threshold: float | None = None
    low_ear_threshold: float = 0.24
    distracted_yaw_threshold: float = 25.0
    distracted_pitch_threshold: float = 20.0
    gaze_label_threshold: float = 0.12
    distracted_gaze_threshold: float = 0.34
    distracted_gaze_up_threshold: float = 0.22
    gaze_meter_threshold: float = 0.30
    distracted_pitch_up_threshold: float = 14.0

    def to_estimator_kwargs(self) -> dict:
        return asdict(self)


def save_calibration(profile: CalibrationProfile, path: Path = DEFAULT_CALIBRATION_PATH) -> None:
    path.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")


def load_calibration(path: Path = DEFAULT_CALIBRATION_PATH) -> CalibrationProfile | None:
    if not path.exists():
        return None

    defaults = {field.name: field.default for field in fields(CalibrationProfile)}
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = {**defaults, **data}
    merged["distracted_gaze_threshold"] = min(float(merged["distracted_gaze_threshold"]), 0.38)
    merged["distracted_yaw_threshold"] = min(float(merged["distracted_yaw_threshold"]), 40.0)
    return CalibrationProfile(**merged)


def build_profile(samples: dict) -> CalibrationProfile:
    baseline_ear = max(samples["center_ear"], 0.15)
    blink_ear = min(samples["blink_ear"], baseline_ear * 0.9)
    corner_gaze = max(samples["corner_gaze"], 0.12)
    away_gaze = max(samples["away_gaze"], corner_gaze * 1.2)
    gaze_up = max(samples["gaze_up"], 0.15)

    return CalibrationProfile(
        ear_blink_threshold=round(max(blink_ear * 1.05, baseline_ear * 0.72), 3),
        ear_blink_open_threshold=round(max(baseline_ear * 0.9, baseline_ear - 0.02), 3),
        low_ear_threshold=round(baseline_ear * 0.82, 3),
        distracted_yaw_threshold=round(min(max(samples["head_yaw"] * 1.15, 22.0), 40.0), 1),
        distracted_pitch_threshold=round(max(samples["head_pitch"] * 1.15, 18.0), 1),
        gaze_label_threshold=round(min(0.12, corner_gaze * 0.45), 3),
        gaze_meter_threshold=round(max(corner_gaze * 1.2, 0.25), 3),
        distracted_gaze_threshold=round(min(max(away_gaze * 0.9, corner_gaze * 1.35, 0.32), 0.38), 3),
        distracted_gaze_up_threshold=round(max(gaze_up * 0.82, 0.18), 3),
        distracted_pitch_up_threshold=round(max(samples["head_pitch_up"] * 0.9, 12.0), 1),
    )
