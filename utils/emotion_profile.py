import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from utils.app_paths import emotion_profile_path

LEGACY_EMOTION_PROFILE_PATH = Path("emotion_profile.json")
ACTIVE_PHASES = ("neutral", "happy", "sad", "mad")

PROFILE_KEYS = (
    "smile_score",
    "cheek_raise",
    "brow_raise",
    "brow_furrow",
    "brow_inner_pinch",
    "mouth_open",
    "ear",
)

PHASE_LABELS = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad/stressed",
    "mad": "mad",
}


@dataclass
class EmotionProfile:
    neutral: dict[str, float] = field(default_factory=dict)
    happy: dict[str, float] = field(default_factory=dict)
    sad: dict[str, float] = field(default_factory=dict)
    mad: dict[str, float] = field(default_factory=dict)

    def set_phase(self, phase: str, snapshot: dict[str, float]) -> None:
        trimmed = {key: float(snapshot[key]) for key in PROFILE_KEYS if key in snapshot}
        setattr(self, phase, trimmed)

    def has_neutral(self) -> bool:
        return bool(self.neutral)

    def has_sad_reference(self) -> bool:
        return bool(self.sad)


def save_emotion_profile(
    profile: EmotionProfile,
    path: Path | None = None,
) -> None:
    if path is None:
        path = emotion_profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")


def load_emotion_profile(
    path: Path | None = None,
) -> EmotionProfile | None:
    if path is None:
        path = emotion_profile_path()
    if not path.exists():
        if LEGACY_EMOTION_PROFILE_PATH.exists():
            path = LEGACY_EMOTION_PROFILE_PATH
        else:
            return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return EmotionProfile(
        neutral=data.get("neutral", {}),
        happy=data.get("happy", {}),
        sad=data.get("sad", {}),
        mad=data.get("mad", {}),
    )
