from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    window_name: str = "Synapse Cognitive State"
