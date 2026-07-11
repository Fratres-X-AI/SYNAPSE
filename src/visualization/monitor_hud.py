"""Compose monitor-facing HUD banners and subtitles."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.cognition.fusion_state import FusionState
from src.perception.presence_detector import PresenceFrame
from src.perception.shoulder_tracker import ShoulderSample
from src.visualization.presence_overlay import presence_hud_note

RULE_BANNERS = {
    "low_engagement": "LOW ENGAGEMENT SUSTAINED",
    "high_distraction": "HIGH DISTRACTION SUSTAINED",
    "fatigue_spike": "FATIGUE SPIKE DETECTED",
    "high_tension": "ELEVATED TENSION SUSTAINED",
}


@dataclass
class SmokingBannerTracker:
    hold_frames: int = 150
    _frames_left: int = 0

    def tick(self, presence: PresenceFrame | None) -> str:
        if presence is not None and "smoking" in presence.active_labels():
            self._frames_left = self.hold_frames
        if self._frames_left > 0:
            self._frames_left -= 1
            return "SMOKING"
        return ""


@dataclass
class VisitorBannerTracker:
    hold_frames: int = 90
    _frames_left: int = 0

    def tick(self, presence_event: str) -> str:
        if presence_event == "visitor":
            self._frames_left = self.hold_frames
        if self._frames_left > 0:
            self._frames_left -= 1
            return "VISITOR IN FRAME"
        return ""


@dataclass
class RuleBannerTracker:
    hold_frames: int = 120
    _message: str = ""
    _frames_left: int = 0

    def update(self, new_messages: list[str]) -> None:
        if new_messages:
            self._message = new_messages[-1].upper()
            self._frames_left = self.hold_frames

    def current(self) -> str:
        if self._frames_left > 0:
            self._frames_left -= 1
            return self._message
        self._message = ""
        return ""


def banner_from_active_alerts(active_alerts: str) -> str:
    if not active_alerts:
        return ""
    first = active_alerts.split(";")[0].strip()
    return RULE_BANNERS.get(first, first.replace("_", " ").upper())


def compose_monitor_banner(
    *,
    active_alerts: str,
    presence_event: str,
    state_message: str,
    visitor_tracker: VisitorBannerTracker,
    rule_tracker: RuleBannerTracker,
    smoking_tracker: SmokingBannerTracker,
    new_rule_messages: list[str],
    presence: PresenceFrame | None = None,
) -> str:
    rule_tracker.update(new_rule_messages)
    visitor = visitor_tracker.tick(presence_event)
    if visitor:
        return visitor
    smoking = smoking_tracker.tick(presence)
    if smoking:
        return smoking
    if state_message:
        return state_message
    rule = rule_tracker.current()
    if rule:
        return rule
    return ""


def build_monitor_subtitle(
    presence: PresenceFrame | None,
    shoulder: ShoulderSample | None,
    fusion: FusionState | None,
    autonomy: float | None,
) -> str:
    del fusion, autonomy
    parts: list[str] = []
    presence_note = presence_hud_note(presence, monitor=True)
    if presence_note:
        parts.append(presence_note)
    if shoulder is not None and shoulder.visible:
        parts.append(shoulder.hud_text())
    return " | ".join(parts)
