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
    new_rule_messages: list[str],
) -> str:
    rule_tracker.update(new_rule_messages)
    visitor = visitor_tracker.tick(presence_event)
    if visitor:
        return visitor
    if state_message:
        return state_message
    rule = rule_tracker.current()
    if rule:
        return rule
    return banner_from_active_alerts(active_alerts)


def build_monitor_subtitle(
    presence: PresenceFrame | None,
    shoulder: ShoulderSample | None,
    fusion: FusionState | None,
    autonomy: float | None,
) -> str:
    parts: list[str] = []
    if fusion is not None:
        phase = fusion.profile_phase or "—"
        parts.append(f"PHASE {phase.upper()}")
    presence_note = presence_hud_note(presence)
    if presence_note:
        parts.append(presence_note)
    if shoulder is not None:
        parts.append(shoulder.hud_text())
    if autonomy is not None:
        parts.append(f"AUT {autonomy:.0%}")
    return " | ".join(parts)
