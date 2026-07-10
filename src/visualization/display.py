import cv2

from src.cognition.cognitive_state import CognitiveState, State
from src.visualization.hud_text import HUD_ACCENT, draw_hud_text
from src.visualization.instrument_theme import STATE_COLORS, draw_annunciator_strip, draw_waiting_state


def render_status(frame, cognitive_state: CognitiveState | None, behavior: str | None):
    if cognitive_state is None:
        draw_waiting_state(frame)
        return frame

    draw_annunciator_strip(frame, cognitive_state.state)
    signals = cognitive_state.signals
    blink_label = "BLINK" if signals["is_blinking"] else "OPEN"
    lines = [
        f"STATE {cognitive_state.state.value.upper()}  ({cognitive_state.confidence:.0%})",
        (
            f"EAR {signals['ear']:.3f}  MEAN {signals['mean_ear']:.3f}  {blink_label}  "
            f"CNT {signals['blink_counter']}  RATE {signals['blink_rate']:.1f}/m"
        ),
        (
            f"HDG Y{signals['head_yaw']:+.1f} P{signals['head_pitch']:+.1f}  "
            f"GAZE {signals['gaze_direction']} ({signals['gaze_x']:+.2f},{signals['gaze_y']:+.2f})"
        ),
        behavior or "",
    ]
    y = 42
    for line in lines:
        if not line:
            continue
        color = STATE_COLORS.get(cognitive_state.state, HUD_ACCENT) if y == 42 else None
        draw_hud_text(frame, line, (16, y), size=12, color=color or (252, 253, 255))
        y += 16

    return frame