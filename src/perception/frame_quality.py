"""Live webcam quality heuristics for in-session coaching."""

from __future__ import annotations

import cv2


def assess_frame_quality(frame, *, face_detected: bool) -> tuple[str, tuple[int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    if not face_detected:
        return "FACE NOT DETECTED", (0, 140, 255)
    if brightness < 65:
        return "LIGHTING LOW", (0, 140, 255)
    if brightness > 215:
        return "LIGHTING BRIGHT", (0, 140, 255)
    if contrast < 22:
        return "IMAGE FLAT", (0, 140, 255)
    return "QUALITY OK", (80, 220, 100)


def draw_quality_pill(frame, message: str, color: tuple[int, int, int]) -> None:
    from src.visualization.hud_text import draw_hud_text

    draw_hud_text(frame, message, (12, frame.shape[0] - 36), size=10, color=color, label=True)
