"""Resolve the local desk-object model used for presence detection."""

from __future__ import annotations

import shutil
import sys
import urllib.request
import os
from pathlib import Path

from utils.app_paths import app_data_dir

MODEL_NAME = "efficientdet_lite0.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
)


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def bundled_model_path() -> Path:
    return resource_root() / "assets" / "models" / MODEL_NAME


def cached_model_path() -> Path:
    return app_data_dir() / "models" / MODEL_NAME


def ensure_object_model() -> Path:
    """Return a stable absolute model path for copying and caching."""
    cached = cached_model_path()
    if cached.exists():
        return cached.resolve()

    cached.parent.mkdir(parents=True, exist_ok=True)
    bundled = bundled_model_path()
    if bundled.exists():
        shutil.copy2(bundled, cached)
        return cached.resolve()

    urllib.request.urlretrieve(MODEL_URL, cached)
    return cached.resolve()


def object_model_bytes() -> bytes:
    return ensure_object_model().read_bytes()
