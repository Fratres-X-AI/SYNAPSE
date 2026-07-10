"""Consent and data-management helpers for local-first Synapse."""

from __future__ import annotations

import json
from datetime import datetime

from utils.app_paths import consent_path, ensure_app_dirs

CONSENT_VERSION = "2026-07-local-webcam-v1"

CONSENT_TEXT = """\
SYNAPSE PRIVACY NOTICE

Synapse uses your webcam locally to estimate focus, fatigue, distraction, and
optional expression-pattern scores. Raw video frames are not saved by Synapse.

Saved local data may include calibration values, expression profile values,
session CSV files, alert logs, and text reports. These files stay on this
Windows user account unless you export or share them.

Synapse is not a medical device, lie detector, emotion oracle, or employee
discipline tool. Use it as a focus and fatigue support tool.
"""


def has_privacy_consent() -> bool:
    path = consent_path()
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("version") == CONSENT_VERSION and data.get("accepted") is True


def record_privacy_consent() -> None:
    ensure_app_dirs()
    payload = {
        "version": CONSENT_VERSION,
        "accepted": True,
        "accepted_at": datetime.now().isoformat(timespec="seconds"),
    }
    consent_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_privacy_consent(auto_accept: bool = False) -> bool:
    if has_privacy_consent():
        return True

    print(CONSENT_TEXT)
    if auto_accept:
        record_privacy_consent()
        return True

    response = input("Type YES to accept and continue: ").strip()
    if response == "YES":
        record_privacy_consent()
        return True

    print("Consent not accepted. Synapse will exit without starting capture.")
    return False
