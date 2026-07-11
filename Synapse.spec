# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = [
    ("src", "src"),
    ("utils", "utils"),
    ("docs/privacy.md", "docs"),
    ("assets/models", "assets/models"),
    ("assets/synapse.ico", "assets"),
    ("assets/synapse_icon.png", "assets"),
]
binaries = []
hiddenimports = [
    "test_onboard",
    "test_monitor",
    "test_fusion_track",
    "test_showcase",
    "replay_monitor",
    "synapse_pilot_summary",
    "synapse_home",
    "cv2",
    "numpy",
    "mediapipe",
    "PIL",
    "pystray",
    "src.adaptation.adaptive_agent",
    "src.cognition.cognitive_state",
    "src.cognition.emotion_state",
    "src.cognition.fusion_state",
    "src.cognition.profile_matcher",
    "src.cognition.soft_scores",
    "src.monitoring.alert_rules",
    "src.monitoring.presence_logger",
    "src.perception.capture",
    "src.perception.frame_quality",
    "src.perception.face_geometry",
    "src.perception.presence_detector",
    "src.perception.shoulder_tracker",
    "src.perception.state_estimator",
    "src.visualization.alerts",
    "src.visualization.debrief",
    "src.visualization.monitor_hud",
    "src.visualization.hud_text",
    "src.visualization.instrument_layout",
    "src.visualization.instrument_theme",
    "src.visualization.landmark_overlay",
    "src.visualization.presence_overlay",
    "src.visualization.display_adapter",
    "src.visualization.display",
    "src.visualization.timeline",
    "utils.app_paths",
    "utils.calibration",
    "utils.config",
    "utils.emotion_profile",
    "utils.fusion_replay",
    "utils.fusion_summary",
    "utils.fps_tracker",
    "utils.presence_model",
    "utils.manager_report",
    "utils.pilot_report",
    "utils.privacy",
    "utils.settings",
]

tmp_ret = collect_all("mediapipe")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

a = Analysis(
    ["synapse_launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="Synapse",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/synapse.ico",
)
