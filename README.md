# Synapse

Synapse is a webcam-first, local-first focus and fatigue support tool. It uses the local webcam to extract face landmarks on-device, estimates coarse attention signals in real time, and saves session summaries for review.

Synapse is pilot software. It is not a medical device, lie detector, emotion oracle, productivity scoring system, or employee discipline tool.

## What It Does

- Runs webcam processing locally with MediaPipe and OpenCV.
- Guides each user through onboarding: attention calibration plus an optional expression profile.
- Monitors live sessions for engagement, fatigue, tension, positivity, distraction, and broad attention state.
- Saves local CSV logs, alert logs, and text reports for review.
- Replays saved monitor sessions for debriefing and quality checks.

Raw webcam video is not saved by Synapse. See `docs/privacy.md` before any pilot use.

## Setup

Use Python 3.11+ on Windows with a working webcam.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional tray and packaging dependencies:

```powershell
pip install -r requirements-dev.txt
```

## Launcher Commands

Run commands from the repository root:

```powershell
python synapse_launcher.py onboard
python synapse_launcher.py monitor
python synapse_launcher.py monitor --fullscreen
python synapse_launcher.py replay "%LOCALAPPDATA%\Synapse\sessions\monitor_YYYYMMDD_HHMMSS.csv"
python synapse_launcher.py --tray
```

`onboard` records privacy consent, then captures calibration values and an expression profile. `monitor` starts a live webcam session and writes logs/reports. `replay` opens a saved session CSV; pass the CSV path explicitly for app-data sessions. Press `q` to quit a camera window and `f` to toggle fullscreen where supported.

## Local Data

On Windows, Synapse stores user data under:

```text
%LOCALAPPDATA%\Synapse
```

Expected contents:

- `config\privacy_consent.json`
- `config\calibration.json`
- `config\emotion_profile.json`
- `config\settings.json`
- `sessions\monitor_*.csv`
- `sessions\monitor_*.alerts.csv`
- `sessions\monitor_*.report.txt`

By default, monitor reports may also be copied to the Desktop as `Synapse_Report_*.txt`.

## Pilot Workflow

Use the no-payment pilot process in `docs/pilot_guide.md` until the software quality gate is met.

1. Explain consent and limitations.
2. Run `python synapse_launcher.py onboard`.
3. Run a short camera/environment check.
4. Run `python synapse_launcher.py monitor`.
5. Stop with `q`, review the generated report, and replay the session if needed.

## Build

For a smoke-test Windows executable, install dev dependencies and run the checked-in build script:

```powershell
pip install -r requirements-dev.txt
.\build.ps1
```

Verify the build with `docs/release_checklist.md` before sharing it with pilot users.

## Documentation

- `docs/privacy.md` - privacy model, saved data, deletion/export controls, consent language.
- `docs/pilot_guide.md` - 5-person no-payment pilot workflow.
- `docs/release_checklist.md` - Windows release verification checklist.
- `docs/known_issues.md` - current limitations and webcam constraints.
- `docs/positioning.md` - approved positioning and prohibited claims.
- `docs/monetization_gate.md` - no-billing rule until the quality gate is proven.
