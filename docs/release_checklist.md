# Synapse Windows Release Checklist

Use this checklist before sharing a build with pilot users. Do not start billing until every required item passes and known blockers are accepted.

## Build Verification

- [ ] Create a fresh virtual environment.
- [ ] Install `requirements.txt`.
- [ ] Install `requirements-dev.txt` if building an executable.
- [ ] Run `.\scripts\check_python.ps1` (or confirm Python 3.11–3.12).
- [ ] Run `python synapse_launcher.py --help`.
- [ ] Build with PyInstaller if shipping an executable:

```powershell
.\build.ps1
.\scripts\verify_release.ps1
```

- [ ] Confirm `docs/windows_install.md` matches the shipped executable behavior.

- [ ] Launch the built executable from `dist`.
- [ ] Confirm missing optional tray dependencies produce a clear message if tray support is unavailable.

## Clean Machine Install

- [ ] Test on a Windows machine or VM without repo-root calibration/session files.
- [ ] Confirm the webcam is available and not locked by another app.
- [ ] Confirm `%LOCALAPPDATA%\Synapse` is created only after Synapse starts.
- [ ] Confirm no raw webcam video or still images are written.

## Onboarding

- [ ] Run `python synapse_launcher.py onboard`.
- [ ] Confirm privacy consent appears before webcam capture.
- [ ] Reject consent once and confirm capture does not start.
- [ ] Accept consent and complete all calibration steps.
- [ ] Confirm `calibration.json` and `emotion_profile.json` are saved under `%LOCALAPPDATA%\Synapse\config`.
- [ ] Repeat with poor lighting and confirm the user sees useful quality guidance.

## Monitor

- [ ] Run `python synapse_launcher.py monitor`.
- [ ] Confirm the monitor window opens and shows live status.
- [ ] Confirm `q` exits cleanly.
- [ ] Confirm `f` toggles fullscreen where supported.
- [ ] Confirm session CSV and alert CSV files are saved under `%LOCALAPPDATA%\Synapse\sessions`.
- [ ] Confirm a report is generated when enough samples are recorded.
- [ ] Confirm desktop report export behavior matches settings.

## Replay

- [ ] Replay with an explicit session CSV path.
- [ ] Confirm the replay window opens and exits with `q`.
- [ ] Confirm report generation during replay does not create duplicate desktop exports.
- [ ] Record whether default latest-session replay works for the build.

## Report Quality

- [ ] Confirm report fields are understandable to a non-engineer.
- [ ] Confirm report text includes no medical, diagnostic, productivity, or discipline claims.
- [ ] Confirm alert flags are phrased as possibilities, not conclusions.
- [ ] Confirm exported filenames contain no participant names by default.

## 30-Minute Soak Test

- [ ] Run monitor continuously for 30 minutes.
- [ ] Keep normal lighting and webcam position stable.
- [ ] Confirm CPU usage and window responsiveness remain acceptable.
- [ ] Confirm CSV rows continue writing throughout the session.
- [ ] Confirm no crash on exit.
- [ ] Confirm final report is generated and readable.

## Release Decision

- [ ] All required checks pass.
- [ ] Known issues in `docs/known_issues.md` are reviewed.
- [ ] Pilot consent language in `docs/privacy.md` is approved.
- [ ] No billing, paid pilot, or commercial claim is enabled before the quality gate.
