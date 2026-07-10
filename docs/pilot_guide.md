# Synapse 5-Person Pilot Guide

This pilot is for product quality learning only. Do not charge participants, imply workplace evaluation, or use Synapse outputs for decisions about a person.

## Pilot Goals

- Confirm onboarding is understandable.
- Confirm webcam monitoring works across common Windows setups.
- Find reliability, lighting, camera, and report issues.
- Learn whether the summary report is useful after a real session.

## Before Each Session

1. Install dependencies or use a verified Windows build.
2. Confirm the webcam works in another app.
3. Close apps that may lock the camera.
4. Read the consent language in `docs/privacy.md`.
5. Explain that the participant can stop at any time with `q`.

## Onboarding

Run:

```powershell
python synapse_launcher.py onboard
```

The participant will complete:

- Attention calibration: center gaze, natural blinking, head turns, screen corners, and looking away.
- Expression profile: neutral, happy, sad/stressed, and mad/frustrated snapshots.

Repeat onboarding if face detection fails, lighting is poor, the participant changes glasses/headwear, or the camera position changes significantly.

## Monitor Session

Run:

```powershell
python synapse_launcher.py monitor
```

Recommended pilot session length: 10-30 minutes. Ask the participant to work normally. Avoid coaching them to "perform" for the metrics.

Stop with `q`. Synapse writes session CSV files, alert logs, and a report under `%LOCALAPPDATA%\Synapse\sessions`. A desktop report copy may also be created.

## Replay And Review

Replay a monitor session with an explicit CSV path:

```powershell
python synapse_launcher.py replay "%LOCALAPPDATA%\Synapse\sessions\monitor_YYYYMMDD_HHMMSS.csv"
```

Use replay to check whether face detection, lighting, gaze, and alert timing looked reasonable. Do not replay in front of others unless the participant consents.

## Summary Exports

For each participant, keep only the files needed for product review:

- `monitor_*.report.txt`
- `monitor_*.alerts.csv`
- `monitor_*.csv` only when debugging metric behavior

Remove names or participant identifiers from filenames before sharing outside the local test team.

## Interpreting Metrics

Use metrics as noisy indicators:

- Engagement: signal mix suggesting on-screen attention.
- Fatigue: blink and eye openness patterns.
- Tension: brow/squint pattern estimate.
- Positivity: smile/cheek pattern estimate.
- Distraction: gaze/head direction away from screen.
- Dominant state: the most frequent coarse state during the session.

Look for obvious mismatches between the report and participant feedback. Those mismatches are valuable pilot findings.

## What Not To Infer

Do not infer medical conditions, mental health, intent, honesty, competence, productivity, emotions as facts, or job performance. Do not compare participants against each other. Do not use pilot outputs for hiring, discipline, grading, compensation, or access decisions.

## End Of Session Questions

- Was onboarding clear?
- Did the camera view feel acceptable?
- Did Synapse interrupt or distract you?
- Did the summary match your experience at a broad level?
- What felt inaccurate, invasive, or confusing?
- Would you consent to another session after fixes?
