# Synapse Privacy Notes

Synapse is designed for local webcam processing during the current pilot. Webcam frames are used on the local Windows machine to detect facial landmarks and estimate coarse focus/fatigue signals.

## What Synapse Does Not Save

- Synapse does not intentionally save raw webcam video.
- Synapse does not intentionally save still images from the webcam.
- Synapse does not upload webcam frames to a cloud service.
- Synapse does not process payments or billing data.

## What Synapse Saves

Synapse saves local files under `%LOCALAPPDATA%\Synapse`:

- `config\privacy_consent.json` records that consent was accepted.
- `config\calibration.json` stores per-user attention calibration values.
- `config\emotion_profile.json` stores expression profile values captured during onboarding.
- `config\settings.json` stores local settings such as retention and export preferences.
- `sessions\monitor_*.csv` stores timestamped session metrics.
- `sessions\monitor_*.alerts.csv` stores alert rule events.
- `sessions\monitor_*.report.txt` stores text summaries.

Monitor reports may also be copied to the Desktop as `Synapse_Report_*.txt` when desktop export is enabled.

## Delete And Export Controls

To inspect local data:

```powershell
python synapse_launcher.py data
```

To delete local Synapse data:

```powershell
python synapse_launcher.py delete-data
```

Also remove any exported `Synapse_Report_*.txt` files from the Desktop or shared folders.

To export pilot data, share only the specific CSV or report files needed for review. Do not share screenshots or screen recordings unless the participant separately consents.

## Limits

Synapse estimates patterns from webcam landmarks. These estimates can be wrong because of lighting, camera angle, glasses, occlusion, face position, sensor quality, fatigue unrelated to work, or individual expression differences.

Synapse is not a medical device, diagnostic tool, lie detector, emotion oracle, mental health assessment, legal evidence tool, productivity score, or employee discipline tool. Treat outputs as supportive session notes, not facts about a person.

## Pilot Consent Language

Use this concise consent language before onboarding:

> Synapse uses this computer's webcam locally to estimate broad focus, fatigue, distraction, and expression-pattern signals during a pilot session. It does not intentionally save raw video or still images. It saves local calibration files, session metrics, alert logs, and text reports under your Windows user account. The results may be inaccurate and are not medical, psychological, employment, or disciplinary conclusions. You can stop at any time by pressing `q`, and you can ask for local Synapse data and exported reports to be deleted.
