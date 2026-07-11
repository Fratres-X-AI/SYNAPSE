# Synapse Windows Install (Pilot)

Use this guide to run Synapse on a Windows machine without installing Python.

## Requirements

- Windows 10 or 11
- A working webcam
- No other app locking the camera

## Install

1. Download `Synapse.exe` from the latest [GitHub Release](https://github.com/Fratres-X-AI/SYNAPSE/releases).
2. Create a folder such as `C:\Synapse` and place `Synapse.exe` there.
3. Create a desktop shortcut (optional but recommended):

```powershell
cd C:\Synapse
.\scripts\install_desktop_shortcut.ps1
```

If you only have the `.exe` (no scripts folder), right-click `Synapse.exe` → **Send to** → **Desktop (create shortcut)**, then edit the shortcut: append `home` to the target so it reads `Synapse.exe home`.

Double-click the desktop icon to open the Synapse launcher window.

## First Run

```powershell
.\Synapse.exe first-run
```

This will:

1. Show the privacy notice and ask for consent.
2. Run onboarding (calibration + optional expression profile).
3. Start your first monitor session.

Press `q` to quit the camera window. Press `f` to toggle fullscreen where supported.

## Common Commands

```powershell
.\Synapse.exe onboard
.\Synapse.exe monitor
.\Synapse.exe monitor --fullscreen
.\Synapse.exe showcase
.\Synapse.exe showcase --fullscreen
.\Synapse.exe replay
.\Synapse.exe data
.\Synapse.exe settings
.\Synapse.exe pilot-summary
.\Synapse.exe --tray
```

Replay without a path uses the latest session under `%LOCALAPPDATA%\Synapse\sessions`.

## Where Data Is Stored

```text
%LOCALAPPDATA%\Synapse
```

Synapse does not intentionally save raw webcam video. Session CSVs, alert logs, and text reports are saved locally.

## Delete Local Data

```powershell
.\Synapse.exe delete-data
```

Type `DELETE` when prompted.

## Troubleshooting

- **Camera unavailable:** Close other apps using the webcam and retry.
- **No face detected:** Improve lighting and center your face in frame.
- **Tray icon missing:** Install is console-only; tray requires optional dependencies in developer builds.
- **Windows SmartScreen warning:** The pilot build is unsigned. Choose "More info" → "Run anyway" only if you trust the source.

## Pilot Checklist

Before sharing with participants, verify `docs/release_checklist.md` on a clean machine.
