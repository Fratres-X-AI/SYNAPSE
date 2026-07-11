# Known Issues And Constraints

These limitations apply to the current webcam-first pilot build.

## Python And Dependencies

- Synapse requires **Python 3.11 or 3.12**. This matches the pinned `mediapipe==0.10.21` wheels on Windows.
- **Python 3.13 and newer** often fail with `No matching distribution found for mediapipe` because no compatible wheel is published for that version.
- **Python 3.7–3.10 are not supported** for this build. Downgrading to 3.7 will not fix MediaPipe install errors.
- Run `.\scripts\check_python.ps1` before `pip install -r requirements.txt` to verify the interpreter.
- If `mediapipe` imports fail on Windows with DLL errors, install the [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) (x64).
- `requirements.txt` pins `numpy<2` because MediaPipe 0.10.21 is incompatible with NumPy 2.x.
- Non-developers should prefer `Synapse.exe` from a GitHub Release (`docs/windows_install.md`) and skip Python entirely.

## Monitor Vs Showcase

- `monitor` is the pilot path: session CSV/logging, reports, and HUD without landmark overlays.
- `showcase` is the demo path: elite helmet shell, dense face mesh, and flight HUD (no session logging).
- Use `python synapse_launcher.py showcase` for demos; use `monitor` for real sessions.

## Webcam And Environment

- Face detection can fail in low light, harsh backlight, glare, or very flat lighting.
- Glasses, sunglasses, masks, hair, hats, hand occlusion, and strong shadows can reduce accuracy.
- Off-center webcams, laptop movement, and unusual camera angles can skew gaze/head estimates.
- Multiple faces in frame are not a supported use case.
- External cameras may require changing local settings before launch.
- Other apps can lock the webcam and prevent Synapse from starting.

## Metrics

- Engagement, fatigue, tension, positivity, and distraction are noisy estimates, not facts.
- Expression matching depends on each user's onboarding profile and may not generalize.
- Blink and eye-openness signals can be affected by dry eyes, contacts, glasses, allergies, lighting, and camera quality.
- Gaze and head-pose estimates are approximate and should be reviewed with participant feedback.

## Workflow

- The current build is local-first and single-user per Windows account.
- Raw video is not intentionally stored, but session CSVs and text reports can still be sensitive.
- Replay can use the latest app-data session by default, or an explicit CSV path such as `%LOCALAPPDATA%\Synapse\sessions\monitor_YYYYMMDD_HHMMSS.csv`.
- The checked-in `build.ps1` creates a PyInstaller smoke-test package; release verification is still required before pilot sharing.

## Product Status

- No billing should be enabled before the software quality gate.
- Pilot results should be used to improve reliability, consent, onboarding, and report clarity.
- Do not use Synapse for employment, medical, legal, or disciplinary decisions.
