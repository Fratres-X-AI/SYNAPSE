# Synapse

Synapse is a closed-loop cognitive state system. It starts with webcam-based facial landmarks, estimates a user's cognitive state in real time, and adapts autonomous behavior from that state.

The long-term target is a wearable form factor, such as a small inward-facing camera mounted near the eye. For that reason, the first estimator prioritizes eye behavior, blink dynamics, and lightweight head movement signals.

## Current Loop

1. `CameraCapture` reads webcam frames and extracts MediaPipe Face Mesh landmarks.
2. `StateEstimator` converts landmarks into signals like EAR, blink rate, and head yaw.
3. `CognitiveState` represents the user's current state and confidence.
4. `AdaptiveAgent` adjusts autonomy based on that state.
5. `render_status` overlays the state and signals on the camera feed.

## Project Structure

```text
synapse/
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── perception/
│   │   ├── capture.py
│   │   └── state_estimator.py
│   ├── cognition/
│   │   └── cognitive_state.py
│   ├── adaptation/
│   │   └── adaptive_agent.py
│   └── visualization/
│       └── display.py
└── utils/
    └── config.py
```

## Setup

```powershell
pip install -r requirements.txt
python main.py
```

Press `q` in the camera window to quit.

## Cognitive States

- `high_attention`
- `moderate`
- `fatigued`
- `distracted`

## Starter Rules

- High blink rate and low EAR -> `fatigued`
- Significant head yaw -> `distracted`
- Normal EAR, low blink rate, and forward head pose -> `high_attention`
- Everything else -> `moderate`

## Next Steps

- Calibrate EAR thresholds per user.
- Replace lightweight yaw estimation with a proper `solvePnP` head pose model.
- Add gaze direction from iris landmarks.
- Add a wearable camera adapter under `src/perception`.
