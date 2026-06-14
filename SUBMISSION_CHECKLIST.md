# CORTEX RailGuard Submission Checklist

## What To Submit

- GitHub repository containing this folder.
- Short demo video showing the full RailGuard loop.
- Optional presentation using the same story as the demo script below.

## Demo Script

1. Start the app.
2. Press `Reset Run`.
3. Press `Run Known Demo`.
4. Show `CURRENT SCENE` telemetry: CPU mode, FPS, people tracked, train context, platform-edge state and risk.
5. Wait for the platform-edge intrusion incident.
6. Show the incident card with timestamp, tracked person ID, train context and evidence thumbnails.
7. Ask: `was train visible during the incident`.
8. Click `Acknowledge`.
9. Click `Generate Incident Report`.
10. Open the report and show summary cards, evidence, task and limitations.

## Local Run

```bash
python -m pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

## Packaged Demo Asset

```text
demo_assets/Demo.mp4
```

## Acceptance Gate

The packaged demo must produce:

- no frame-0 incident
- bounded platform-intrusion incidents without duplicate spam
- separate evidence package for each distinct person/entry
- visual train context for CRITICAL severity
- evidence frame and object crop
- response task
- evidence-backed query answer
- HTML report
