# CORTEX RailGuard

Far Away 2026 Railway Theme MVP.

RailGuard pivots the existing visual-memory MVP into a focused railway safety agent:

- video input
- person tracking
- platform-edge restricted-zone reasoning
- dynamic train-adjacent restricted zone
- train-context severity escalation
- evidence snapshot
- incident memory
- visible alert
- response task and action log
- evidence-backed query
- incident acknowledgement
- HTML incident report

The detector, object memory, foveal crop and Flask interface are reused. The RailGuard layer adds the PRD P0 adapters around that working foundation.

## Run

From `/home/manish/Desktop/Prototype`:

```bash
source adas-env/bin/activate
cd far_away_gppe_v1_mvp
python app.py
```

If `models/yolov8n.pt` is missing after a fresh clone, run:

```bash
python download_model.py
```

For a clean Python environment:

```bash
python -m pip install -r requirements.txt
```

Open:

```text
http://127.0.0.1:7860
```

## Recommended Demo

1. Press `Run Known Demo`.
2. Wait for a `CRITICAL` platform-intrusion incident.
3. Ask: `when did the intrusion occur`.
4. Ask: `was train visible during the incident`.
5. Press `Acknowledge` on the incident card.
6. Press `Generate Incident Report`.
7. Open the generated report link.

The one-click known demo uses the packaged clip:

```text
demo_assets/Demo.mp4
```

The clip is committed with the project so judges can run the demo without rebuilding assets. `CRITICAL` severity is produced from visual train context, not from a public UI override.

## P0 Build Scope

Implemented:

- `FramePacket`: frame ID, timestamp and source wrapper.
- `Tracker`: existing IoU/centroid memory keeps stable IDs for the demo clip.
- `Zone manager`: normalized fallback polygon plus dynamic train-adjacent zone from detected train box.
- `Context builder`: visible classes, counts, zone membership and train context.
- `Policy engine`: person foot-point entering platform edge.
- `Severity`: `HIGH`, escalated to `CRITICAL` when train context is true.
- `Temporal train context`: visual train detections persist for a short window to prevent severity flicker.
- `Deduplication`: one incident on `OUTSIDE -> INSIDE`, plus scene-level cooldown to avoid duplicate events from tracker ID churn.
- `EventStore`: SQLite incidents, action logs and response tasks.
- `Evidence`: annotated frame and object crop for each incident; thumbnails are clickable in the timeline.
- `Actions`: local alert, save evidence and response task.
- `Memory query`: deterministic incident answers with evidence references.
- `Acknowledge`: incident and response task acknowledgement.
- `Report`: HTML report from stored incident/action data.
- `Demo mode`: one-click known demo video.

## Tested Gates

The local reliability test passed five consecutive end-to-end runs:

```text
video input -> person tracking -> zone entry -> train-context CRITICAL severity
-> evidence -> incident memory -> action/task -> evidence-backed query -> acknowledgement -> report
```

Latest final acceptance:

```text
railguard_logic_acceptance PASS
```

## Honest Limitations

- Detector is pretrained YOLOv8n, not a custom railway-trained model.
- The known demo clip is packaged in `demo_assets/Demo.mp4`.
- Actions are local demo actions, not connected to railway control systems.
- This is an MVP for hackathon demonstration, not a certified safety product.
