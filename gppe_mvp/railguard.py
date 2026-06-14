from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import ROOT
from .detector import Detection
from .memory import Track


MISSION_ID = "platform_intrusion_v1"
RULE_ID = "person_enters_platform_edge"
SOURCE_ID = "demo_platform_01"
EVENT_TYPE = "RESTRICTED_ZONE_INTRUSION"
INCIDENT_COOLDOWN_MS = 8000
INCIDENT_ARMING_MS = 1500
TRAIN_CONTEXT_MEMORY_MS = 3000
TRAIN_CONTEXT_WINDOW = 5
TRAIN_CONTEXT_MIN_HITS = 2


@dataclass
class Zone:
    zone_id: str
    name: str
    zone_type: str
    polygon_norm: list[tuple[float, float]]

    def polygon_px(self, frame_shape: tuple[int, int, int]) -> np.ndarray:
        h, w = frame_shape[:2]
        pts = [(int(x * w), int(y * h)) for x, y in self.polygon_norm]
        return np.array(pts, dtype=np.int32)

    def contains_foot(self, box: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> bool:
        x1, _, x2, y2 = box
        foot = ((x1 + x2) * 0.5, float(y2))
        return cv2.pointPolygonTest(self.polygon_px(frame_shape), foot, False) >= 0

    def foot_point(self, box: tuple[int, int, int, int]) -> tuple[int, int]:
        x1, _, x2, y2 = box
        return ((x1 + x2) // 2, y2)


@dataclass
class Mission:
    mission_id: str = MISSION_ID
    text: str = "Monitor the platform edge. Critical if a train is visible."
    primary_zone: str = "platform_edge"
    objects: list[str] = field(default_factory=lambda: ["person", "train", "suitcase", "backpack"])
    severity_base: str = "HIGH"
    cooldown_frames: int = 45
    escalate_on_train: bool = True
    demo_train_context: bool = False
    actions: list[str] = field(default_factory=lambda: ["save_evidence", "alert_ui", "create_task"])

    def compiled_rule(self) -> str:
        escalation = "CRITICAL if train_visible else HIGH" if self.escalate_on_train else self.severity_base
        return f"{RULE_ID}: person foot-point enters {self.primary_zone}; severity={escalation}; actions={', '.join(self.actions)}"


@dataclass
class FramePacket:
    frame_id: int
    timestamp_ms: int
    source_id: str
    image: np.ndarray


@dataclass
class SceneContext:
    visible_classes: list[str]
    counts: dict[str, int]
    train_visible: bool
    train_confidence: float
    zone_membership: dict[int, dict[str, bool]]
    active_zones: list[Zone] = field(default_factory=list)


@dataclass
class IncidentEvent:
    event_id: str
    mission_id: str
    rule_id: str
    source_id: str
    frame_id: int
    timestamp_ms: int
    occurred_at: str
    track_id: int
    object_class: str
    zone_id: str
    event_type: str
    severity: str
    condition_values: dict[str, Any]
    evidence: dict[str, str]
    status: str = "OPEN"


@dataclass
class ActionResult:
    action_type: str
    status: str
    executed_at: str
    detail: str


@dataclass
class ResponseTask:
    task_id: str
    event_id: str
    title: str
    status: str
    created_at: str


class EventStore:
    def __init__(self, db_path: Path, evidence_dir: Path, reports_dir: Path) -> None:
        self.db_path = db_path
        self.evidence_dir = evidence_dir
        self.reports_dir = reports_dir
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def reset_run(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM action_logs")
            con.execute("DELETE FROM response_tasks")
            con.execute("DELETE FROM incidents")
            con.commit()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    event_id TEXT PRIMARY KEY,
                    mission_id TEXT NOT NULL,
                    rule_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    frame_id INTEGER NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    track_id INTEGER,
                    object_class TEXT,
                    zone_id TEXT,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    condition_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN'
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS action_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    executed_at TEXT NOT NULL,
                    detail TEXT,
                    FOREIGN KEY(event_id) REFERENCES incidents(event_id)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS response_tasks (
                    task_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES incidents(event_id)
                )
                """
            )
            con.commit()

    def next_event_id(self) -> str:
        count = len(self.list_incidents()) + 1
        return f"RG-{datetime.now().strftime('%Y%m%d')}-{count:04d}"

    def save_incident(self, event: IncidentEvent) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO incidents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.mission_id,
                    event.rule_id,
                    event.source_id,
                    event.frame_id,
                    event.timestamp_ms,
                    event.occurred_at,
                    event.track_id,
                    event.object_class,
                    event.zone_id,
                    event.event_type,
                    event.severity,
                    json.dumps(event.condition_values),
                    json.dumps(event.evidence),
                    event.status,
                ),
            )
            con.commit()

    def save_action(self, event_id: str, action: ActionResult) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO action_logs(event_id, action_type, status, executed_at, detail) VALUES (?, ?, ?, ?, ?)",
                (event_id, action.action_type, action.status, action.executed_at, action.detail),
            )
            con.commit()

    def save_task(self, task: ResponseTask) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO response_tasks VALUES (?, ?, ?, ?, ?)",
                (task.task_id, task.event_id, task.title, task.status, task.created_at),
            )
            con.commit()

    def acknowledge_incident(self, event_id: str) -> bool:
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute("UPDATE incidents SET status='ACKNOWLEDGED' WHERE event_id=?", (event_id,))
            con.execute("UPDATE response_tasks SET status='DISPATCHED' WHERE event_id=?", (event_id,))
            con.commit()
            return cur.rowcount > 0

    def list_incidents(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM incidents ORDER BY timestamp_ms DESC, event_id DESC").fetchall()
        return [self._decode_incident(dict(r)) for r in rows]

    def list_actions(self, event_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM action_logs"
        params: tuple[Any, ...] = ()
        if event_id:
            query += " WHERE event_id=?"
            params = (event_id,)
        query += " ORDER BY id DESC"
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def list_tasks(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM response_tasks ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _decode_incident(row: dict[str, Any]) -> dict[str, Any]:
        row["condition_values"] = json.loads(row.pop("condition_json") or "{}")
        row["evidence"] = json.loads(row.pop("evidence_json") or "{}")
        return row


class RailGuardEngine:
    def __init__(self) -> None:
        self.mission = Mission()
        self.zones = [
            Zone(
                zone_id="platform_edge",
                name="Platform Edge Restricted Zone",
                zone_type="platform_edge",
                polygon_norm=[(0.0, 0.60), (1.0, 0.60), (1.0, 1.0), (0.0, 1.0)],
            )
        ]
        self.store = EventStore(
            db_path=ROOT / "railguard_data" / "railguard.sqlite3",
            evidence_dir=ROOT / "static" / "railguard_evidence",
            reports_dir=ROOT / "static" / "railguard_reports",
        )
        self.track_inside: dict[tuple[int, str], bool] = {}
        self.cooldowns: dict[tuple[int, str], int] = {}
        self.scene_cooldown_until_ms = 0
        self.last_train_seen_ms = -10_000_000
        self.last_train_confidence = 0.0
        self.train_history: list[tuple[int, bool, float]] = []
        self.active_zones = list(self.zones)
        self.latest_context = SceneContext([], {}, False, 0.0, {}, list(self.zones))
        self.latest_alert: dict[str, Any] | None = None
        self.run_started_at = now_iso()

    def reset_run(self) -> None:
        self.store.reset_run()
        self.track_inside.clear()
        self.cooldowns.clear()
        self.scene_cooldown_until_ms = 0
        self.last_train_seen_ms = -10_000_000
        self.last_train_confidence = 0.0
        self.train_history = []
        self.active_zones = list(self.zones)
        self.latest_context = SceneContext([], {}, False, 0.0, {}, list(self.zones))
        self.latest_alert = None
        self.run_started_at = now_iso()

    def build_context(self, packet: FramePacket, tracks: list[Track]) -> SceneContext:
        counts: dict[str, int] = {}
        visible_classes = []
        zone_membership: dict[int, dict[str, bool]] = {}
        active_zones = self._zones_for_frame(packet.image, tracks)
        self.active_zones = active_zones
        raw_train_confidence = 0.0
        for track in tracks:
            if not track.visible:
                continue
            counts[track.label] = counts.get(track.label, 0) + 1
            visible_classes.append(track.label)
            if track.label == "train":
                raw_train_confidence = max(raw_train_confidence, track.conf)
            zone_membership[track.track_id] = {
                zone.zone_id: zone.contains_foot(track.box, packet.image.shape) for zone in active_zones
            }
        raw_train_visible = counts.get("train", 0) > 0
        self._update_train_memory(packet.timestamp_ms, raw_train_visible, raw_train_confidence)
        train_visible = self.mission.demo_train_context or self._train_context_active(packet.timestamp_ms)
        train_confidence = max(raw_train_confidence, self.last_train_confidence if train_visible else 0.0)
        context = SceneContext(
            visible_classes=sorted(set(visible_classes)),
            counts=counts,
            train_visible=train_visible,
            train_confidence=train_confidence,
            zone_membership=zone_membership,
            active_zones=active_zones,
        )
        self.latest_context = context
        return context

    def _update_train_memory(self, timestamp_ms: int, detected: bool, confidence: float) -> None:
        self.train_history.append((timestamp_ms, detected, confidence))
        self.train_history = self.train_history[-TRAIN_CONTEXT_WINDOW:]
        if detected:
            self.last_train_seen_ms = timestamp_ms
            self.last_train_confidence = max(float(confidence), self.last_train_confidence * 0.92)
        elif timestamp_ms - self.last_train_seen_ms > TRAIN_CONTEXT_MEMORY_MS:
            self.last_train_confidence = 0.0

    def _train_context_active(self, timestamp_ms: int) -> bool:
        if timestamp_ms - self.last_train_seen_ms <= TRAIN_CONTEXT_MEMORY_MS:
            return True
        recent = self.train_history[-TRAIN_CONTEXT_WINDOW:]
        hits = sum(1 for _, detected, _ in recent if detected)
        return hits >= TRAIN_CONTEXT_MIN_HITS

    def process(self, packet: FramePacket, tracks: list[Track]) -> list[IncidentEvent]:
        context = self.build_context(packet, tracks)
        incidents: list[IncidentEvent] = []
        visible_person_keys: set[tuple[int, str]] = set()
        armed = packet.timestamp_ms >= INCIDENT_ARMING_MS
        for track in tracks:
            if not track.visible or track.label != "person":
                continue
            zone_id = self.mission.primary_zone
            key = (track.track_id, zone_id)
            visible_person_keys.add(key)
            inside = context.zone_membership.get(track.track_id, {}).get(zone_id, False)
            was_inside = self.track_inside.get(key, False)
            if not armed:
                self.track_inside[key] = inside
                continue
            cooldown_until = self.cooldowns.get(key, -1)
            can_create = packet.frame_id >= cooldown_until and packet.timestamp_ms >= self.scene_cooldown_until_ms
            if inside and not was_inside and can_create:
                event = self._create_intrusion(packet, track, context)
                self._save_evidence(packet.image, track, event, context.active_zones)
                self.store.save_incident(event)
                self._execute_actions(event)
                self.latest_alert = {
                    "event_id": event.event_id,
                    "severity": event.severity,
                    "message": f"LATEST INCIDENT - {event.severity}: P-{track.track_id:02d} entered platform edge"
                    + (" while train context was detected" if context.train_visible else ""),
                }
                incidents.append(event)
                self.cooldowns[key] = packet.frame_id + self.mission.cooldown_frames
                self.scene_cooldown_until_ms = packet.timestamp_ms + INCIDENT_COOLDOWN_MS
                self.track_inside[key] = inside
            self.track_inside[key] = inside
        for key in list(self.track_inside):
            if key not in visible_person_keys:
                self.track_inside[key] = False
        return incidents

    def _zones_for_frame(self, frame: np.ndarray, tracks: list[Track]) -> list[Zone]:
        train_tracks = [t for t in tracks if t.visible and t.label == "train"]
        if not train_tracks:
            return list(self.zones)
        h, w = frame.shape[:2]
        x1 = min(t.box[0] for t in train_tracks)
        y1 = min(t.box[1] for t in train_tracks)
        x2 = max(t.box[2] for t in train_tracks)
        y2 = max(t.box[3] for t in train_tracks)
        pad_x = int((x2 - x1) * 0.18)
        pad_y = int((y2 - y1) * 0.75)
        zx1 = max(0, x1 - pad_x) / max(1, w)
        zx2 = min(w, x2 + pad_x) / max(1, w)
        zy1 = max(0, y1 - int((y2 - y1) * 0.08)) / max(1, h)
        zy2 = min(h, y2 + pad_y) / max(1, h)
        return [
            Zone(
                zone_id="platform_edge",
                name="Train-adjacent restricted zone",
                zone_type="train_adjacent_platform_edge",
                polygon_norm=[(zx1, zy1), (zx2, zy1), (zx2, zy2), (zx1, zy2)],
            )
        ]

    def _create_intrusion(self, packet: FramePacket, track: Track, context: SceneContext) -> IncidentEvent:
        severity = "CRITICAL" if context.train_visible and self.mission.escalate_on_train else self.mission.severity_base
        return IncidentEvent(
            event_id=self.store.next_event_id(),
            mission_id=self.mission.mission_id,
            rule_id=RULE_ID,
            source_id=packet.source_id,
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            occurred_at=now_iso(),
            track_id=track.track_id,
            object_class=track.label,
            zone_id=self.mission.primary_zone,
            event_type=EVENT_TYPE,
            severity=severity,
            condition_values={
                "inside_zone": True,
                "train_visible": context.train_visible,
                "train_confidence": round(context.train_confidence, 3),
                "train_context_memory_seconds": round(TRAIN_CONTEXT_MEMORY_MS / 1000, 1),
                "person_count_in_zone": self._person_count_in_zone(context),
            },
            evidence={"annotated_frame": "", "object_crop": "", "evidence_status": "PENDING"},
        )

    def _person_count_in_zone(self, context: SceneContext) -> int:
        return sum(1 for zones in context.zone_membership.values() if zones.get(self.mission.primary_zone))

    def _save_evidence(self, frame: np.ndarray, track: Track, event: IncidentEvent, zones: list[Zone]) -> None:
        try:
            annotated = draw_railguard_overlay(frame, [track], zones, event)
            crop = crop_box(frame, track.box)
            full_name = f"{event.event_id}-frame.jpg"
            crop_name = f"{event.event_id}-crop.jpg"
            full_path = self.store.evidence_dir / full_name
            crop_path = self.store.evidence_dir / crop_name
            cv2.imwrite(str(full_path), annotated)
            cv2.imwrite(str(crop_path), crop)
            event.evidence = {
                "annotated_frame": f"/static/railguard_evidence/{full_name}",
                "object_crop": f"/static/railguard_evidence/{crop_name}",
                "evidence_status": "OK",
            }
        except Exception as exc:  # pragma: no cover - defensive for demo storage failures
            event.evidence = {"annotated_frame": "", "object_crop": "", "evidence_status": f"FAILED: {exc}"}

    def _execute_actions(self, event: IncidentEvent) -> None:
        for action_type in self.mission.actions:
            detail = "DEMO local action executed"
            if action_type == "alert_ui":
                detail = f"Visible operator alert for {event.severity} incident"
            elif action_type == "create_task":
                task = ResponseTask(
                    task_id=f"TASK-{event.event_id}",
                    event_id=event.event_id,
                    title=f"Dispatch station security | Target: Platform edge | Subject: P-{event.track_id:02d} | Priority: {event.severity}",
                    status="OPEN",
                    created_at=now_iso(),
                )
                self.store.save_task(task)
                detail = f"Response task created: {task.task_id}"
            elif action_type == "save_evidence":
                detail = f"Evidence status: {event.evidence.get('evidence_status', 'UNKNOWN')}"
            self.store.save_action(
                event.event_id,
                ActionResult(action_type=action_type, status="SUCCESS", executed_at=now_iso(), detail=detail),
            )

    def query(self, question: str) -> str:
        q = question.lower().strip()
        incidents = self.store.list_incidents()
        if not incidents:
            return "No matching incident found in this run."
        if "critical" in q:
            event = sorted(incidents, key=lambda e: severity_rank(e["severity"]), reverse=True)[0]
            return evidence_answer(event, "Most critical incident")
        if "how many" in q or "count" in q or "entered" in q:
            return f"Based on incident memory: {len(incidents)} restricted-zone intrusion event(s) were recorded."
        if "train" in q:
            event = incidents[0]
            visible = event["condition_values"].get("train_visible")
            conf = event["condition_values"].get("train_confidence", 0)
            person = f"P-{int(event['track_id']):02d}" if event.get("track_id") is not None else "the tracked person"
            if visible:
                return evidence_answer(
                    event,
                    f"Yes. A train was visually detected with confidence {float(conf):.2f} when {person} entered the train-adjacent restricted zone at {format_ms(event['timestamp_ms'])}",
                )
            return evidence_answer(event, f"No. Train context was not detected when {person} entered the restricted zone")
        if "when" in q or "occur" in q or "time" in q:
            event = incidents[0]
            person = f"P-{int(event['track_id']):02d}" if event.get("track_id") is not None else "the tracked person"
            return evidence_answer(event, f"{person} entered the train-adjacent restricted zone at {format_ms(event['timestamp_ms'])}, frame {event['frame_id']}")
        if "report" in q:
            path = self.generate_report()
            return f"Report generated: {path}"
        return evidence_answer(incidents[0], "Latest incident")

    def incident_rows(self) -> list[dict[str, Any]]:
        return self.store.list_incidents()

    def action_rows(self) -> list[dict[str, Any]]:
        return self.store.list_actions()[:60]

    def task_rows(self) -> list[dict[str, Any]]:
        return self.store.list_tasks()[:60]

    def acknowledge(self, event_id: str) -> str:
        if not event_id:
            incidents = self.store.list_incidents()
            event_id = incidents[0]["event_id"] if incidents else ""
        if not event_id:
            return "No incident is available to acknowledge."
        ok = self.store.acknowledge_incident(event_id)
        if ok:
            self.store.save_action(
                event_id,
                ActionResult(
                    action_type="acknowledge_incident",
                    status="SUCCESS",
                    executed_at=now_iso(),
                    detail=f"Operator acknowledged {event_id}; response task marked DISPATCHED",
                ),
            )
            return f"Incident {event_id} acknowledged."
        return f"Incident {event_id} was not found."

    def status_summary(self, fps: float, frame_idx: int) -> dict[str, Any]:
        context = self.latest_context
        incidents = self.store.list_incidents()
        latest_incident = incidents[0] if incidents else None
        zone_breached = any(
            zones.get(self.mission.primary_zone, False) for zones in context.zone_membership.values()
        )
        if zone_breached and context.train_visible:
            risk = "CRITICAL"
        elif zone_breached:
            risk = "HIGH"
        else:
            risk = "CLEAR"
        return {
            "monitoring": True,
            "runtime": "CPU MODE",
            "fps": round(float(fps), 2),
            "frame_idx": int(frame_idx),
            "people_tracked": int(context.counts.get("person", 0)),
            "train_context": "DETECTED" if context.train_visible else "NOT DETECTED",
            "train_confidence": round(float(context.train_confidence), 2),
            "train_source": "Visual perception" if context.train_visible and context.train_confidence > 0 else "Not detected",
            "zone_state": "BREACHED" if zone_breached else "SECURE",
            "risk": risk,
            "latest_incident": latest_incident_summary(latest_incident),
        }

    def generate_report(self) -> str:
        incidents = sorted(self.store.list_incidents(), key=lambda e: (int(e["timestamp_ms"]), str(e["event_id"])))
        actions = self.store.list_actions()
        tasks = self.store.list_tasks()
        counts: dict[str, int] = {}
        for event in incidents:
            counts[event["severity"]] = counts.get(event["severity"], 0) + 1
        max_ms = max((int(e["timestamp_ms"]) for e in incidents), default=0)
        unique_people = len({e["track_id"] for e in incidents if e.get("track_id") is not None})
        name = f"railguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        path = self.store.reports_dir / name
        rows = "\n".join(
            incident_report_row(e)
            for e in incidents
        )
        action_rows = "\n".join(
            f"<li>{a['event_id']} - {a['action_type']} - {a['status']} - {a['detail']}</li>" for a in actions
        )
        task_rows = "\n".join(
            f"<tr><td>{t['task_id']}</td><td>{t['status']}</td><td>{t['title']}</td><td>{t['created_at']}</td></tr>"
            for t in tasks
        )
        html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>CORTEX RailGuard Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:32px;background:#f6f8fb;color:#101828;line-height:1.45}}
.card{{background:white;border:1px solid #d9e1ee;border-radius:18px;padding:18px;margin:16px 0;box-shadow:0 12px 34px rgba(16,24,40,.06)}}
.summary{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:18px 0}}
.metric{{background:#fff;border:1px solid #d9e1ee;border-radius:16px;padding:14px}}
.metric span{{display:block;color:#667085;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em}}
.metric strong{{display:block;margin-top:8px;font-size:24px;color:#101828}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #ccd3df;padding:8px;vertical-align:top}}th{{background:#edf3fb;text-align:left}}img{{max-width:240px;border-radius:10px;border:1px solid #ccd3df}}.critical{{color:#b42318;font-weight:800}}.high{{color:#b54708;font-weight:800}}</style></head>
<body>
<h1>CORTEX RailGuard Incident Report</h1>
<div class="card">
<p><strong>Mission name:</strong> Platform-edge intrusion monitoring</p>
<p><strong>Monitoring policy:</strong> Create an incident when a tracked person's foot-point enters the platform-edge restricted zone. Escalate the incident from HIGH to CRITICAL when train context is active. Preserve evidence, alert the operator and create a security-response task.</p>
<p><strong>Video/source:</strong> {SOURCE_ID}</p>
<p><strong>Run started:</strong> {self.run_started_at}</p>
</div>
<div class="summary">
<div class="metric"><span>Monitoring duration</span><strong>{format_ms(max_ms)}</strong></div>
<div class="metric"><span>Unique people tracked</span><strong>{unique_people}</strong></div>
<div class="metric"><span>Platform intrusions</span><strong>{len(incidents)}</strong></div>
<div class="metric"><span>Critical incidents</span><strong>{counts.get("CRITICAL", 0)}</strong></div>
<div class="metric"><span>High incidents</span><strong>{counts.get("HIGH", 0)}</strong></div>
<div class="metric"><span>Response tasks</span><strong>{len(tasks)}</strong></div>
</div>
<h2>Incident Evidence</h2>
<table><thead><tr><th>Incident</th><th>When</th><th>Subject</th><th>Zone</th><th>Train Context</th><th>Severity</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Response Tasks</h2>
<table><thead><tr><th>Task</th><th>Status</th><th>Details</th><th>Created</th></tr></thead><tbody>{task_rows}</tbody></table>
<h2>Actions</h2><ul>{action_rows}</ul>
<h2>Operator Acknowledgment</h2><p>Incidents marked ACKNOWLEDGED were reviewed by the operator; linked response tasks are marked DISPATCHED.</p>
<h2>Limitations</h2><p>Hackathon MVP; detector is pretrained COCO YOLOv8n, local demo actions only, not connected to railway control systems.</p>
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return f"/static/railguard_reports/{name}"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def severity_rank(severity: str) -> int:
    return {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(severity.upper(), 0)


def evidence_answer(event: dict[str, Any], text: str) -> str:
    return f"{text}.\n\nIncident: {event['event_id']}\nSeverity: {event['severity']}\nEvidence: View captured frame in the incident timeline."


def format_ms(timestamp_ms: int) -> str:
    seconds = max(0.0, timestamp_ms / 1000.0)
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:04.1f}"


def incident_report_row(event: dict[str, Any]) -> str:
    values = event.get("condition_values", {})
    evidence = event.get("evidence", {})
    frame = evidence.get("annotated_frame", "")
    crop = evidence.get("object_crop", "")
    subject = f"P-{int(event['track_id']):02d}" if event.get("track_id") is not None else "tracked person"
    train = "DETECTED" if values.get("train_visible") else "NOT DETECTED"
    conf = float(values.get("train_confidence") or 0.0)
    image_html = ""
    if frame:
        image_html += f"<a href='{frame}'><img src='{frame}' alt='evidence frame'></a>"
    if crop:
        image_html += f"<br><a href='{crop}'>Object crop</a>"
    severity_class = str(event["severity"]).lower()
    return (
        f"<tr><td>{event['event_id']}</td><td>{format_ms(int(event['timestamp_ms']))}<br>Frame {event['frame_id']}</td>"
        f"<td>{subject}</td><td>{event['zone_id']}</td><td>{train}<br>Confidence {conf:.2f}</td>"
        f"<td class='{severity_class}'>{event['severity']}</td><td>{event['status']}</td><td>{image_html}</td></tr>"
    )


def latest_incident_summary(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not event:
        return None
    values = event.get("condition_values", {})
    subject = f"P-{int(event['track_id']):02d}" if event.get("track_id") is not None else "tracked person"
    return {
        "event_id": event["event_id"],
        "subject": subject,
        "severity": event["severity"],
        "timestamp": format_ms(int(event["timestamp_ms"])),
        "frame_id": int(event["frame_id"]),
        "train_detected_at_incident": bool(values.get("train_visible")),
        "train_confidence_at_incident": round(float(values.get("train_confidence") or 0.0), 2),
        "status": event["status"],
    }


def crop_box(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad = 20
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((120, 160, 3), dtype=np.uint8)
    return crop


def draw_railguard_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    zones: list[Zone],
    incident: IncidentEvent | None = None,
) -> np.ndarray:
    out = frame.copy()
    for zone in zones:
        pts = zone.polygon_px(out.shape)
        color = (0, 0, 255) if incident and incident.severity == "CRITICAL" else (0, 165, 255)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.16, out, 0.84, 0, out)
        cv2.polylines(out, [pts], True, color, 3)
        x, y = pts[0]
        cv2.putText(out, "RESTRICTED ZONE", (x + 8, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
    for track in tracks:
        if not track.visible or track.label not in {"person", "train"}:
            continue
        x1, y1, x2, y2 = track.box
        if track.label == "train":
            color = (255, 210, 80)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, "TRAIN CONTEXT", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
            continue
        inside = any(zone.contains_foot(track.box, out.shape) for zone in zones)
        color = (0, 0, 255) if inside else (0, 215, 255)
        thickness = 4 if inside else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        foot = ((x1 + x2) // 2, y2)
        cv2.circle(out, foot, 7, color, -1)
        cv2.circle(out, foot, 11, (255, 255, 255), 2)
        label = f"P-{track.track_id:02d}"
        state = "INSIDE ZONE" if inside else "APPROACHING"
        cv2.putText(out, label, (x1, max(22, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
        cv2.putText(out, state, (x1, max(44, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
        if inside:
            cv2.putText(out, "foot-point inside restricted zone", (max(8, foot[0] - 130), min(out.shape[0] - 12, foot[1] + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)
    if incident:
        banner = f"{incident.severity} {incident.event_id}: PLATFORM INTRUSION"
        cv2.rectangle(out, (0, 0), (out.shape[1], 46), (0, 0, 170), -1)
        cv2.putText(out, banner, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    return out
