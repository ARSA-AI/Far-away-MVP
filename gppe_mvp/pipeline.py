from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import DETECT_EVERY_N_FRAMES
from .detector import Detection, YoloDetector
from .foveal import choose_focus, make_foveal_crop
from .memory import ObjectMemory, Track
from .railguard import FramePacket, RailGuardEngine, draw_railguard_overlay
from .render import draw_overlay, encode_jpeg


@dataclass
class RuntimeState:
    query: str = ""
    source: str = "idle"
    mode: str = "idle"
    fps: float = 0.0
    frame_idx: int = 0
    focus_reason: str = "none"
    focus_name: str = "none"
    answer: str = "Upload an image or video, then ask what the system sees."
    labels: list[str] = field(default_factory=list)
    memory_rows: list[dict] = field(default_factory=list)
    incidents: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    tasks: list[dict] = field(default_factory=list)
    system_status: dict = field(default_factory=dict)
    compiled_rule: str = ""
    alert: dict | None = None
    report_url: str = ""
    last_frame_jpeg: bytes = b""
    foveal_jpeg: bytes = b""
    running: bool = False
    error: str = ""


class GPPEMvpRuntime:
    def __init__(self) -> None:
        self.detector = YoloDetector()
        self.memory = ObjectMemory()
        self.railguard = RailGuardEngine()
        self.state = RuntimeState()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_detections: list[Detection] = []
        self._current_frame: np.ndarray | None = None
        self._current_tracks: list[Track] = []
        self._image_history: list[dict] = []

    def set_query(self, query: str) -> None:
        target = query.strip().lower()
        with self._lock:
            self.state.query = target
        self._rerender_current()
        if target:
            answer = self._locate_answer(target)
            with self._lock:
                self.state.answer = answer

    def clear_memory(self) -> None:
        self.stop()
        self.memory = ObjectMemory()
        self.railguard.reset_run()
        self._last_detections = []
        self._current_frame = None
        self._current_tracks = []
        self._image_history = []
        with self._lock:
            self.state = RuntimeState(answer="Memory cleared. Upload images or start a video source.")

    def start(self, source: str | int = 0) -> None:
        self.stop()
        self._stop.clear()
        self.memory = ObjectMemory()
        self.railguard.reset_run()
        self._last_detections = []
        self._current_frame = None
        self._current_tracks = []
        self._image_history = []
        with self._lock:
            self.state = RuntimeState(
                source=str(source),
                mode="railguard",
                running=True,
                answer="RailGuard monitoring started.",
                compiled_rule=self.railguard.mission.compiled_rule(),
            )
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def set_train_context(self, enabled: bool) -> None:
        self.railguard.mission.demo_train_context = enabled
        with self._lock:
            self.state.compiled_rule = self.railguard.mission.compiled_rule()

    def generate_report(self) -> str:
        report_url = self.railguard.generate_report()
        with self._lock:
            self.state.report_url = report_url
            self.state.answer = f"RailGuard report generated: {report_url}"
        return report_url

    def acknowledge_incident(self, event_id: str = "") -> str:
        answer = self.railguard.acknowledge(event_id)
        with self._lock:
            self.state.answer = answer
            self.state.incidents = self.railguard.incident_rows()
            self.state.actions = self.railguard.action_rows()
            self.state.tasks = self.railguard.task_rows()
        return answer

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        with self._lock:
            self.state.running = False

    def snapshot(self) -> RuntimeState:
        with self._lock:
            return RuntimeState(**self.state.__dict__)

    def process_images(self, paths: list[Path]) -> None:
        self.stop()
        valid_paths = [p for p in paths if p.exists()]
        if not valid_paths:
            with self._lock:
                self.state.error = "No valid image files uploaded."
            return

        started = time.perf_counter()
        processed = 0
        query = ""
        with self._lock:
            query = ""
            self.state.query = ""
            self.state.mode = "image"
            self.state.source = f"{len(valid_paths)} image(s)"
            self.state.running = True
            self.state.error = ""
            self.state.answer = "Processing uploaded image(s)..."

        for path in valid_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            processed += 1
            frame = self._resize_max(frame, max_w=960)
            detections = self.detector.detect(frame)
            self._last_detections = detections
            frame_idx = self.state.frame_idx + 1
            tracks = self.memory.update(frame, detections, frame_idx)
            self._current_frame = frame.copy()
            self._current_tracks = tracks
            self._remember_image(path.name, frame_idx, tracks)
            with self._lock:
                query = self.state.query
            focus, reason = choose_focus(tracks, query)
            overlay = draw_overlay(frame, tracks, focus.track_id if focus else None, query)
            foveal = make_foveal_crop(frame, focus)
            self._publish_frame(
                frame_idx=frame_idx,
                fps=processed / max(1e-6, time.perf_counter() - started),
                tracks=tracks,
                focus=focus,
                reason=reason,
                overlay=overlay,
                foveal=foveal,
                running=True,
            )

        with self._lock:
            self.state.running = False
            if processed == 0:
                self.state.error = "Uploaded files could not be read as images."
                self.state.answer = self.state.error
            else:
                self.state.answer = self._summarize_visible_locked()

    def answer_question(self, question: str) -> str:
        q = question.strip().lower()
        with self._lock:
            current_rows = list(self.state.memory_rows)
            labels = list(self.state.labels)
            mode = self.state.mode
        if mode == "railguard":
            answer = self.railguard.query(question)
            with self._lock:
                self.state.answer = answer
                self.state.incidents = self.railguard.incident_rows()
                self.state.actions = self.railguard.action_rows()
                self.state.tasks = self.railguard.task_rows()
            return answer
        memory_rows = self._all_memory_rows()
        locate_target = self._extract_locate_target(q, memory_rows)
        if locate_target:
            with self._lock:
                self.state.query = locate_target
            self._rerender_current()
        if self._asks_history(q):
            answer = self._answer_history(q)
        else:
            answer = self._answer_from_rows(q, current_rows, memory_rows, labels)
        with self._lock:
            self.state.answer = answer
        return answer

    def _run(self, source: str | int) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            with self._lock:
                self.state.error = f"Could not open source: {source}"
                self.state.running = False
            return

        frame_idx = 0
        ema_fps = 0.0
        last_t = time.perf_counter()

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            # Keep UI responsive on weak CPUs.
            frame = self._resize_max(frame, max_w=960)
            if frame_idx % DETECT_EVERY_N_FRAMES == 0:
                self._last_detections = self.detector.detect(frame)

            tracks = self.memory.update(frame, self._last_detections, frame_idx)
            self._current_frame = frame.copy()
            self._current_tracks = tracks
            with self._lock:
                query = self.state.query
            focus, reason = choose_focus(tracks, query)
            packet = FramePacket(
                frame_id=frame_idx,
                timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)) if cap.get(cv2.CAP_PROP_POS_MSEC) else int(frame_idx * 1000 / 24),
                source_id="demo_platform_01",
                image=frame,
            )
            new_incidents = self.railguard.process(packet, tracks)
            rail_incidents = self.railguard.incident_rows()
            overlay = draw_railguard_overlay(frame, tracks, self.railguard.active_zones, None)
            foveal = make_foveal_crop(frame, focus)

            now = time.perf_counter()
            dt = max(1e-6, now - last_t)
            last_t = now
            inst_fps = 1.0 / dt
            ema_fps = inst_fps if ema_fps <= 0 else 0.9 * ema_fps + 0.1 * inst_fps

            labels = sorted({t.label for t in tracks if t.visible})
            rows = [self._track_row(t, frame_idx) for t in sorted(tracks, key=lambda t: (not t.visible, t.label, t.track_id))]
            self._publish_rows(
                frame_idx=frame_idx,
                fps=ema_fps,
                labels=labels,
                rows=rows,
                focus_name=focus.name if focus else "none",
                reason=reason,
                overlay=overlay,
                foveal=foveal,
                running=True,
            )
            with self._lock:
                self.state.incidents = rail_incidents
                self.state.actions = self.railguard.action_rows()
                self.state.tasks = self.railguard.task_rows()
                self.state.system_status = self.railguard.status_summary(ema_fps, frame_idx)
                self.state.alert = self.railguard.latest_alert
                self.state.compiled_rule = self.railguard.mission.compiled_rule()
                if new_incidents:
                    event = new_incidents[-1]
                    self.state.answer = f"Based on incident {event.event_id}: {event.severity} platform intrusion recorded with evidence."

            frame_idx += 1

        cap.release()
        with self._lock:
            self.state.running = False

    def _remember_image(self, name: str, frame_idx: int, tracks: list[Track]) -> None:
        rows = [self._track_row(t, frame_idx) for t in tracks if t.visible]
        self._image_history.append(
            {
                "index": len(self._image_history) + 1,
                "name": name,
                "frame_idx": frame_idx,
                "rows": rows,
            }
        )
        self._image_history = self._image_history[-24:]

    def _rerender_current(self) -> None:
        frame = self._current_frame
        tracks = list(self._current_tracks)
        if frame is None or not tracks:
            return
        with self._lock:
            query = self.state.query
            frame_idx = self.state.frame_idx
            fps = self.state.fps
            running = self.state.running
        focus, reason = choose_focus(tracks, query)
        overlay = draw_overlay(frame, tracks, focus.track_id if focus else None, query)
        foveal = make_foveal_crop(frame, focus)
        self._publish_frame(
            frame_idx=frame_idx,
            fps=fps,
            tracks=tracks,
            focus=focus,
            reason=reason,
            overlay=overlay,
            foveal=foveal,
            running=running,
        )

    def _publish_frame(
        self,
        frame_idx: int,
        fps: float,
        tracks: list[Track],
        focus: Track | None,
        reason: str,
        overlay: np.ndarray,
        foveal: np.ndarray,
        running: bool,
    ) -> None:
        labels = sorted({t.label for t in tracks if t.visible})
        rows = [self._track_row(t, frame_idx) for t in sorted(tracks, key=lambda t: (not t.visible, t.label, t.track_id))]
        self._publish_rows(
            frame_idx=frame_idx,
            fps=fps,
            labels=labels,
            rows=rows,
            focus_name=focus.name if focus else "none",
            reason=reason,
            overlay=overlay,
            foveal=foveal,
            running=running,
        )

    def _publish_rows(
        self,
        frame_idx: int,
        fps: float,
        labels: list[str],
        rows: list[dict],
        focus_name: str,
        reason: str,
        overlay: np.ndarray,
        foveal: np.ndarray,
        running: bool,
    ) -> None:
        with self._lock:
            self.state.fps = fps
            self.state.frame_idx = frame_idx
            self.state.focus_reason = reason
            self.state.focus_name = focus_name
            self.state.labels = labels
            self.state.memory_rows = rows[:20]
            self.state.last_frame_jpeg = encode_jpeg(overlay)
            self.state.foveal_jpeg = encode_jpeg(foveal)
            self.state.running = running

    def _summarize_visible_locked(self) -> str:
        rows = list(self.state.memory_rows)
        labels = list(self.state.labels)
        return self._answer_from_rows("what do you see", rows, rows, labels)

    def _all_memory_rows(self) -> list[dict]:
        with self._lock:
            frame_idx = self.state.frame_idx
        return [self._track_row(t, frame_idx) for t in self.memory.tracks.values()]

    def _locate_answer(self, target: str) -> str:
        rows = [self._track_row(t, self.state.frame_idx) for t in self._current_tracks if t.visible]
        matching = self._filter_rows(rows, target)
        if matching:
            dirs = ", ".join(f"{r['id']} at {r['location']}" for r in matching[:6])
            return f"Showing only {target}. Found {len(matching)} match(es): {dirs}."
        return f"{target} not present in this image."

    @classmethod
    def _answer_from_rows(cls, question: str, rows: list[dict], memory_rows: list[dict], labels: list[str]) -> str:
        if not rows:
            if memory_rows:
                return "No object is visible in the current image, but I still have earlier detections in memory."
            return "I do not have any detections in memory yet."

        visible = [r for r in rows if r.get("status") == "visible"]
        recent = visible or rows
        q = question or "what do you see"
        target = cls._extract_locate_target(q, memory_rows or recent)
        direction = cls._extract_direction(q)

        if "last" in q or "remember" in q or "memory" in q:
            source = memory_rows or recent
            names = ", ".join(f"{r['id']} at {r['location']} ({r['status']})" for r in source[:10])
            return f"Memory currently contains: {names}."

        if "where" in q or "locate" in q or "find" in q:
            search_rows = cls._filter_rows(recent, target) if target else recent
            if direction:
                search_rows = [r for r in search_rows if direction in r["location"].lower()]
            if target and not cls._filter_rows(recent, target):
                return f"{target} not present in this image."
            if target and direction and not search_rows:
                return f"{target} is not present on the {direction} side in this image."
            for row in search_rows:
                if not target or row["class"].lower() == target or row["id"].lower() == target:
                    return f"{row['id']} is at {row['location']} with confidence {row['confidence']}."
            if labels:
                return f"I can locate these visible classes: {', '.join(labels)}. Ask for one by name."
            return "No visible object can be located right now."

        if "how many" in q or "count" in q:
            if target:
                filtered = cls._filter_rows(recent, target)
                if direction:
                    filtered = [r for r in filtered if direction in r["location"].lower()]
                side = f" on the {direction} side" if direction else ""
                return f"I count {len(filtered)} visible {target} object(s){side}."
            counts: dict[str, int] = {}
            for row in recent:
                counts[row["class"]] = counts.get(row["class"], 0) + 1
            return "Counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "."

        counts: dict[str, int] = {}
        for row in recent:
            counts[row["class"]] = counts.get(row["class"], 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        focus = recent[0]
        return f"I see {summary}. Current focus is {focus['id']} at {focus['location']}."

    @staticmethod
    def _extract_locate_target(question: str, rows: list[dict]) -> str:
        if not question:
            return ""
        for row in sorted(rows, key=lambda r: len(r.get("class", "")), reverse=True):
            label = row.get("class", "").lower()
            obj_id = row.get("id", "").lower()
            if label and label in question:
                return label
            if obj_id and obj_id in question:
                return obj_id
        return ""

    @staticmethod
    def _extract_direction(question: str) -> str:
        for direction in ("left", "right", "top", "bottom", "center", "middle"):
            if direction in question:
                return "center" if direction == "middle" else direction
        return ""

    @staticmethod
    def _filter_rows(rows: list[dict], target: str) -> list[dict]:
        if not target:
            return rows
        q = target.lower()
        return [r for r in rows if r.get("class", "").lower() == q or r.get("id", "").lower() == q]

    @staticmethod
    def _asks_history(question: str) -> bool:
        return any(word in question for word in ("past", "previous", "earlier", "first image", "second image", "third image", "uploaded image", "old image"))

    def _answer_history(self, question: str) -> str:
        if not self._image_history:
            return "No past image memory is available yet."
        target = self._extract_locate_target(question, self._all_memory_rows())
        direction = self._extract_direction(question)
        selected = self._select_history(question)
        parts = []
        for item in selected:
            rows = item["rows"]
            if target:
                rows = self._filter_rows(rows, target)
            if direction:
                rows = [r for r in rows if direction in r["location"].lower()]
            if not rows:
                label = target or "requested object"
                side = f" on the {direction} side" if direction else ""
                parts.append(f"image {item['index']}: {label} not present{side}")
                continue
            counts: dict[str, int] = {}
            for row in rows:
                counts[row["class"]] = counts.get(row["class"], 0) + 1
            summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
            locs = ", ".join(f"{r['id']} at {r['location']}" for r in rows[:5])
            parts.append(f"image {item['index']}: {summary} ({locs})")
        return "Past image memory: " + "; ".join(parts) + "."

    def _select_history(self, question: str) -> list[dict]:
        if "previous" in question or "earlier" in question or "old image" in question:
            return self._image_history[-2:-1] or self._image_history[-1:]
        ordinals = {"first image": 1, "second image": 2, "third image": 3}
        for phrase, idx in ordinals.items():
            if phrase in question:
                return [item for item in self._image_history if item["index"] == idx] or self._image_history[-1:]
        return self._image_history[-5:]

    @staticmethod
    def _resize_max(frame: np.ndarray, max_w: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame
        scale = max_w / float(w)
        return cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _track_row(track: Track, frame_idx: int) -> dict:
        return {
            "id": track.name,
            "class": track.label,
            "status": "visible" if track.visible else "lost",
            "confidence": f"{track.conf:.2f}",
            "location": track.last_location,
            "last_seen": "now" if track.visible else f"{track.age_missing(frame_idx)} frames ago",
        }
