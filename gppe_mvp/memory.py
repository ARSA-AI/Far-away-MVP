from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import IOU_MATCH_THRESHOLD, MAX_MEMORY_OBJECTS, MEMORY_TTL_FRAMES
from .detector import Detection


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter + 1e-6)


def center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax = (a[0] + a[2]) * 0.5
    ay = (a[1] + a[3]) * 0.5
    bx = (b[0] + b[2]) * 0.5
    by = (b[1] + b[3]) * 0.5
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def location_name(box: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> str:
    h, w = frame_shape[:2]
    cx = (box[0] + box[2]) * 0.5 / max(1, w)
    cy = (box[1] + box[3]) * 0.5 / max(1, h)
    horiz = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
    vert = "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle"
    return f"{vert}-{horiz}"


@dataclass
class Track:
    track_id: int
    label: str
    conf: float
    box: tuple[int, int, int, int]
    first_frame: int
    last_seen_frame: int
    last_location: str
    visible: bool = True
    crop: Optional[np.ndarray] = None
    trail: list[tuple[int, int]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.label}_{self.track_id:02d}"

    def age_missing(self, frame_idx: int) -> int:
        return frame_idx - self.last_seen_frame


class ObjectMemory:
    def __init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, frame: np.ndarray, detections: list[Detection], frame_idx: int) -> list[Track]:
        unmatched = set(range(len(detections)))
        used_tracks: set[int] = set()

        for track_id, track in list(self.tracks.items()):
            best_idx = None
            best_score = -1.0
            for idx in unmatched:
                det = detections[idx]
                if det.label != track.label:
                    continue
                overlap = iou(track.box, det.xyxy)
                dist = center_distance(track.box, det.xyxy)
                diag = (frame.shape[0] ** 2 + frame.shape[1] ** 2) ** 0.5
                score = overlap - 0.15 * (dist / max(1.0, diag))
                if overlap >= IOU_MATCH_THRESHOLD or dist < 80:
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            if best_idx is not None:
                det = detections[best_idx]
                unmatched.remove(best_idx)
                used_tracks.add(track_id)
                self._update_track(track, det, frame, frame_idx)

        for idx in unmatched:
            det = detections[idx]
            track = Track(
                track_id=self.next_id,
                label=det.label,
                conf=det.conf,
                box=det.xyxy,
                first_frame=frame_idx,
                last_seen_frame=frame_idx,
                last_location=location_name(det.xyxy, frame.shape),
                crop=self._crop(frame, det.xyxy),
            )
            track.trail.append(self._center(det.xyxy))
            self.tracks[track.track_id] = track
            used_tracks.add(track.track_id)
            self.next_id += 1

        for track_id, track in list(self.tracks.items()):
            if track_id not in used_tracks:
                track.visible = False
            if track.age_missing(frame_idx) > MEMORY_TTL_FRAMES:
                del self.tracks[track_id]

        if len(self.tracks) > MAX_MEMORY_OBJECTS:
            oldest = sorted(self.tracks.values(), key=lambda t: t.last_seen_frame)
            for track in oldest[: len(self.tracks) - MAX_MEMORY_OBJECTS]:
                self.tracks.pop(track.track_id, None)

        return list(self.tracks.values())

    def _update_track(self, track: Track, det: Detection, frame: np.ndarray, frame_idx: int) -> None:
        track.box = det.xyxy
        track.conf = det.conf
        track.visible = True
        track.last_seen_frame = frame_idx
        track.last_location = location_name(det.xyxy, frame.shape)
        track.crop = self._crop(frame, det.xyxy)
        track.trail.append(self._center(det.xyxy))
        track.trail = track.trail[-24:]

    @staticmethod
    def _center(box: tuple[int, int, int, int]) -> tuple[int, int]:
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    @staticmethod
    def _crop(frame: np.ndarray, box: tuple[int, int, int, int], pad: int = 16) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return cv2.resize(crop, (220, 160), interpolation=cv2.INTER_AREA)

    def query(self, text: str) -> list[Track]:
        q = text.strip().lower()
        if not q:
            return []
        return [t for t in self.tracks.values() if q in t.label.lower() or q in t.name.lower()]

