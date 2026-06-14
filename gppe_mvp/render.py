from __future__ import annotations

import cv2
import numpy as np

from .memory import Track


COLORS = {
    "person": (40, 220, 80),
    "car": (30, 144, 255),
    "bus": (30, 144, 255),
    "truck": (30, 144, 255),
    "bottle": (255, 180, 40),
    "backpack": (255, 80, 200),
    "handbag": (255, 80, 200),
    "suitcase": (255, 80, 200),
}


def draw_overlay(frame: np.ndarray, tracks: list[Track], focus_id: int | None, query: str) -> np.ndarray:
    out = frame.copy()
    q = query.strip().lower()
    for track in tracks:
        if not track.visible:
            continue
        if q and q not in track.label.lower() and q not in track.name.lower():
            continue
        color = COLORS.get(track.label, (230, 230, 230))
        if focus_id == track.track_id:
            color = (0, 0, 255)
            thickness = 3
        else:
            thickness = 2
        x1, y1, x2, y2 = track.box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{track.name} {track.conf:.2f}"
        cv2.rectangle(out, (x1, max(0, y1 - 22)), (x1 + min(220, 9 * len(label)), y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        if len(track.trail) > 1:
            pts = np.array(track.trail, dtype=np.int32)
            cv2.polylines(out, [pts], False, color, 2)
    return out


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()
