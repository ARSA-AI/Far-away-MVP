from __future__ import annotations

import cv2
import numpy as np

from .config import FOCUS_CLASSES
from .memory import Track


def choose_focus(tracks: list[Track], query: str = "") -> tuple[Track | None, str]:
    visible = [t for t in tracks if t.visible]
    if not visible:
        return None, "no visible object"

    q = query.strip().lower()
    if q:
        matches = [t for t in visible if q in t.label.lower() or q in t.name.lower()]
        if matches:
            return max(matches, key=lambda t: t.conf), f"user query: {query}"

    def priority(t: Track) -> float:
        x1, y1, x2, y2 = t.box
        area = max(1, (x2 - x1) * (y2 - y1))
        small_object_bonus = 1.0 / (area ** 0.25)
        return FOCUS_CLASSES.get(t.label, 1.0) + 2.0 * t.conf + small_object_bonus

    return max(visible, key=priority), "automatic priority"


def make_foveal_crop(frame: np.ndarray, track: Track | None, scale: int = 2) -> np.ndarray:
    if track is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = track.box
    bw, bh = x2 - x1, y2 - y1
    pad_x = max(24, int(bw * 0.45))
    pad_y = max(24, int(bh * 0.45))
    x1 = max(0, x1 - pad_x)
    x2 = min(w, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    out_w = min(420, max(240, crop.shape[1] * scale))
    out_h = min(300, max(180, crop.shape[0] * scale))
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

