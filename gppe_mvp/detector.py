from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import CONF_THRESHOLD, DETECT_IMGSZ, MODEL_PATH


@dataclass
class Detection:
    xyxy: tuple[int, int, int, int]
    cls_id: int
    label: str
    conf: float


class YoloDetector:
    """Small YOLO backend used as a replaceable detector engine."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        imgsz: int = DETECT_IMGSZ,
        conf: float = CONF_THRESHOLD,
    ) -> None:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.model = YOLO(str(self.model_path))
        self.names = self.model.names
        self._warmup()

    def _warmup(self) -> None:
        """Pay first-inference setup cost before the live demo starts."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.imgsz, conf=self.conf, verbose=False, device="cpu")

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
            device="cpu",
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None:
            return []

        detections: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        h, w = frame_bgr.shape[:2]
        for box, cls_id, score in zip(xyxy, cls, conf):
            x1, y1, x2, y2 = box.astype(int).tolist()
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            label = str(self.names.get(int(cls_id), cls_id))
            detections.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    cls_id=int(cls_id),
                    label=label,
                    conf=float(score),
                )
            )
        return detections


def labels_for(detections: Iterable[Detection]) -> list[str]:
    return sorted({d.label for d in detections})
