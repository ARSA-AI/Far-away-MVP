from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE_OUTSIDE = Path("/home/manish/Downloads/logos/near_train].avif")
SOURCE_INTRUSION = Path("/home/manish/Downloads/logos/near__train.webp")
OUT = ROOT / "demo_assets" / "railguard_platform_intrusion_demo.mp4"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    outside = cv2.imread(str(SOURCE_OUTSIDE))
    intrusion = cv2.imread(str(SOURCE_INTRUSION))
    if outside is None:
        raise FileNotFoundError(f"Could not read source image: {SOURCE_OUTSIDE}")
    if intrusion is None:
        raise FileNotFoundError(f"Could not read source image: {SOURCE_INTRUSION}")
    outside = shift_up(letterbox(outside, 640, 360), 150)
    intrusion = letterbox(intrusion, 640, 360)
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), 12, (640, 360))
    for _ in range(24):
        writer.write(outside)
    for _ in range(84):
        writer.write(intrusion)
    writer.release()
    print(f"Wrote {OUT}")


def letterbox(frame, width: int, height: int):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    out = cv2.copyMakeBorder(
        resized,
        (height - resized.shape[0]) // 2,
        height - resized.shape[0] - (height - resized.shape[0]) // 2,
        (width - resized.shape[1]) // 2,
        width - resized.shape[1] - (width - resized.shape[1]) // 2,
        cv2.BORDER_CONSTANT,
        value=(8, 12, 24),
    )
    return out


def shift_up(frame, pixels: int):
    out = np.zeros_like(frame)
    out[:] = (8, 12, 24)
    pixels = max(0, min(frame.shape[0] - 1, pixels))
    out[: frame.shape[0] - pixels, :] = frame[pixels:, :]
    return out


if __name__ == "__main__":
    main()
