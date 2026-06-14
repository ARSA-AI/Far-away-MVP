from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def main() -> None:
    out_path = Path("demo_assets/synthetic_demo.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 960, 540
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 18, (w, h))
    for i in range(220):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (245, 238, 226)
        cv2.rectangle(frame, (0, 360), (w, h), (65, 70, 74), -1)
        cv2.line(frame, (0, 450), (w, 450), (220, 220, 220), 3)
        x = 40 + int(i * 3.2) % 780
        cv2.rectangle(frame, (x, 300), (x + 120, 370), (30, 60, 210), -1)
        cv2.circle(frame, (x + 25, 372), 18, (20, 20, 20), -1)
        cv2.circle(frame, (x + 95, 372), 18, (20, 20, 20), -1)
        px = 760 - int(i * 1.9) % 620
        cv2.circle(frame, (px, 292), 18, (70, 70, 70), -1)
        cv2.line(frame, (px, 310), (px, 365), (70, 70, 70), 6)
        cv2.line(frame, (px - 25, 330), (px + 25, 330), (70, 70, 70), 5)
        cv2.line(frame, (px, 365), (px - 18, 410), (70, 70, 70), 5)
        cv2.line(frame, (px, 365), (px + 18, 410), (70, 70, 70), 5)
        cv2.rectangle(frame, (120, 390), (185, 455), (20, 20, 170), -1)
        cv2.putText(frame, "GPPE V1 synthetic demo", (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 20, 50), 2)
        writer.write(frame)
    writer.release()
    print(out_path)


if __name__ == "__main__":
    main()

