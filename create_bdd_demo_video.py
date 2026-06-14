from __future__ import annotations

import glob
from pathlib import Path

import cv2


BDD_ROOT = "/home/manish/Desktop/Prototype/unified_perception/dataset/bdd100k"


def main() -> None:
    paths = glob.glob(f"{BDD_ROOT}/**/*.jpg", recursive=True)
    if not paths:
        raise SystemExit(f"No BDD100K jpg images found under {BDD_ROOT}")

    out_path = Path("demo_assets/bdd_road_demo.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 960, 540
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (width, height))

    written = 0
    for path in paths[:240]:
        frame = cv2.imread(path)
        if frame is None:
            continue
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
        if written >= 96:
            break

    writer.release()
    if written == 0:
        raise SystemExit("No readable frames were written")
    print(f"{out_path} ({written} frames)")


if __name__ == "__main__":
    main()

