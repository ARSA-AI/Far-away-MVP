from __future__ import annotations

from gppe_mvp.config import MODEL_PATH, MODELS_DIR


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"Model already exists: {MODEL_PATH}")
        return
    from ultralytics import YOLO

    print("Downloading YOLOv8n model through Ultralytics...")
    model = YOLO("yolov8n.pt")
    source = model.ckpt_path
    print(f"Downloaded source: {source}")
    import shutil

    shutil.copy2(source, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

