from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
UPLOAD_DIR = ROOT / "static" / "uploads"
OUTPUT_DIR = ROOT / "static" / "outputs"
RAILGUARD_DEMO_VIDEO = ROOT / "demo_assets" / "Demo.mp4"

MODEL_NAME = "yolov8n.pt"
MODEL_PATH = MODELS_DIR / MODEL_NAME

DETECT_IMGSZ = 320
DETECT_EVERY_N_FRAMES = 2
CONF_THRESHOLD = 0.30
IOU_MATCH_THRESHOLD = 0.25
MEMORY_TTL_FRAMES = 90
MAX_MEMORY_OBJECTS = 64

FOCUS_CLASSES = {
    "person": 5.0,
    "car": 4.0,
    "truck": 4.0,
    "bus": 4.0,
    "motorcycle": 3.5,
    "bicycle": 3.5,
    "backpack": 3.0,
    "handbag": 3.0,
    "suitcase": 3.0,
    "bottle": 2.5,
    "cell phone": 2.5,
}
