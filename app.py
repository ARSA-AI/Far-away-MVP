from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from gppe_mvp.config import RAILGUARD_DEMO_VIDEO, UPLOAD_DIR
from gppe_mvp.pipeline import GPPEMvpRuntime


app = Flask(__name__)
runtime = GPPEMvpRuntime()


def _make_placeholder(text: str) -> bytes:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:] = (12, 20, 44)
    cv2.putText(frame, "GPPE V1", (36, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (245, 245, 245), 3)
    cv2.putText(frame, text, (36, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (190, 205, 230), 2)
    cv2.putText(frame, "Upload image(s), video, or start webcam", (36, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 180, 255), 2)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return buf.tobytes() if ok else b""


PLACEHOLDER_MAIN = _make_placeholder("Global view waiting")
PLACEHOLDER_FOVEAL = _make_placeholder("Foveal crop waiting")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_webcam", methods=["POST"])
def start_webcam():
    runtime.start(0)
    return redirect(url_for("index"))


@app.route("/start_demo", methods=["POST"])
def start_demo():
    runtime.set_train_context(False)
    runtime.start(str(RAILGUARD_DEMO_VIDEO))
    return redirect(url_for("index"))


@app.route("/stop", methods=["POST"])
def stop():
    runtime.stop()
    return redirect(url_for("index"))


@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    runtime.clear_memory()
    return redirect(url_for("index"))


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or not file.filename:
        return redirect(url_for("index"))
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    path = UPLOAD_DIR / filename
    file.save(path)
    runtime.start(str(path))
    return redirect(url_for("index"))


@app.route("/upload_images", methods=["POST"])
def upload_images():
    files = request.files.getlist("images")
    if not files:
        return redirect(url_for("index"))
    image_dir = UPLOAD_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    stamp = int(time.time() * 1000)
    for idx, file in enumerate(files):
        if not file or not file.filename:
            continue
        filename = secure_filename(file.filename)
        path = image_dir / f"{stamp}_{idx}_{filename}"
        file.save(path)
        paths.append(path)
    runtime.process_images(paths)
    return redirect(url_for("index"))


@app.route("/set_query", methods=["POST"])
def set_query():
    runtime.set_query(request.form.get("query", ""))
    return redirect(url_for("index"))


@app.route("/ask", methods=["POST"])
def ask():
    runtime.answer_question(request.form.get("question", ""))
    return redirect(url_for("index"))


@app.route("/set_train_context", methods=["POST"])
def set_train_context():
    runtime.set_train_context(request.form.get("demo_train_context") == "on")
    return redirect(url_for("index"))


@app.route("/report", methods=["POST"])
def report():
    runtime.generate_report()
    return redirect(url_for("index"))


@app.route("/ack", methods=["POST"])
def ack():
    runtime.acknowledge_incident(request.form.get("event_id", ""))
    return redirect(url_for("index"))


@app.route("/video_feed")
def video_feed():
    return Response(_jpeg_stream("main"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/foveal_feed")
def foveal_feed():
    return Response(_jpeg_stream("foveal"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def state():
    s = runtime.snapshot()
    return jsonify(
        {
            "query": s.query,
            "source": s.source,
            "mode": s.mode,
            "fps": round(s.fps, 2),
            "frame_idx": s.frame_idx,
            "focus_reason": s.focus_reason,
            "focus_name": s.focus_name,
            "answer": s.answer,
            "labels": s.labels,
            "memory_rows": s.memory_rows,
            "incidents": s.incidents,
            "actions": s.actions,
            "tasks": s.tasks,
            "system_status": s.system_status,
            "compiled_rule": s.compiled_rule,
            "alert": s.alert,
            "report_url": s.report_url,
            "running": s.running,
            "error": s.error,
        }
    )


def _jpeg_stream(kind: str):
    while True:
        s = runtime.snapshot()
        data = s.foveal_jpeg if kind == "foveal" else s.last_frame_jpeg
        if not data:
            data = PLACEHOLDER_FOVEAL if kind == "foveal" else PLACEHOLDER_MAIN
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        time.sleep(0.08)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)
