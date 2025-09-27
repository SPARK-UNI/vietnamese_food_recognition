# -*- coding: utf-8 -*-
"""
Flask inference app for CNN model
- Reads protocol.json to sync IMG_SIZE, color space, normalization, and class order.
"""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model


def load_artifacts(artifacts_dir: Path):
    protocol_path = artifacts_dir / "protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(f"protocol.json not found in {artifacts_dir}")

    with protocol_path.open("r", encoding="utf-8") as f:
        protocol = json.load(f)

    model_path = artifacts_dir / protocol.get("model_file", "food_cnn.h5")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("🔄 Loading model:", model_path)
    model = load_model(str(model_path), compile=False)
    print("✅ Model loaded!")

    labels_txt = artifacts_dir / "labels.txt"
    if labels_txt.exists():
        with labels_txt.open("r", encoding="utf-8") as f:
            labels_txt_list = [ln.strip() for ln in f.readlines() if ln.strip()]
    else:
        labels_txt_list = None

    class_names = protocol.get("class_names", [])
    labels = class_names if class_names else (labels_txt_list or [])

    settings = {
        "IMG_SIZE": tuple(protocol.get("img_size", [224, 224])),
        "COLOR_SPACE": protocol.get("color_space", "RGB"),
        "RESCALING_IN_MODEL": bool(protocol.get("rescaling_in_model", True)),
    }
    print(f"✅ Labels: {labels}")
    print(f"✅ Settings: {settings}")
    return model, labels, settings


def decode_base64_image(image_data: str) -> np.ndarray:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame_bgr


def make_app(model, labels, settings):
    app = Flask(__name__, static_folder="static", template_folder="templates")

    IMG_SIZE = settings["IMG_SIZE"]
    COLOR_SPACE = settings["COLOR_SPACE"]
    RESCALING_IN_MODEL = settings["RESCALING_IN_MODEL"]

    def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None:
            return None
        if COLOR_SPACE.upper() == "RGB":
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame = frame_bgr
        img = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img = img.astype("float32")
        if not RESCALING_IN_MODEL:
            img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict_ndarray(image_bgr: np.ndarray) -> str:
        x = preprocess(image_bgr)
        if x is None:
            return "Invalid image"
        pred = model.predict(x, verbose=0)
        class_idx = int(np.argmax(pred, axis=1)[0])
        confidence = float(np.max(pred))
        if 0 <= class_idx < len(labels):
            return f"{labels[class_idx]} ({confidence:.2f})"
        return f"Unknown ({confidence:.2f})"

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/labels")
    def get_labels():
        return jsonify({"labels": labels})

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"prediction": "No image provided", "error": True})
        frame_bgr = decode_base64_image(data["image"])
        result = predict_ndarray(frame_bgr)
        return jsonify({"prediction": result, "error": False})

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=str, default="./artifacts",
                        help="Folder containing food_cnn.h5, labels.txt, protocol.json")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    model, labels, settings = load_artifacts(artifacts_dir)
    app = make_app(model, labels, settings)

    print("🚀 Starting API…")
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
