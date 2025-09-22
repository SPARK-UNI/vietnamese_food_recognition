from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ================================
# Load model và labels
# ================================
MODEL_PATH = "food_ann.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = (224, 224)  # Teachable Machine thường dùng 224x224

print("🔄 Loading model...")
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"✅ Labels loaded: {labels}")

except Exception as e:
    print(f"❌ Error loading model or labels: {e}")
    model = None
    labels = []


# ================================
# Hàm dự đoán
# ================================
def predict_image(image_data: str):
    try:
        # Xóa prefix base64 nếu có
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 → numpy
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return "Invalid image"

        # Resize & normalize
        img = cv2.resize(frame, IMG_SIZE)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)  # (1,224,224,3)

        # Predict
        pred = model.predict(img)
        class_idx = np.argmax(pred, axis=1)[0]
        confidence = float(np.max(pred))

        # Lấy nhãn
        if 0 <= class_idx < len(labels):
            result = labels[class_idx].split(' ', 1)[-1]
            return f"{result} ({confidence:.2f})"
        else:
            return f"Unknown ({confidence:.2f})"

    except Exception as e:
        return f"Error: {str(e)}"


# ================================
# Routes
# ================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def css():
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def js():
    return send_from_directory('.', 'script.js')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':  # CORS preflight
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    if model is None or not labels:
        response = jsonify({'prediction': 'Model not loaded', 'error': True})
    else:
        data = request.get_json()
        if not data or 'image' not in data:
            response = jsonify({'prediction': 'No image provided', 'error': True})
        else:
            prediction = predict_image(data['image'])
            response = jsonify({'prediction': prediction, 'error': False})

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# ================================
# Main
# ================================
if __name__ == '__main__':
    print("🚀 Starting API...")
    print("Server running at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
