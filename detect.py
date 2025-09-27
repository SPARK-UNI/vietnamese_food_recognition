from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder="static", template_folder="templates")

# ================================
# Load model và labels
# ================================
MODEL_PATH = r"your_model_path"
LABELS_PATH = r"labels.txt" 
IMG_SIZE = (224, 224)

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
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return "Invalid image"

        img = cv2.resize(frame, IMG_SIZE)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        class_idx = np.argmax(pred, axis=1)[0]
        confidence = float(np.max(pred))

        if 0 <= class_idx < len(labels):
            result = labels[class_idx]
            return f"{result} ({confidence:.2f})"
        else:
            return f"Unknown ({confidence:.2f})"

    except Exception as e:
        return f"Error: {str(e)}"


# ================================
# Routes
# ================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or not labels:
        return jsonify({'prediction': 'Model not loaded', 'error': True})

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'prediction': 'No image provided', 'error': True})

    prediction = predict_image(data['image'])
    return jsonify({'prediction': prediction, 'error': False})


# ================================
# Main
# ================================
if __name__ == '__main__':
    print("🚀 Starting API...")
    app.run(debug=True, host='127.0.0.1', port=5000)
