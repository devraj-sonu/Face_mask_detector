"""
Face Mask Detection - Flask Backend
=====================================
Updated to work with mask_detector_saved (SavedModel format)
- Input size: 100x100
- Normalization: /255
- Output: [mask_prob, no_mask_prob]  (index 0 = Mask, index 1 = No Mask)
"""

import os
import json
import base64
import datetime
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
MODEL_PATH       = "models/mask_detector_saved"  # SavedModel folder
MODEL_INPUT_SIZE = (100, 100)                     # must match Colab IMG_SIZE
PREPROCESS_MODE  = "div255"                       # /255 normalization

PROTOTXT            = "models/deploy.prototxt"
CAFFE_MODEL         = "models/res10_300x300_ssd_iter_140000.caffemodel"
FACE_CONF_THRESHOLD = 0.3
ROI_PADDING         = 0.20
LOG_FILE            = "logs/detections.json"

# ══════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════
mask_model   = None
face_net     = None

def load_models():
    global mask_model, face_net

    # Load mask detector
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading mask model: {MODEL_PATH}")
        mask_model = tf.keras.layers.TFSMLayer(
            MODEL_PATH, call_endpoint='serving_default'
        )
        # Warm up with dummy prediction to confirm it works
        dummy   = np.zeros((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3), dtype="float32")
        out     = mask_model(dummy)
        key     = list(out.keys())[0]
        probs   = out[key].numpy()[0]
        print(f"[INFO] Model loaded ✓")
        print(f"[INFO] Output key: '{key}'  |  Dummy output: {probs}")
        print(f"[INFO] index 0 = Mask ({probs[0]:.3f})  |  index 1 = No Mask ({probs[1]:.3f})")
    else:
        print(f"[WARN] {MODEL_PATH} not found — running in DEMO mode")
        print(f"[WARN] Place mask_detector_saved folder inside models/")

    # Load OpenCV DNN face detector
    if os.path.exists(PROTOTXT) and os.path.exists(CAFFE_MODEL):
        face_net = cv2.dnn.readNet(PROTOTXT, CAFFE_MODEL)
        print("[INFO] OpenCV DNN face detector loaded ✓")
    else:
        print("[WARN] DNN face model not found — using Haar cascade fallback")

load_models()

# Haar cascade fallback
haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ══════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════
def preprocess_face(face_bgr):
    face_rgb     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, MODEL_INPUT_SIZE)
    face_array   = face_resized.astype("float32") / 255.0
    return np.expand_dims(face_array, axis=0)  # (1, 100, 100, 3)

# ══════════════════════════════════════════════════════
# INTERPRET OUTPUT
# ══════════════════════════════════════════════════════
def get_label_and_confidence(probs):
    mask_prob    = float(probs[0])   # index 0 = Mask
    no_mask_prob = float(probs[1])   # index 1 = No Mask
    if mask_prob > no_mask_prob:
        return "Mask", mask_prob
    else:
        return "No Mask", no_mask_prob

# ══════════════════════════════════════════════════════
# FACE ROI WITH PADDING
# ══════════════════════════════════════════════════════
def get_padded_roi(frame, x1, y1, x2, y2):
    h, w  = frame.shape[:2]
    pad_x = int((x2 - x1) * ROI_PADDING)
    pad_y = int((y2 - y1) * ROI_PADDING)
    return frame[max(0, y1-pad_y):min(h, y2+pad_y),
                 max(0, x1-pad_x):min(w, x2+pad_x)]

# ══════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════
def log_detection(results):
    os.makedirs("logs", exist_ok=True)
    entry = {"timestamp": datetime.datetime.now().isoformat(), "detections": results}
    logs  = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE) as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    logs = logs[-500:]
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

# ══════════════════════════════════════════════════════
# CORE DETECTION PIPELINE
# ══════════════════════════════════════════════════════
def detect_and_predict(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, []

    h, w  = frame.shape[:2]
    faces = []

    # ── Detect faces ──────────────────────────────────
    if face_net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        face_net.setInput(blob)
        detections = face_net.forward()
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > FACE_CONF_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                if (x2-x1) > 20 and (y2-y1) > 20:
                    faces.append((x1, y1, x2, y2))
    else:
        gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = haar_cascade.detectMultiScale(gray, 1.05, 3, minSize=(50, 50))
        for (x, y, fw, fh) in found:
            faces.append((x, y, x+fw, y+fh))

    print(f"[DEBUG] Faces detected: {len(faces)}")
    predictions = []

    # ── Classify each face ────────────────────────────
    for (x1, y1, x2, y2) in faces:
        face_roi = get_padded_roi(frame, x1, y1, x2, y2)
        if face_roi.size == 0:
            continue

        if mask_model is not None:
            face_input = preprocess_face(face_roi)
            raw_output = mask_model(face_input)
            # TFSMLayer returns a dict — extract the probability array
            out_key    = list(raw_output.keys())[0]
            probs      = raw_output[out_key].numpy()[0]
            label, confidence = get_label_and_confidence(probs)
            print(f"[DEBUG] {label} — mask:{probs[0]:.3f} no_mask:{probs[1]:.3f}")
        else:
            import random
            label      = random.choice(["Mask", "No Mask"])
            confidence = random.uniform(0.75, 0.98)

        # Draw box + label
        color       = (0, 200, 80) if label == "Mask" else (0, 50, 230)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text        = f"{label}: {confidence*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-th-14), (x1+tw+8, y1), color, -1)
        cv2.putText(frame, text, (x1+4, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        predictions.append({
            "label":      label,
            "confidence": round(confidence * 100, 2),
            "box":        {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
        })

    _, buf  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64 = base64.b64encode(buf).decode("utf-8")
    return img_b64, predictions

# ══════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/detect", methods=["POST"])
def detect():
    image_bytes = None
    if "image" in request.files:
        image_bytes = request.files["image"].read()
    elif request.is_json:
        data = request.get_json()
        b64  = data.get("image", "")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        image_bytes = base64.b64decode(b64)
    else:
        return jsonify({"success": False, "error": "No image provided"}), 400

    annotated_b64, predictions = detect_and_predict(image_bytes)
    if annotated_b64 is None:
        return jsonify({"success": False, "error": "Could not decode image"}), 400

    log_detection(predictions)
    no_mask_count = sum(1 for p in predictions if p["label"] == "No Mask")

    return jsonify({
        "success":         True,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "predictions":     predictions,
        "face_count":      len(predictions),
        "no_mask_alert":   no_mask_count > 0
    })

@app.route("/api/logs", methods=["GET"])
def get_logs():
    n = int(request.args.get("n", 50))
    if not os.path.exists(LOG_FILE):
        return jsonify([])
    with open(LOG_FILE) as f:
        logs = json.load(f)
    return jsonify(logs[-n:])

@app.route("/api/stats", methods=["GET"])
def get_stats():
    if not os.path.exists(LOG_FILE):
        return jsonify({"total": 0, "mask": 0, "no_mask": 0, "recent": []})
    with open(LOG_FILE) as f:
        logs = json.load(f)
    total = mask_count = no_mask_count = 0
    hourly = {}
    for entry in logs:
        for d in entry["detections"]:
            total += 1
            if d["label"] == "Mask":
                mask_count += 1
            else:
                no_mask_count += 1
        hour = entry["timestamp"][:13]
        hourly[hour] = hourly.get(hour, 0) + len(entry["detections"])
    recent_hours = sorted(hourly.items())[-12:]
    return jsonify({
        "total":   total,
        "mask":    mask_count,
        "no_mask": no_mask_count,
        "recent":  [{"hour": h, "count": c} for h, c in recent_hours]
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)