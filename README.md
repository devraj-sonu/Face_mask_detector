<<<<<<< HEAD
# ⬡ MaskGuard AI — Face Mask Detection Web App

> **College Project** · TensorFlow + OpenCV + Flask  
> Detects faces in images/webcam, classifies mask usage, shows confidence scores with bounding boxes.

---

## 📁 Project Structure

```
face-mask-detector/
├── app.py                      ← Flask backend (main server)
├── download_face_model.py      ← One-time setup: downloads OpenCV face detector
├── requirements.txt
├── Procfile                    ← For Render / Railway deployment
├── render.yaml                 ← One-click Render deploy config
├── .gitignore
├── models/
│   ├── mask_detector.model     ← YOUR trained Keras model (copy here)
│   ├── deploy.prototxt         ← Auto-downloaded by setup script
│   └── res10_300x300_ssd_iter_140000.caffemodel  ← Auto-downloaded
├── logs/
│   └── detections.json         ← Auto-created, stores detection history
├── static/
│   ├── css/style.css
│   └── js/app.js
└── templates/
    └── index.html
```

---

## 🚀 Run Locally (Step-by-Step)

### 1. Clone / unzip the project
```bash
cd face-mask-detector
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Copy your trained model
```bash
cp /path/to/your/mask_detector.model  models/mask_detector.model
```

### 5. Download OpenCV face detector (run once)
```bash
python download_face_model.py
```

### 6. Start the server
```bash
python app.py
```

Open your browser → **http://localhost:5000**

> **No model file?** The app runs in *DEMO mode* with random predictions so the UI still works.

---

## 🔌 API Reference

| Method | Endpoint        | Description                          |
|--------|-----------------|--------------------------------------|
| POST   | `/api/detect`   | Send image → get predictions         |
| GET    | `/api/logs`     | Recent detection log (`?n=50`)       |
| GET    | `/api/stats`    | Aggregated dashboard statistics      |

### POST /api/detect
**Option A – Form upload:**
```bash
curl -X POST -F "image=@photo.jpg" http://localhost:5000/api/detect
```

**Option B – Base64 JSON:**
```json
{ "image": "data:image/jpeg;base64,/9j/4AAQ..." }
```

**Response:**
```json
{
  "success": true,
  "annotated_image": "data:image/jpeg;base64,...",
  "predictions": [
    { "label": "Mask", "confidence": 97.3, "box": {"x1":50,"y1":30,"x2":180,"y2":200} }
  ],
  "face_count": 1,
  "no_mask_alert": false
}
```

---

## 🌐 Deploy for Free

### Option 1: Render (recommended)
1. Push project to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo — Render will detect `render.yaml` automatically
4. Upload `mask_detector.model` via Render's *Disk* or Environment settings
5. Deploy!

### Option 2: Railway
1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Set build command: `pip install -r requirements.txt && python download_face_model.py`
4. Set start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

### ⚠️ Model file on free tiers
Free tiers have ephemeral storage. Options:
- Store your model in **GitHub LFS** (up to 1 GB free)
- Upload to **Google Drive** and download at startup
- Use **Hugging Face Hub** model storage (free)

---

## ✨ Features

| Feature | Status |
|---------|--------|
| Upload image detection | ✅ |
| Webcam live detection | ✅ |
| Auto-detect every 2s | ✅ |
| Bounding boxes on faces | ✅ |
| Confidence percentage | ✅ |
| No-Mask alert banner | ✅ |
| Alert sound (Web Audio) | ✅ |
| Detection logs (JSON) | ✅ |
| Dashboard with chart | ✅ |
| Responsive design | ✅ |
| Dark mode UI | ✅ |

---

## 💡 Innovation Ideas (to impress evaluators)

1. **Multi-face tracking** — Track face IDs across webcam frames
2. **Attendance integration** — Log mask-compliant entries with timestamp
3. **Email/SMS alerts** — Send notification when no-mask detected (Twilio/SMTP)
4. **Export logs as CSV/PDF** — Download reports
5. **Heatmap overlay** — Show which areas of frame had violations
6. **Model confidence threshold slider** — Let user tune sensitivity
7. **Multiple model comparison** — MobileNetV2 vs EfficientNet
8. **Progressive Web App (PWA)** — Install on mobile, works offline

---

## 🛠 Tech Stack

- **Backend**: Python 3.11, Flask 3.0, OpenCV, TensorFlow/Keras
- **Frontend**: HTML5, CSS3 (custom, no framework), Vanilla JS, Chart.js
- **Face Detector**: OpenCV DNN (ResNet SSD Caffe) with Haar fallback
- **Mask Classifier**: Your trained MobileNetV2 Keras model

---

## 👤 Author

College Project — Face Mask Detection using Deep Learning  
Built with ❤️ using TensorFlow + OpenCV + Flask
=======
# Face_mask_detector
>>>>>>> a217ea31d51ead4f02a15de0e38b6380a49c6cdb
