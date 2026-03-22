"""
download_face_model.py
======================
Downloads the OpenCV Caffe face detector files needed for this project.
Run once before starting the app: python download_face_model.py
"""
import os, urllib.request

os.makedirs("models", exist_ok=True)

FILES = {
    "models/deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "models/res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}

for path, url in FILES.items():
    if os.path.exists(path):
        print(f"[SKIP] {path} already exists")
        continue
    print(f"[DOWNLOAD] {path} ...")
    urllib.request.urlretrieve(url, path)
    print(f"[OK] {path}")

print("\nAll face-detector files ready.")
print("Now copy your mask_detector.model into the models/ folder and run: python app.py")
