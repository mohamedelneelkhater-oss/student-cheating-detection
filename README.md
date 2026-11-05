# 🚨 Cheating Tracking and Alert System

An end-to-end AI system that **tracks students in exam videos, detects cheating behaviour, turns the
student's bounding box red, captures a screenshot, stores the incident in a database and raises an
instant alert** - available as a **Tkinter desktop app**, a **Streamlit web app** and a **CLI**.

> Red box = cheating &nbsp;|&nbsp; Green box = normal behaviour

## ✨ Features

| Requirement | Implementation |
|---|---|
| Detect cheating behaviour | YOLOv8 vs YOLO11 trained on a Roboflow dataset, best model kept |
| Track students | ByteTrack (Ultralytics) - persistent student IDs |
| Red / green boxes | `src/detector.py` - cheating classes are drawn in red, others in green |
| Screenshot on cheating | Saved to `alerts/screenshots/` the moment an alert is raised |
| Database | SQLite (`src/database.py`) + optional Firebase Firestore / Storage (`src/firebase_client.py`) |
| Desktop app (Tkinter) | `desktop_app.py` - Upload Video, live preview, alert history, screenshot viewer, app icon |
| Notifications | Instant on-screen toast: behaviour, time, "Screenshot saved successfully." |
| Web demo | `app.py` (Streamlit) - upload a file or paste a video URL |

## 🧠 How alerts work

1. Every frame is detected and tracked (each student keeps an ID).
2. A student is flagged only if a cheating class persists for several consecutive frames
   (`min_consecutive_frames`) - this removes single-frame false positives.
3. A cooldown per student (`cooldown_seconds`) prevents alert spam.
4. When an alert fires: screenshot -> database record -> (optional) Firebase -> notification.

## 📊 Dataset

Roboflow project `student-cheating-detection` (YOLO format). Classes: {{CLASSES}}

## 🏆 Model comparison (YOLOv8 vs YOLO11)

Selected model: **{{BEST_MODEL}}**

{{RESULTS_TABLE}}

![comparison](assets/model_comparison.png)
![curves](assets/training_curves.png)

## 🖥 Desktop app

{{DESKTOP_SHOT}}

## 🗂 Project structure

```
.
├── app.py                 # Streamlit web app
├── desktop_app.py         # Tkinter desktop app
├── main.py                # Command line interface
├── requirements.txt
├── models/best.pt         # Best trained weights
├── src/
│   ├── detector.py        # Detection + tracking + alert logic
│   ├── database.py        # SQLite alert storage
│   └── firebase_client.py # Optional Firebase sync
├── samples/               # Sample input / annotated videos
├── assets/                # Charts, icon, result images
└── tests/
```

## 🚀 Quick start

```bash
git clone https://github.com/<your-username>/cheating-detection-system.git
cd cheating-detection-system
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Desktop app (Tkinter)**
```bash
python desktop_app.py
```

**Web app (Streamlit)**
```bash
streamlit run app.py
```

**CLI**
```bash
python main.py --video samples/demo_exam.mp4 --output outputs/annotated.mp4
```

**Tests**
```bash
pip install pytest && pytest -q
```

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (keep `models/best.pt` in the repo; it must be below 100 MB).
2. Open <https://share.streamlit.io>, click **New app**, choose the repo, branch `main`, file `app.py`.
3. In *Advanced settings* choose Python 3.11 and deploy.

## 🔥 Optional: Firebase

Create a Firebase project, enable Firestore and Storage, download a service-account JSON and run:

```bash
export FIREBASE_CREDENTIALS=/path/to/serviceAccount.json
export FIREBASE_STORAGE_BUCKET=your-project.appspot.com
pip install firebase-admin
python main.py --video samples/demo_exam.mp4
```

Never commit the service-account file (it is listed in `.gitignore`).

## ⚠️ Limitations and ideas

- The demo video may be synthetic (built from dataset images) if no real exam video was available; use a real exam recording for a real evaluation.
- Models trained on classroom-camera images can underperform on single-webcam proctoring videos (domain gap).
- Accuracy depends on camera angle and dataset size; collect more data for new classrooms.
- Future work: multi-camera support, face-blurring for privacy, e-mail/SMS notifications, model export to ONNX/TensorRT.

## 📚 Data sources and credits

- Training data: Roboflow Universe project `student-cheating-detection` (workspace `mohamed-elneel`).
- Real exam videos for demos (check each license before redistributing):
  - Online Exam Proctoring (OEP) dataset on Kaggle: <https://www.kaggle.com/datasets/raajanwankhade/oep-dataset>
  - Cheating Scenario Dataset in Online Exam (Mendeley Data): <https://data.mendeley.com/datasets/mjrfmvsh7d/1>
  - Students suspicious behaviors dataset (Mendeley Data): <https://data.mendeley.com/datasets/39xs8th543/1>
- Extra Roboflow datasets with clearly named classes:
  <https://universe.roboflow.com/dnn317/exam_cheating_detection>,
  <https://universe.roboflow.com/mahmoud-mohamed-phhz1/cheat-0ogbv>

## 📄 License

MIT
