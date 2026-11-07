"""
Exam Cheating Detection System - Streamlit UI
Sources: uploaded file, dataset videos (auto-discovered), direct URL, camera snapshot.
Detection: YOLO (phones / books / laptops / people) + MediaPipe head-turn analysis.
"""

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Optional AI libraries (loaded safely)
# ----------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Exam Cheating Detection System", page_icon="🎓", layout="wide")
st.title("🎓 Exam Cheating Detection & Monitoring System")
st.caption("YOLO object detection + head-turn analysis for real-time cheating alerts")
st.markdown("---")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_DATASET_DIRS = ["/kaggle/input", "samples", "data", "dataset"]


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(weights: str):
    """Load the requested YOLO weights, falling back to YOLOv8n if unavailable."""
    if not YOLO_AVAILABLE:
        return None
    for candidate in (weights, "yolov8n.pt"):
        try:
            return YOLO(candidate)
        except Exception:
            continue
    return None


# ----------------------------------------------------------------------------
# Dataset video discovery
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def find_dataset_videos(search_dirs: tuple) -> list:
    """Recursively collect video files from the given directories."""
    videos = []
    for directory in search_dirs:
        root = Path(directory.strip())
        if root.exists() and root.is_dir():
            for path in root.rglob("*"):
                if path.suffix.lower() in VIDEO_EXTENSIONS:
                    videos.append(str(path))
    return sorted(set(videos))


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------
class CheatingDetector:
    def __init__(self, yolo_model, conf_threshold=0.45, head_threshold=0.15, max_faces=5):
        self.model = yolo_model
        self.conf_threshold = conf_threshold
        self.head_threshold = head_threshold
        self.face_mesh = None

        if MEDIAPIPE_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=max_faces,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
            except Exception:
                self.face_mesh = None

    def estimate_head_pose(self, frame):
        """Return (status_message, is_alert) based on nose offset between the eyes."""
        status, alert = "Normal posture", False
        if self.face_mesh is None:
            return status, alert
        try:
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return "Face not visible or turned away completely", True

            for face in results.multi_face_landmarks:
                nose, left_eye, right_eye = face.landmark[1], face.landmark[33], face.landmark[263]
                eye_dist = abs(right_eye.x - left_eye.x)
                if eye_dist <= 0:
                    continue
                offset = (nose.x - (left_eye.x + right_eye.x) / 2) / eye_dist
                if offset < -self.head_threshold:
                    return "Student is looking LEFT (possible peeking at a neighbor)", True
                if offset > self.head_threshold:
                    return "Student is looking RIGHT (possible peeking at a neighbor)", True
        except Exception:
            pass
        return status, alert

    def process_frame(self, frame):
        """Run YOLO + head pose. Returns (frame, alerts, stats_dict)."""
        alerts = []
        stats = {"persons": 0, "phone": False, "material": False, "head_alert": False}

        if self.model is not None:
            try:
                for result in self.model(frame, verbose=False):
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf < self.conf_threshold:
                            continue
                        name = self.model.names[int(box.cls[0])]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        if name == "person" and (x2 - x1) * (y2 - y1) > 2500:
                            stats["persons"] += 1
                            self._draw(frame, (x1, y1, x2, y2), f"Person {conf:.2f}", (0, 255, 0), 2)
                        elif name in ("cell phone", "phone"):
                            stats["phone"] = True
                            self._draw(frame, (x1, y1, x2, y2), f"PHONE {conf:.2f}", (0, 0, 255), 3)
                        elif name in ("book", "laptop"):
                            stats["material"] = True
                            self._draw(frame, (x1, y1, x2, y2), f"{name.upper()} {conf:.2f}", (0, 165, 255), 3)
            except Exception:
                pass

        if stats["phone"]:
            alerts.append("🚨 Confirmed: mobile phone detected")
        if stats["material"]:
            alerts.append("📚 Confirmed: book / laptop / external material detected")

        head_msg, head_alert = self.estimate_head_pose(frame)
        stats["head_alert"] = head_alert
        stats["head_msg"] = head_msg
        if head_alert:
            alerts.append(f"👀 {head_msg}")

        if stats["persons"] > 1:
            alerts.append(f"👥 Group alert: {stats['persons']} people in camera view")

        return frame, alerts, stats

    @staticmethod
    def _draw(frame, box, label, color, thickness):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def play_alarm_sound():
    st.markdown(
        """<audio autoplay><source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3"
        type="audio/mpeg"></audio>""",
        unsafe_allow_html=True,
    )


def render_dashboard(stats, alerts, widgets, sound_on, last_sound):
    """Update live widgets. Returns the timestamp of the last alarm played."""
    with widgets["metrics"].container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("People detected", stats["persons"])
        c2.metric("Phone", "Detected 🚨" if stats["phone"] else "Clear ✅")
        c3.metric("Books / Laptop", "Detected 🚨" if stats["material"] else "Clear ✅")
        c4.metric("Head turn", "Alert 🚨" if stats["head_alert"] else "Normal ✅")

    with widgets["alerts"].container():
        if alerts:
            for a in alerts:
                st.error(a)
            if sound_on and time.time() - last_sound > 3:  # 3s cooldown
                play_alarm_sound()
                last_sound = time.time()
        else:
            st.success("✨ Status normal - no suspicious behavior detected.")
    return last_sound


def make_widgets():
    return {
        "frame": st.empty(),
        "progress": st.progress(0.0),
        "metrics": st.empty(),
        "alerts": st.empty(),
    }


def analyze_video(source, detector, frame_skip, sound_on):
    """Shared loop for uploaded files, dataset videos and URLs."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Could not open the video source. Check the file path or URL.")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    widgets = make_widgets()
    events, frame_idx, last_sound = [], 0, 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        frame, alerts, stats = detector.process_frame(frame)
        widgets["frame"].image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        if total > 0:
            widgets["progress"].progress(min(frame_idx / total, 1.0))
        last_sound = render_dashboard(stats, alerts, widgets, sound_on, last_sound)

        for a in alerts:
            events.append({
                "video_time_sec": round(frame_idx / fps, 2),
                "frame": frame_idx,
                "event": a,
                "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    cap.release()
    widgets["progress"].progress(1.0)
    st.info(f"Analysis finished - {frame_idx} frames read.")
    show_report(events)


def show_report(events):
    st.subheader("📋 Incident Report")
    if not events:
        st.success("No incidents were recorded.")
        return
    df = pd.DataFrame(events)
    c1, c2 = st.columns([1, 2])
    c1.metric("Total incidents", len(df))
    c1.dataframe(df["event"].value_counts().rename("count"))
    c2.dataframe(df, use_container_width=True, height=300)
    st.download_button(
        "⬇️ Download report (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"cheating_report_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
weights = st.sidebar.selectbox("YOLO weights", ["yolo11n.pt", "yolov8s.pt", "yolov8n.pt"])
conf_threshold = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.45, 0.05)
head_threshold = st.sidebar.slider("Head-turn sensitivity (lower = stricter)", 0.05, 0.40, 0.15, 0.01)
frame_skip = st.sidebar.slider("Process every Nth frame", 1, 10, 3)
sound_on = st.sidebar.checkbox("Alarm sound", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🎥 Video Source")
source_option = st.sidebar.selectbox(
    "Choose a source:",
    ("Upload from device", "Dataset videos (auto-discovered)", "Direct video URL", "Device camera"),
)

yolo_model = load_yolo_model(weights)
if yolo_model is None:
    st.sidebar.error("YOLO model unavailable - install `ultralytics` and check the weights file.")
if not MEDIAPIPE_AVAILABLE:
    st.sidebar.warning("MediaPipe not installed - head-turn analysis is disabled.")

detector = CheatingDetector(yolo_model, conf_threshold, head_threshold)

# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------
if source_option == "Upload from device":
    st.markdown("### 📁 Analyze an uploaded video")
    uploaded = st.file_uploader("Upload a video (MP4, AVI, MOV, MKV)", type=["mp4", "avi", "mov", "mkv"])
    if uploaded and st.button("▶️ Start analysis", type="primary"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            analyze_video(tmp_path, detector, frame_skip, sound_on)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

elif source_option == "Dataset videos (auto-discovered)":
    st.markdown("### 🗂️ Analyze a video from your dataset")
    dirs_text = st.sidebar.text_area(
        "Dataset folders (one per line)", value="\n".join(DEFAULT_DATASET_DIRS), height=110
    )
    search_dirs = tuple(d for d in dirs_text.splitlines() if d.strip())
    if st.sidebar.button("🔄 Rescan folders"):
        find_dataset_videos.clear()

    videos = find_dataset_videos(search_dirs)
    if not videos:
        st.warning("No videos found in the configured folders. Edit the folder list in the sidebar or upload a video.")
    else:
        st.success(f"Found {len(videos)} video(s).")
        query = st.text_input("Filter by name (optional)")
        filtered = [v for v in videos if query.lower() in v.lower()] if query else videos
        if filtered:
            selected = st.selectbox("Select a video:", filtered, format_func=lambda p: Path(p).name + "  —  " + p)
            with st.expander("Preview original video"):
                st.video(selected)
            if st.button("▶️ Start analysis", type="primary"):
                analyze_video(selected, detector, frame_skip, sound_on)
        else:
            st.info("No videos match your filter.")

elif source_option == "Direct video URL":
    st.markdown("### 🌐 Analyze a video from a direct URL")
    url = st.text_input("Direct link to a video file (e.g. .mp4):")
    if url and st.button("▶️ Start analysis", type="primary"):
        analyze_video(url, detector, frame_skip, sound_on)

else:
    st.markdown("### 📷 Device camera snapshot")
    snapshot = st.camera_input("Take a picture for instant verification")
    if snapshot is not None:
        image = cv2.imdecode(np.frombuffer(snapshot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        frame, alerts, stats = detector.process_frame(image)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        widgets = {"metrics": st.empty(), "alerts": st.empty()}
        render_dashboard(stats, alerts, widgets, sound_on, 0.0)
        if alerts:
            show_report([{
                "video_time_sec": 0, "frame": 1, "event": a,
                "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            } for a in alerts])
