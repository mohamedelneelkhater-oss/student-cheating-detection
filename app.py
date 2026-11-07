"""Streamlit UI for the Exam Cheating Detection System.

Run with:  streamlit run app.py
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

from src.detector import MEDIAPIPE_AVAILABLE, CheatingDetector, load_yolo_model
from src.video_utils import DEFAULT_DATASET_DIRS, find_videos

st.set_page_config(page_title="Exam Cheating Detection System", page_icon="🎓", layout="wide")
st.title("🎓 Exam Cheating Detection & Monitoring System")
st.caption("YOLO object detection + head-turn analysis for real-time cheating alerts")
st.markdown("---")


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource
def get_model(weights):
    return load_yolo_model(weights)


@st.cache_data(show_spinner=False)
def get_dataset_videos(search_dirs):
    return find_videos(search_dirs)


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
    """Update live widgets and return the time of the last alarm."""
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
            if sound_on and time.time() - last_sound > 3:  # 3-second cooldown
                play_alarm_sound()
                last_sound = time.time()
        else:
            st.success("✨ Status normal - no suspicious behavior detected.")
    return last_sound


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


def analyze_video(source, detector, frame_skip, sound_on):
    """Shared loop for uploaded files, dataset videos and URLs."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Could not open the video source. Check the file path or URL.")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_slot, progress = st.empty(), st.progress(0.0)
    widgets = {"metrics": st.empty(), "alerts": st.empty()}
    events, frame_idx, last_sound = [], 0, 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        frame, alerts, stats = detector.process_frame(frame)
        frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        if total > 0:
            progress.progress(min(frame_idx / total, 1.0))
        last_sound = render_dashboard(stats, alerts, widgets, sound_on, last_sound)

        for a in alerts:
            events.append({
                "video_time_sec": round(frame_idx / fps, 2),
                "frame": frame_idx,
                "event": a,
                "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    cap.release()
    progress.progress(1.0)
    st.info(f"Analysis finished - {frame_idx} frames read.")
    show_report(events)


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

model = get_model(weights)
if model is None:
    st.sidebar.error("YOLO model unavailable - install `ultralytics` and check the weights file.")
if not MEDIAPIPE_AVAILABLE:
    st.sidebar.warning("MediaPipe not installed - head-turn analysis is disabled.")

detector = CheatingDetector(model=model, conf_threshold=conf_threshold, head_threshold=head_threshold)

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
    dirs_text = st.sidebar.text_area("Dataset folders (one per line)",
                                     value="\n".join(DEFAULT_DATASET_DIRS), height=110)
    search_dirs = tuple(d for d in dirs_text.splitlines() if d.strip())
    if st.sidebar.button("🔄 Rescan folders"):
        get_dataset_videos.clear()

    videos = get_dataset_videos(search_dirs)
    if not videos:
        st.warning("No videos found. Edit the folder list in the sidebar or upload a video.")
    else:
        st.success(f"Found {len(videos)} video(s).")
        query = st.text_input("Filter by name (optional)")
        filtered = [v for v in videos if query.lower() in v.lower()] if query else videos
        if filtered:
            selected = st.selectbox("Select a video:", filtered,
                                    format_func=lambda p: f"{Path(p).name}  —  {p}")
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
        render_dashboard(stats, alerts, {"metrics": st.empty(), "alerts": st.empty()}, sound_on, 0.0)
        if alerts:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            show_report([{"video_time_sec": 0, "frame": 1, "event": a, "logged_at": now} for a in alerts])
