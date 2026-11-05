import sys
import os
import tempfile
import threading
from pathlib import Path

import sys
from pathlib import Path

# ==========================================
# 0. FIX PYTHON PATH FOR STREAMLIT CLOUD
# ==========================================
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2
import pandas as pd
import streamlit as st

from src.database import AlertDatabase
from src.detector import CheatingMonitor, DetectorConfig, format_notification


# ==========================================
# 1. APP CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Cheating Detection", page_icon="🎓", layout="wide"
)

MODEL_PATH = ROOT / "models" / "best.pt"
SCREENSHOT_DIR = ROOT / "alerts" / "screenshots"
DB_PATH = ROOT / "alerts" / "alerts.db"

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. INITIALIZE SESSION STATE
# ==========================================

if "alert_events" not in st.session_state:
    st.session_state.alert_events = []

if "db" not in st.session_state:
    st.session_state.db = AlertDatabase(str(DB_PATH))

if "monitor" not in st.session_state:
    cfg = DetectorConfig(
        weights=str(MODEL_PATH),
        screenshot_dir=str(SCREENSHOT_DIR),
        db_path=str(DB_PATH),
    )

    st.session_state.monitor = CheatingMonitor(
        cfg,
        db=st.session_state.db,
        on_alert=lambda event: st.session_state.alert_events.append(event),
    )


# ==========================================
# 3. SIDEBAR
# ==========================================

st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Select Page", ["Home", "Cheating Detection", "Alert History", "About"]
)


# ==========================================
# 4. HOME PAGE
# ==========================================

if page == "Home":

    st.title("🎓 Student Cheating Detection System")

    st.subheader("AI-Powered Exam Monitoring and Cheating Detection")

    st.write(
        """
    Welcome to the Student Cheating Detection System.

    This application uses Computer Vision and Deep Learning
    to analyze examination videos and identify suspicious
    student behavior.
    """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Detection System", "AI / Computer Vision")

    with col2:
        st.metric("Monitoring", "Video Analysis")

    with col3:
        st.metric("Alerts", "Real Time During Processing")

    st.divider()

    st.info(
        "Upload an examination video from the Cheating Detection page "
        "to start analyzing student behavior."
    )


# ==========================================
# 5. CHEATING DETECTION PAGE
# ==========================================

elif page == "Cheating Detection":

    st.title("📹 Cheating Detection")

    st.write("Upload an examination video to analyze suspicious behavior.")

    st.divider()

    # Check model
    if not MODEL_PATH.exists():

        st.error(f"Model weights not found: {MODEL_PATH}")

        st.stop()

    # Upload video
    uploaded_video = st.file_uploader(
        "Upload Examination Video", type=["mp4", "avi", "mov", "mkv"]
    )

    # Confidence threshold
    confidence = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    st.divider()

    # Video display
    video_placeholder = st.empty()

    # Progress bar
    progress_bar = st.progress(0)

    status_text = st.empty()

    # Start detection
    if st.button(
        "🚀 Start Detection",
        type="primary",
        disabled=uploaded_video is None,
    ):

        if uploaded_video is None:

            st.warning("Please upload a video first.")

        else:

            # Reset alerts for this run
            st.session_state.alert_events = []

            temp_path = None

            try:

                # Save uploaded video temporarily
                suffix = Path(uploaded_video.name).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp_file:

                    temp_file.write(uploaded_video.getvalue())

                    temp_path = temp_file.name

                # Configure model
                monitor = st.session_state.monitor

                monitor.cfg.conf = confidence

                stop_event = threading.Event()

                status_text.info("Loading video and AI model...")

                # Process video
                with st.spinner("Analyzing examination video..."):

                    for item in monitor.stream_video(
                        temp_path, stop_event=stop_event
                    ):

                        frame = item["frame"]

                        frame_index = item["frame_index"]

                        total_frames = item["total_frames"] or 1

                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Display processed frame
                        video_placeholder.image(
                            frame_rgb,
                            caption="Live Video Analysis",
                            use_container_width=True,
                        )

                        # Update progress
                        progress = min(frame_index / total_frames, 1.0)

                        progress_bar.progress(progress)

                        status_text.info(
                            f"Processing frame: {frame_index} / {total_frames}"
                        )

                progress_bar.progress(1.0)

                status_text.success("Video analysis completed successfully!")

                st.success("Detection process finished.")

                # Display detected alerts
                if st.session_state.alert_events:

                    st.subheader("🚨 Detected Suspicious Activities")

                    for event in st.session_state.alert_events:

                        try:
                            message = format_notification(event)
                        except Exception:
                            message = str(event)

                        st.warning(message)

                else:

                    st.success(
                        "No suspicious activities were reported by the detector."
                    )

            except Exception as e:

                st.error("An error occurred during video processing.")

                st.exception(e)

            finally:

                # Remove temporary video
                if temp_path and os.path.exists(temp_path):

                    os.remove(temp_path)


# ==========================================
# 6. ALERT HISTORY PAGE
# ==========================================

elif page == "Alert History":

    st.title("🚨 Alert History")

    st.write("View all previously recorded suspicious activities.")

    if st.button("🔄 Refresh History"):

        st.rerun()

    try:

        records = st.session_state.db.fetch_all(limit=500)

        if records:

            history = []

            for r in records:

                history.append(
                    {
                        "ID": r["id"],
                        "Date / Time": str(r["detected_at"]).replace(
                            "T", " "
                        ),
                        "Behavior": r["behavior"],
                        "Confidence": round(float(r["confidence"]), 2),
                        "Screenshot": r["screenshot_path"],
                    }
                )

            df = pd.DataFrame(history)

            st.metric("Total Recorded Alerts", len(df))

            st.dataframe(df, use_container_width=True, hide_index=True)

            st.divider()

            # Screenshot viewer
            st.subheader("🖼️ View Alert Screenshot")

            screenshot_options = [
                r["Screenshot"] for r in history if r["Screenshot"]
            ]

            if screenshot_options:

                selected_screenshot = st.selectbox(
                    "Select Screenshot", screenshot_options
                )

                if os.path.exists(selected_screenshot):

                    st.image(
                        selected_screenshot,
                        caption="Detected Suspicious Activity",
                        use_container_width=True,
                    )

                else:

                    st.warning("Screenshot file not found.")

        else:

            st.info("No alerts have been recorded yet.")

    except Exception as e:

        st.error(f"Could not load alert history: {e}")


# ==========================================
# 7. ABOUT PAGE
# ==========================================

elif page == "About":

    st.title("ℹ️ About the Project")

    st.write(
        """
    # Student Cheating Detection System

    This project uses Artificial Intelligence and Computer Vision
    to monitor examination videos and detect suspicious behavior.

    ### Main Features

    - Upload examination videos.
    - Analyze video frames.
    - Detect suspicious student behavior using the trained model.
    - Generate alerts.
    - Save detection screenshots.
    - Store alert history in a database.

    ### Technologies

    - Python
    - Streamlit
    - OpenCV
    - Machine Learning / Deep Learning
    - Computer Vision
    - SQLite
    """
    )

    st.success("Student Cheating Detection System")
