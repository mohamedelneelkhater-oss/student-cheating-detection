"""Streamlit web app: analyse an exam video (upload or URL), see alerts and screenshots.

Run locally:  streamlit run app.py
"""
import ipaddress
import socket
import tempfile
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

from src.database import AlertDatabase
from src.detector import CheatingMonitor, DetectorConfig, format_notification

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "best.pt"
ALERT_DIR = ROOT / "alerts"
MAX_DOWNLOAD_MB = 300

st.set_page_config(page_title="Cheating Tracking & Alert System", page_icon="🚨", layout="wide")


def download_video(url: str, dest_dir: Path) -> Path:
    """Download a direct video link (http/https only, public hosts only, size limited)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Please provide a valid http(s) link to a video file.")
    for info in socket.getaddrinfo(parsed.hostname, None):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("Links to private or local addresses are not allowed.")
    name = Path(parsed.path).name or "video.mp4"
    dest = dest_dir / name
    size = 0
    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD_MB * 1024 * 1024:
                    raise ValueError(f"Video is larger than {MAX_DOWNLOAD_MB} MB.")
                fh.write(chunk)
    return dest


def analyse(video_path: Path, cfg: DetectorConfig, max_frames: int, preview_every: int) -> dict:
    db = AlertDatabase(cfg.db_path)
    monitor = CheatingMonitor(cfg, db=db)
    frame_slot, bar = st.empty(), st.progress(0.0, text="Analysing video ...")
    live_alerts = st.container()

    def on_progress(fraction: float, item: dict) -> None:
        bar.progress(min(fraction, 1.0), text=f"Analysing video ... {fraction:.0%}")
        if item["frame_index"] % preview_every == 0:
            frame_slot.image(item["frame"][:, :, ::-1])
        for event in item["events"]:
            st.toast(f"Cheating alert: {event.behavior}", icon="🚨")
            with live_alerts:
                st.error(format_notification(event))

    out_path = Path(tempfile.mkdtemp()) / "annotated.mp4"
    summary = monitor.process_video(video_path, output_path=out_path, max_frames=max_frames,
                                    progress_cb=on_progress)
    bar.empty()
    frame_slot.empty()
    return summary


def render_result(res: dict) -> None:
    alerts = res["alerts"]
    st.success(f"Processed {res['frames_processed']} frames in {res['elapsed_sec']:.1f}s "
               f"- {len(alerts)} alert(s) raised.")
    if res["output_video"] and Path(res["output_video"]).exists():
        video_bytes = Path(res["output_video"]).read_bytes()
        st.video(video_bytes)
        st.download_button("Download annotated video", video_bytes, "annotated.mp4", "video/mp4")
    if alerts:
        df = pd.DataFrame([a.to_dict() for a in alerts]).drop(columns=["firebase_url"])
        st.subheader("Alerts")
        st.dataframe(df)
        st.download_button("Download alerts (CSV)", df.to_csv(index=False), "alerts.csv", "text/csv")
        st.subheader("Screenshots")
        cols = st.columns(3)
        for i, a in enumerate(alerts[:12]):
            if Path(a.screenshot_path).exists():
                cols[i % 3].image(a.screenshot_path,
                                  caption=f"{a.behavior} | {a.confidence:.0%} | t={a.video_time_sec}s")


# ---------------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
conf = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
min_frames = st.sidebar.slider("Frames before alert", 1, 15, 5,
                               help="A student must be flagged this many processed frames in a row.")
cooldown = st.sidebar.slider("Alert cooldown (seconds)", 1, 60, 10)
stride = st.sidebar.slider("Process every N-th frame", 1, 10, 2,
                           help="Higher = faster, lower = more precise.")
imgsz = st.sidebar.selectbox("Inference image size", [320, 480, 640], index=2)
max_frames = st.sidebar.number_input("Max frames to process", 30, 5000, 600, step=30)
preview_every = st.sidebar.slider("Live preview every N frames", 1, 20, 3)

st.title("🚨 Cheating Tracking and Alert System")
st.caption("YOLO detection + ByteTrack tracking. Red box = cheating, green box = normal behaviour.")

if not WEIGHTS.exists():
    st.error("Model weights not found at models/best.pt. Train the model (see the Kaggle notebook) "
             "and copy the weights into the models/ folder.")
    st.stop()

tab_video, tab_history, tab_model, tab_about = st.tabs(
    ["Video Analysis", "Alert History", "Model Comparison", "About"])

with tab_video:
    source = st.radio("Video source", ["Upload a file", "Video URL (direct link)"], horizontal=True)
    video_file, video_url = None, ""
    if source == "Upload a file":
        video_file = st.file_uploader("Exam video", type=["mp4", "avi", "mov", "mkv"])
    else:
        video_url = st.text_input("Direct link to a video file (https://.../exam.mp4)")

    if st.button("Analyse video", type="primary"):
        try:
            work_dir = Path(tempfile.mkdtemp())
            if video_file is not None:
                video_path = work_dir / video_file.name
                video_path.write_bytes(video_file.getbuffer())
            elif video_url.strip():
                with st.spinner("Downloading video ..."):
                    video_path = download_video(video_url.strip(), work_dir)
            else:
                st.warning("Upload a video or paste a link first.")
                st.stop()
            cfg = replace(DetectorConfig(), weights=str(WEIGHTS), conf=conf, imgsz=imgsz,
                          min_consecutive_frames=min_frames, cooldown_seconds=float(cooldown),
                          frame_stride=stride, screenshot_dir=str(ALERT_DIR / "screenshots"),
                          db_path=str(ALERT_DIR / "alerts.db"))
            st.session_state["last_run"] = analyse(video_path, cfg, int(max_frames), preview_every)
        except Exception as exc:
            st.error(f"Could not analyse the video: {exc}")

    if "last_run" in st.session_state:
        render_result(st.session_state["last_run"])

with tab_history:
    rows = AlertDatabase(str(ALERT_DIR / "alerts.db")).fetch_all()
    if not rows:
        st.info("No alerts yet. Analyse a video first.")
    else:
        df = pd.DataFrame(rows)
        st.dataframe(df.drop(columns=["firebase_url"]))
        pick = st.selectbox("Open a saved screenshot", df["id"], format_func=lambda i: (
            f"#{i} - {df.loc[df['id'] == i, 'behavior'].iloc[0]} - "
            f"{df.loc[df['id'] == i, 'detected_at'].iloc[0]}"))
        shot = Path(df.loc[df["id"] == pick, "screenshot_path"].iloc[0])
        if shot.exists():
            st.image(str(shot))
        else:
            st.warning("The screenshot file is no longer available on this server.")

with tab_model:
    csv_path = ROOT / "assets" / "model_comparison.csv"
    if csv_path.exists():
        st.subheader("YOLOv8 vs YOLO11")
        st.dataframe(pd.read_csv(csv_path))
    for image in ("model_comparison.png", "training_curves.png", "confusion_matrix.png"):
        if (ROOT / "assets" / image).exists():
            st.image(str(ROOT / "assets" / image))

with tab_about:
    st.markdown(
        "**Pipeline:** video frame -> YOLO detection -> ByteTrack IDs -> cheating rule "
        "(persistence + cooldown) -> red box, screenshot, database record and notification.\n\n"
        "**Note:** on Streamlit Community Cloud the alert history is temporary and resets when "
        "the app restarts. Use the desktop app or Firebase for permanent storage.")
