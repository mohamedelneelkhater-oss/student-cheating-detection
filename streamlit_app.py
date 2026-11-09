"""
Exam Cheating Detection System - Streamlit app
Model: YOLO fine-tuned on a Roboflow exam cheating dataset (weights/best.pt)

Inputs : image, camera, video upload, internet link, dataset folders
Output : red boxes for cheating classes, green boxes for normal classes, alert history,
         screenshots of every alert, CSV export
"""

import io
import os
import queue
import shutil
import tempfile
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Exam Cheating Detection", page_icon="🎓", layout="wide")

MODEL_PATH = Path(__file__).parent / "weights" / "best.pt"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET_DIRS = ["samples", "data", "dataset", "/kaggle/input"]
PAGE_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "facebook.com", "tiktok.com", "dailymotion.com")
NORMAL_WORDS = ("normal", "not_cheat", "not cheat", "not-cheat", "notcheat", "no_cheat", "non_cheat",
                "non-cheat", "honest", "clean")
RED, GREEN = (0, 0, 255), (0, 200, 0)  # BGR
MAX_EVIDENCE = 60

# Speed presets: inference size, analyze every Nth frame, max frame width, preview interval, turbo (no live video)
PRESETS = {
    "🚀 Ultra": dict(imgsz=640, frame_skip=6, max_width=640, preview_every=8, turbo=True),
    "⚡ Fast": dict(imgsz=640, frame_skip=4, max_width=640, preview_every=5, turbo=False),
    "⚖️ Balanced": dict(imgsz=800, frame_skip=3, max_width=960, preview_every=3, turbo=False),
    "🎯 Accurate": dict(imgsz=960, frame_skip=2, max_width=1280, preview_every=2, turbo=False),
}


# --------------------------------------------------------------------------- #
# Model loading (cached so it loads only once)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model():
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{MODEL_PATH}'. "
            "Make sure 'weights/best.pt' is uploaded to the repository."
        )
    return YOLO(str(MODEL_PATH))


@st.cache_data(show_spinner=False)
def find_dataset_files(search_dirs):
    videos, images = [], []
    for directory in search_dirs:
        root = Path(directory.strip())
        if root.exists() and root.is_dir():
            for path in root.rglob("*"):
                ext = path.suffix.lower()
                if ext in VIDEO_EXTENSIONS:
                    videos.append(str(path))
                elif ext in IMAGE_EXTENSIONS:
                    images.append(str(path))
    return sorted(set(videos)), sorted(set(images))


# --------------------------------------------------------------------------- #
# Detection, drawing and alert logic
# --------------------------------------------------------------------------- #
def detect(model, frame, conf, iou, imgsz, track=False, persist=False):
    """Run the model on a BGR frame and return a list of detections."""
    kw = dict(imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    if track:
        res = model.track(frame, persist=persist, tracker="bytetrack.yaml", **kw)[0]
    else:
        res = model.predict(frame, **kw)[0]
    dets = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.int().cpu().tolist()
        confs = res.boxes.conf.cpu().tolist()
        ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None] * len(classes)
        for box, cls_id, cf, tid in zip(xyxy, classes, confs, ids):
            dets.append({"box": [float(v) for v in box], "name": model.names[cls_id],
                         "conf": float(cf), "track": tid})
    return dets


def draw_detections(frame, dets, cheat_classes):
    """Red box for cheating classes, green box for normal classes."""
    out = frame.copy()
    scale = max(0.5, out.shape[1] / 1280 * 0.7)
    th = max(2, int(scale * 3))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for d in dets:
        color = RED if d["name"] in cheat_classes else GREEN
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, th)
        tag = f"{d['name']} {d['conf']:.2f}" + (f" #{d['track']}" if d["track"] is not None else "")
        (tw, text_h), _ = cv2.getTextSize(tag, font, scale, max(1, th - 1))
        top = max(y1 - text_h - 8, 0)
        cv2.rectangle(out, (x1, top), (x1 + tw + 4, top + text_h + 8), color, -1)
        cv2.putText(out, tag, (x1 + 2, top + text_h + 3), font, scale, (255, 255, 255), max(1, th - 1))
    return out


class AlertTracker:
    """Raises an alert when the same tracked person shows a cheating class for N consecutive frames."""

    def __init__(self, min_frames=3, cooldown=10.0):
        self.min_frames, self.cooldown = min_frames, cooldown
        self.state, self.alerts, self.persons = {}, [], set()

    def update(self, t, cheat_dets):
        best = {}
        for d in cheat_dets:
            key = d["track"] if d["track"] is not None else f"class:{d['name']}"
            if key not in best or d["conf"] > best[key]["conf"]:
                best[key] = d

        new = []
        for key, d in best.items():
            st_ = self.state.setdefault(key, {"streak": 0, "last_alert": None, "best": 0.0})
            st_["streak"] += 1
            st_["best"] = max(st_["best"], d["conf"])
            if st_["streak"] >= self.min_frames and (st_["last_alert"] is None or t - st_["last_alert"] >= self.cooldown):
                st_["last_alert"] = t
                alert = {"id": len(self.alerts) + 1, "time_sec": round(t, 2), "person": key,
                         "class": d["name"], "confidence": round(st_["best"], 2), "screenshot": None}
                self.alerts.append(alert)
                self.persons.add(key)
                new.append(alert)
        for key, st_ in self.state.items():
            if key not in best:
                st_["streak"], st_["best"] = 0, 0.0
        return new


def fmt_time(seconds):
    seconds = max(float(seconds), 0.0)
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def person_label(key):
    return f"#{key}" if isinstance(key, int) else "unassigned"


def alarm_html():
    return (f'<audio autoplay data-t="{time.time()}">'
            '<source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" '
            'type="audio/mpeg"></audio>')


# --------------------------------------------------------------------------- #
# Video helpers (background decoding thread + internet links)
# --------------------------------------------------------------------------- #
def _put(q, item, stop):
    while not stop.is_set():
        try:
            q.put(item, timeout=0.2)
            return True
        except queue.Full:
            continue
    return False


def frame_reader(cap, frame_skip, max_width, max_frames, out_q, stop, counter):
    """Decodes and resizes frames in a background thread so decoding overlaps with inference."""
    idx = 0
    try:
        while not stop.is_set():
            if not cap.grab():
                break
            idx += 1
            counter[0] = idx
            if max_frames and idx > max_frames:
                break
            if idx % frame_skip:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
            if not _put(out_q, (idx, frame), stop):
                break
    finally:
        _put(out_q, None, stop)


def resolve_url(url):
    """Return (playable_source, temp_dir_or_None, error_or_None)."""
    url = url.strip()
    lower = url.lower().split("?")[0]
    if lower.endswith(tuple(VIDEO_EXTENSIONS)) and not any(h in url for h in PAGE_HOSTS):
        cap = cv2.VideoCapture(url)
        ok = cap.isOpened()
        cap.release()
        if ok:
            return url, None, None
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "video" + Path(lower).suffix)
            urllib.request.urlretrieve(url, path)
            return path, tmp, None
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            return None, None, f"Could not open or download the link: {exc}"

    tmp = tempfile.mkdtemp()
    try:
        import yt_dlp
        opts = {"format": "best[height<=720][ext=mp4]/best[height<=720]/best",
                "outtmpl": os.path.join(tmp, "video.%(ext)s"), "quiet": True,
                "noplaylist": True, "max_filesize": 300 * 1024 * 1024}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info), tmp, None
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        return None, None, (f"Could not download this link ({exc}). Sites like YouTube often block cloud "
                            "servers - use a direct .mp4 link or upload the file.")


def analyze_video(model, source, source_name, cfg, cheat_classes):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Could not open the video source. Check the file path or link.")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    max_frames = int(cfg["max_seconds"] * fps) if cfg["max_seconds"] else 0
    tracker = AlertTracker(cfg["min_frames"], cfg["cooldown"])
    evidence = []

    frame_slot, progress, status = st.empty(), st.progress(0.0), st.empty()
    metrics_slot, alert_slot, sound_slot = st.empty(), st.empty(), st.empty()
    processed, last_sound, first, started = 0, 0.0, True, time.time()

    frames_q, stop, counter = queue.Queue(maxsize=8), threading.Event(), [0]
    reader = threading.Thread(target=frame_reader, daemon=True,
                              args=(cap, cfg["frame_skip"], cfg["max_width"], max_frames, frames_q, stop, counter))
    reader.start()

    try:
        while True:
            item = frames_q.get()
            if item is None:
                break
            idx, frame = item
            t = idx / fps
            dets = detect(model, frame, cfg["conf"], cfg["iou"], cfg["imgsz"], track=True, persist=not first)
            first = False
            new_alerts = tracker.update(t, [d for d in dets if d["name"] in cheat_classes])
            processed += 1

            will_show = (not cfg["turbo"]) and processed % cfg["preview_every"] == 0
            annotated = draw_detections(frame, dets, cheat_classes) if (will_show or new_alerts) else None

            for alert in new_alerts:  # screenshot on every cheating alert
                if len(evidence) < MAX_EVIDENCE:
                    small = annotated
                    if small.shape[1] > 640:
                        small = cv2.resize(small, (640, int(small.shape[0] * 640 / small.shape[1])))
                    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        alert["screenshot"] = len(evidence)
                        evidence.append(buf.tobytes())
                alert_slot.error(f"🚨 Cheating alert at {fmt_time(t)} - {alert['class']} "
                                 f"(person {person_label(alert['person'])}, conf {alert['confidence']:.2f})")

            if will_show or new_alerts:
                if not cfg["turbo"]:
                    frame_slot.image(annotated, channels="BGR", output_format="JPEG", use_container_width=True)
                with metrics_slot.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Detections in frame", len(dets))
                    c2.metric("Cheating persons", len(tracker.persons))
                    c3.metric("Alerts so far", len(tracker.alerts))

            if total > 0 and processed % 5 == 0:
                progress.progress(min(idx / total, 1.0))
            if processed % 10 == 0:
                speed = processed / max(time.time() - started, 1e-6)
                status.caption(f"Analyzed {processed} frames | {speed:.1f} fps | alerts so far: {len(tracker.alerts)}")
            if cfg["sound"] and new_alerts and time.time() - last_sound > 3:
                sound_slot.markdown(alarm_html(), unsafe_allow_html=True)
                last_sound = time.time()
    finally:
        stop.set()
        reader.join(timeout=3)
        cap.release()

    progress.progress(1.0)
    elapsed = max(time.time() - started, 1e-6)
    status.success(f"Done in {elapsed:.1f}s - {processed} frames analyzed ({processed / elapsed:.1f} fps)")
    return {
        "alerts": tracker.alerts, "evidence": evidence, "persons": sorted(tracker.persons, key=str),
        "frames": counter[0], "source": source_name,
        "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def show_report(result):
    st.divider()
    st.header("📋 Alert Report")
    st.caption(f"Source: {result['source']}  |  Frames read: {result['frames']}  |  Finished: {result['finished']}")
    st.caption("Automated alerts are indicators for human review, not proof of cheating.")

    alerts, persons = result["alerts"], result["persons"]
    if not alerts:
        st.success("No cheating alerts were raised.")
        return
    st.error(f"⚠️ {len(persons)} person(s) flagged for possible cheating.")
    c1, c2 = st.columns(2)
    c1.metric("Cheating cases (persons)", len(persons))
    c2.metric("Total alerts", len(alerts))

    df = pd.DataFrame(alerts)
    df["time"] = df["time_sec"].apply(fmt_time)
    df["person"] = df["person"].apply(person_label)
    st.subheader("Alert history")
    st.dataframe(df[["id", "time", "person", "class", "confidence"]], use_container_width=True, hide_index=True)

    shots = [(row, result["evidence"][int(row["screenshot"])]) for _, row in df.iterrows()
             if pd.notna(row["screenshot"]) and int(row["screenshot"]) < len(result["evidence"])]
    if shots:
        st.subheader("Alert screenshots")
        grid = st.columns(3)
        for i, (row, jpg) in enumerate(shots):
            grid[i % 3].image(jpg, caption=f"#{row['id']} {row['time']} - {row['person']} - {row['class']}")

    st.download_button("⬇️ Download alert history (CSV)",
                       df.drop(columns=["screenshot"]).to_csv(index=False).encode("utf-8"),
                       file_name=f"cheating_alerts_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")


def show_image_result(model, image, cfg, cheat_classes):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    dets = detect(model, frame, cfg["conf"], cfg["iou"], cfg["imgsz"])
    annotated = Image.fromarray(cv2.cvtColor(draw_detections(frame, dets, cheat_classes), cv2.COLOR_BGR2RGB))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detection result")
        st.image(annotated, use_container_width=True)
    st.divider()

    counts = {}
    for d in dets:
        counts[d["name"]] = counts.get(d["name"], 0) + 1
    cheating = sum(n for name, n in counts.items() if name in cheat_classes)
    if not counts:
        st.info("Nothing detected at the current confidence threshold. Try lowering it from the sidebar.")
    else:
        if cheating:
            st.error(f"⚠️ {cheating} cheating detection(s) found.")
        else:
            st.success("No cheating detected.")
        st.subheader(f"Detections ({sum(counts.values())} total)")
        st.table({"Class": list(counts), "Count": list(counts.values()),
                  "Type": ["cheating" if n in cheat_classes else "normal" for n in counts]})

    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    st.download_button("⬇️ Download result", data=buffer.getvalue(),
                       file_name="exam_cheating_result.png", mime="image/png")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🎓 Exam Cheating Detection System")
st.caption("Powered by YOLO | Red boxes = cheating, green boxes = normal behavior")

# --------------------------------------------------------------------------- #
# Load model (with a clear error message if something is wrong)
# --------------------------------------------------------------------------- #
try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

all_classes = list(model.names.values())
default_cheat = [n for n in all_classes if not any(w in n.lower() for w in NORMAL_WORDS)]

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    preset = st.selectbox("Speed preset", list(PRESETS), index=0)
    PS = PRESETS[preset]
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05,
                     help="Lower values detect more but may add false alarms.")
    iou = st.slider("NMS IoU threshold", 0.10, 0.95, 0.45, 0.05,
                    help="Controls how overlapping boxes are merged.")
    cheat_classes = set(st.multiselect("Classes treated as cheating", all_classes, default=default_cheat))

    st.divider()
    st.subheader("Alerts and speed")
    min_frames = st.slider("Alert after N consecutive frames", 1, 10, 3)
    cooldown = st.slider("Seconds before the same person can alert again", 2, 60, 10)
    imgsz = st.select_slider("Inference size", options=[416, 512, 640, 800, 960, 1280],
                             value=PS["imgsz"], key=f"imgsz_{preset}")
    frame_skip = st.slider("Analyze every Nth frame", 1, 10, PS["frame_skip"], key=f"skip_{preset}")
    max_width = st.select_slider("Max frame width", options=[480, 640, 960, 1280],
                                 value=PS["max_width"], key=f"width_{preset}")
    turbo = st.checkbox("Turbo: no live video, report only (fastest)", value=PS["turbo"], key=f"turbo_{preset}")
    preview_every = st.slider("Live preview every Nth analyzed frame", 1, 20, PS["preview_every"], key=f"prev_{preset}")
    max_seconds = st.number_input("Analyze only the first N seconds (0 = whole video)", min_value=0, value=0, step=10)
    sound_on = st.checkbox("Alarm sound", value=True)

    st.divider()
    input_mode = st.radio("Input source",
                          ["Upload image", "Use camera", "Upload video", "Video link", "Dataset (samples)"])
    st.divider()
    st.markdown("**About**  \nFine-tuned YOLO model trained on a Roboflow exam cheating dataset. "
                "Built with Ultralytics, PyTorch, and Streamlit.")

cfg = dict(conf=conf, iou=iou, imgsz=imgsz, frame_skip=frame_skip, max_width=max_width, turbo=turbo,
           preview_every=preview_every, max_seconds=max_seconds, min_frames=min_frames,
           cooldown=cooldown, sound=sound_on)


def store(result):
    if result is not None:
        st.session_state["report"] = result


# --------------------------------------------------------------------------- #
# Input, inference and results
# --------------------------------------------------------------------------- #
if input_mode == "Upload image":
    source = st.file_uploader("Upload an exam image", type=["jpg", "jpeg", "png", "webp"])
    if source is not None:
        with st.spinner("Detecting..."):
            show_image_result(model, Image.open(source).convert("RGB"), cfg, cheat_classes)
    else:
        st.info("Upload an image to start.")

elif input_mode == "Use camera":
    source = st.camera_input("Take a photo of the exam room")
    if source is not None:
        with st.spinner("Detecting..."):
            show_image_result(model, Image.open(source).convert("RGB"), cfg, cheat_classes)
    else:
        st.info("Use your camera to start.")

elif input_mode == "Upload video":
    uploaded = st.file_uploader("Upload an exam video", type=[e.strip(".") for e in VIDEO_EXTENSIONS])
    if uploaded and st.button("▶️ Start analysis", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            store(analyze_video(model, tmp_path, uploaded.name, cfg, cheat_classes))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    elif not uploaded:
        st.info("Upload a video to start.")

elif input_mode == "Video link":
    st.caption("Direct file links (.mp4 ...) work best. YouTube / Vimeo links are downloaded first "
               "(they may be blocked on cloud servers).")
    url = st.text_input("Video link:")
    if url and st.button("▶️ Start analysis", type="primary"):
        with st.spinner("Preparing the video..."):
            video_source, tmp_dir, err = resolve_url(url)
        if err:
            st.error(err)
        else:
            try:
                store(analyze_video(model, video_source, url, cfg, cheat_classes))
            finally:
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
    elif not url:
        st.info("Paste a video link to start.")

else:  # Dataset (samples)
    dirs_text = st.sidebar.text_area("Dataset folders (one per line)", value="\n".join(DEFAULT_DATASET_DIRS), height=110)
    search_dirs = tuple(d for d in dirs_text.splitlines() if d.strip())
    if st.sidebar.button("🔄 Rescan folders"):
        find_dataset_files.clear()
    videos, images = find_dataset_files(search_dirs)
    st.write(f"Found **{len(videos)}** video(s) and **{len(images)}** image(s).")

    kind = st.radio("File type", ["Videos", "Images"], horizontal=True)
    files = videos if kind == "Videos" else images
    if not files:
        st.warning(f"No {kind.lower()} found. Put files in a `samples/` folder in the repository.")
    else:
        selected = st.selectbox("Select a file:", files, format_func=lambda p: f"{Path(p).name}  -  {p}")
        if kind == "Videos":
            with st.expander("Preview original video"):
                st.video(selected)
            if st.button("▶️ Start analysis", type="primary"):
                store(analyze_video(model, selected, Path(selected).name, cfg, cheat_classes))
        else:
            with st.spinner("Detecting..."):
                show_image_result(model, Image.open(selected).convert("RGB"), cfg, cheat_classes)

# The video report lives in session_state so it survives reruns (e.g. clicking the download button).
if "report" in st.session_state and input_mode in ("Upload video", "Video link", "Dataset (samples)"):
    show_report(st.session_state["report"])
