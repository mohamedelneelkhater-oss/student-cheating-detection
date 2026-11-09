"""
Exam Cheating Detection & Analytics System  -  v25.0 Universal Video Formats & Clean Session Support
"""

import os
import shutil
import tempfile
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

APP_VERSION = "v25.0 UNIVERSAL"

# ----------------------------------------------------------------------------
# Constants & Settings (تم توسيع نطاق صيغ الفيديو لتشمل كل الأنواع)
# ----------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET_DIRS = ["samples", "data", "dataset", "/kaggle/input", "../input"]
PAGE_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "facebook.com", "tiktok.com", "dailymotion.com")

KP_CONF = 0.35
MIN_PERSON_AREA = 700   
MAX_EVIDENCE = 60
NORMAL_WORDS = ("normal", "not_cheat", "not cheat", "no_cheat", "non_cheat", "non-cheat", "honest", "clean")

OBJECT_INFO = {
    "cell phone": ("Mobile phone / Smartphone", "HIGH"),
    "book": ("Book / notes / paper", "HIGH"),
    "laptop": ("Laptop / tablet", "HIGH"),
    "remote": ("Unknown electronic device", "MEDIUM"),
}
SEV_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
SEV_COLOR = {"NONE": (0, 200, 0), "LOW": (0, 220, 255), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}
SEV_ICON = {"LOW": "LOW", "MEDIUM": "MED", "HIGH": "HIGH"}

PRESETS = {
    "Balanced": dict(imgsz=640, frame_skip=2, object_every=2, max_width=960, preview_every=2),
    "Accurate": dict(imgsz=960, frame_skip=1, object_every=1, max_width=1280, preview_every=1),
    "Fast": dict(imgsz=480, frame_skip=4, object_every=3, max_width=640, preview_every=3),
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def inside(box, x, y):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)


def filter_clean_persons(persons, iou_threshold=0.25):
    if not persons:
        return []
    sorted_persons = sorted(persons, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]), reverse=True)
    kept = []
    for p in sorted_persons:
        should_keep = True
        for k in kept:
            box1, box2 = p["box"], k["box"]
            iou = compute_iou(box1, box2)
            x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
            x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            if iou > iou_threshold or (area1 > 0 and inter / area1 > 0.6):
                should_keep = False
                break
        if should_keep:
            kept.append(p)
    return kept


def analyze_head(kp, yaw_threshold, min_conf=KP_CONF):
    nose, l_eye, r_eye = kp[0], kp[1], kp[2]
    if nose[2] < min_conf:
        return None, None, False
    l_ok, r_ok = l_eye[2] >= min_conf, r_eye[2] >= min_conf
    yaw = None
    if l_ok and r_ok:
        eye_dist = abs(r_eye[0] - l_eye[0])
        if eye_dist > 1e-3:
            yaw = float((nose[0] - (l_eye[0] + r_eye[0]) / 2) / eye_dist)
    elif l_ok != r_ok:
        eye = l_eye if l_ok else r_eye
        yaw = 1.0 if nose[0] > eye[0] else -1.0
    if yaw is None:
        return None, None, True
    if yaw < -yaw_threshold:
        return yaw, "LEFT", True
    if yaw > yaw_threshold:
        return yaw, "RIGHT", True
    return yaw, None, True


def find_neighbor(person, direction, persons, reach):
    x1, y1, x2, y2 = person["box"]
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    best = None
    for o in persons:
        if o["id"] == person["id"]:
            continue
        ox1, oy1, ox2, oy2 = o["box"]
        dx = (ox1 + ox2) / 2 - cx
        dy = (oy1 + oy2) / 2 - cy
        if (direction == "LEFT" and dx >= 0) or (direction == "RIGHT" and dx <= 0):
            continue
        if abs(dx) > reach * w or abs(dy) > 1.5 * h:
            continue
        if best is None or abs(dx) < best[1]:
            best = (o["id"], abs(dx))
    return best[0] if best else None


def assign_owner(obj_box, persons):
    cx, cy = (obj_box[0] + obj_box[2]) / 2, (obj_box[1] + obj_box[3]) / 2
    best, best_d = None, None
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        d = (((x1 + x2) / 2 - cx) ** 2 + ((y1 + y2) / 2 - cy) ** 2) ** 0.5
        if inside(p["box"], cx, cy) or d < (x2 - x1) * 1.8:
            if best_d is None or d < best_d:
                best, best_d = p["id"], d
    return best


def same_item(a, b, thr=0.5):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return False
    return inter / union > thr


def dedupe_objects(objects):
    kept = []
    for o in sorted(objects, key=lambda d: -d["conf"]):
        if not any(same_item(o["box"], k["box"]) for k in kept):
            kept.append(o)
    return kept


def get_stable_student_count(persons):
    if not persons:
        return 0
    return len(persons)


def fmt_time(seconds):
    seconds = max(float(seconds), 0.0)
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


# ----------------------------------------------------------------------------
# Incident Manager with Frame Counter Integration (>= 30 frames constraint)
# ----------------------------------------------------------------------------
class IncidentManager:
    def __init__(self, grace=1.5, min_frame_count=30):
        self.grace = grace
        self.min_frame_count = min_frame_count
        self.state = {}
        self.events = []
        self.evidence = []

    def update(self, t, conditions, frame_idx):
        new_events = []
        for key, c in conditions.items():
            st_ = self.state.setdefault(key, {"first_t": t, "first_frame": frame_idx, "last_t": t, "last_frame": frame_idx, "event": None})
            st_["last_t"] = t
            st_["last_frame"] = frame_idx
            st_["grace"] = c.get("grace", self.grace)
            
            frame_duration_count = st_["last_frame"] - st_["first_frame"]
            time_duration = t - st_["first_t"]

            if st_["event"] is None:
                if frame_duration_count >= self.min_frame_count and time_duration >= c["min_sec"]:
                    ev = {
                        "id": len(self.events) + 1,
                        "student": c["student"],
                        "severity": c["severity"],
                        "category": c["category"],
                        "event": c["label"],
                        "start_sec": round(st_["first_t"], 2),
                        "end_sec": round(t, 2),
                        "duration_sec": round(time_duration, 2),
                        "frame": frame_idx,
                        "evidence": None,
                        "box": c.get("box", (0, 0, 0, 0))
                    }
                    self.events.append(ev)
                    st_["event"] = ev
                    new_events.append(ev)
            else:
                ev = st_["event"]
                ev["end_sec"] = round(t, 2)
                ev["duration_sec"] = round(t - ev["start_sec"], 2)
                if ev["student"] is None and c["student"] is not None:
                    ev["student"], ev["event"] = c["student"], c["label"]
                if SEV_ORDER[c["severity"]] > SEV_ORDER[ev["severity"]]:
                    ev["severity"], ev["event"] = c["severity"], c["label"]

        for key in list(self.state):
            if key not in conditions and t - self.state[key]["last_t"] > self.state[key].get("grace", self.grace):
                del self.state[key]

        active = [s["event"] for k, s in self.state.items() if k in conditions and s["event"] is not None]
        return new_events, active

    def attach_evidence(self, ev, frame):
        if len(self.evidence) >= MAX_EVIDENCE:
            return
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            ev["evidence"] = len(self.evidence)
            self.evidence.append(buf.tobytes())


def consolidate_student_ids(events):
    if not events:
        return events
    student_boxes = {}
    for ev in events:
        sid = ev.get("student")
        if sid is not None:
            student_boxes.setdefault(sid, []).append(ev.get("box", (0, 0, 0, 0)))

    student_centers = {}
    for sid, boxes in student_boxes.items():
        if boxes:
            arr = np.array(boxes)
            cx = np.mean((arr[:, 0] + arr[:, 2]) / 2)
            cy = np.mean((arr[:, 1] + arr[:, 3]) / 2)
            student_centers[sid] = (cx, cy)

    sids = sorted(student_centers.keys())
    mapping = {s: s for s in sids}
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            s1, s2 = sids[i], sids[j]
            c1, c2 = student_centers[s1], student_centers[s2]
            dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
            if dist < 90:
                mapping[s2] = mapping[s1]

    for ev in events:
        sid = ev.get("student")
        if sid in mapping:
            ev["student"] = mapping[sid]
    return events


# ----------------------------------------------------------------------------
# Exam Monitor Core Engine
# ----------------------------------------------------------------------------
class Settings:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ExamMonitor:
    def __init__(self, pose_model, obj_model, custom_model, settings):
        self.pose, self.obj, self.custom, self.s = pose_model, obj_model, custom_model, settings
        self.incidents = IncidentManager(min_frame_count=settings.min_frame_count)
        self.first_frame = True
        self.last_error = None
        self.yaw_hist = {}
        self.n = 0
        self.cached_objects = []
        self.obj_tracks = []
        self.next_obj_id = 1
        self.obj_ids = []
        self.custom_ids = []
        if obj_model is not None and hasattr(obj_model, "names"):
            self.obj_ids = [i for i, n in obj_model.names.items() if n in set(settings.prohibited)]
        if custom_model is not None and hasattr(custom_model, "names"):
            self.custom_ids = [i for i, n in custom_model.names.items() if n in set(settings.custom_classes)]

    def _detect_people(self, frame, track):
        if self.pose is None:
            return []
        kw = dict(imgsz=self.s.imgsz, conf=self.s.person_conf, verbose=False)
        try:
            if track:
                res = self.pose.track(frame, persist=not self.first_frame, tracker="bytetrack.yaml", **kw)
                self.first_frame = False
            else:
                res = self.pose.predict(frame, **kw)
        except Exception as exc:
            self.last_error = f"Pose model error: {exc}"
            return []
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return []
        r = res[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = (r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None
               else np.arange(1, len(xyxy) + 1))
        kps = (r.keypoints.data.cpu().numpy() if r.keypoints is not None
               else np.zeros((len(xyxy), 17, 3)))
        
        persons = []
        for box, pid, kp in zip(xyxy, ids, kps):
            x1, y1, x2, y2 = [float(v) for v in box]
            area = (x2 - x1) * (y2 - y1)
            if area >= MIN_PERSON_AREA:
                persons.append({"id": int(pid), "box": (x1, y1, x2, y2), "kp": kp, "area": area})
        
        persons = filter_clean_persons(persons)
        return persons

    def _run_detector(self, model, class_ids, conf, frame, source):
        if model is None or not class_ids:
            return []
        try:
            res = model.predict(frame, imgsz=self.s.imgsz, conf=conf, classes=class_ids, verbose=False)
        except Exception as exc:
            self.last_error = f"{source} model error: {exc}"
            return []
        if not res or res[0].boxes is None:
            return []
        out = []
        for box in res[0].boxes:
            name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
            out.append({"name": name, "conf": float(box.conf[0]), "box": (x1, y1, x2, y2), "source": source})
        return out

    def _track_objects(self, objects):
        for o in objects:
            x1, y1, x2, y2 = o["box"]
            cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
            best = None
            for tr in self.obj_tracks:
                tx1, ty1, tx2, ty2 = tr["box"]
                d = ((cx - (tx1 + tx2) / 2) ** 2 + (cy - (ty1 + ty2) / 2) ** 2) ** 0.5
                if d < max(size, tr["size"], 30) and (best is None or d < best[0]):
                    best = (d, tr)
            if best:
                tr = best[1]
                tr["box"], tr["size"] = o["box"], size
            else:
                tr = {"id": self.next_obj_id, "box": o["box"], "size": size}
                self.next_obj_id += 1
                self.obj_tracks.append(tr)
            o["track"] = tr["id"]
        return objects

    def process(self, frame, t, frame_idx, track=True, instant=False):
        s = self.s
        persons = self._detect_people(frame, track)
        self.n += 1
        if instant or (self.n - 1) % max(1, s.object_every) == 0:
            found = (self._run_detector(self.obj, self.obj_ids, s.object_conf, frame, "coco")
                     + self._run_detector(self.custom, self.custom_ids, s.custom_conf, frame, "custom"))
            self.cached_objects = self._track_objects(dedupe_objects(found))
        objects = [dict(o) for o in self.cached_objects]
        conditions, yaw_map = {}, {}

        for p in persons:
            sid = p["id"]
            raw_yaw, _, visible = analyze_head(p["kp"], s.yaw_threshold)
            yaw = None
            if raw_yaw is not None:
                hist = self.yaw_hist.setdefault(sid, deque(maxlen=5))
                hist.append(raw_yaw)
                yaw = float(np.median(hist))
            yaw_map[sid] = yaw
            direction = None
            if yaw is not None:
                direction = "LEFT" if yaw < -s.yaw_threshold else ("RIGHT" if yaw > s.yaw_threshold else None)

            if direction and s.check_head:
                nb = find_neighbor(p, direction, persons, s.neighbor_reach)
                if nb is not None:
                    conditions[("head", sid)] = dict(
                        student=sid, severity="HIGH", category="Copying from neighbor",
                        label=f"Looking {direction} at Student #{nb} (possible copying)", 
                        min_sec=s.head_seconds, box=p["box"])
                else:
                    conditions[("head", sid)] = dict(
                        student=sid, severity="MEDIUM", category="Head turn",
                        label=f"Head turned {direction} (looking away)", 
                        min_sec=s.head_seconds, box=p["box"])

        for o in objects:
            owner = assign_owner(o["box"], persons)
            where = f"Student #{owner}" if owner is not None else "the exam area"
            o["owner"] = owner
            if o["source"] == "coco":
                label, sev = OBJECT_INFO.get(o["name"], ("Prohibited item", "HIGH"))
                category = "Prohibited item"
                text = f"{label} near {where}"
            else:
                label, sev, category = o["name"], "HIGH", "Trained-model detection"
                text = f"{o['name']} near {where}"
            o["label"] = label
            conditions[("obj", o["track"])] = dict(
                student=owner, severity=sev, category=category, label=text,
                min_sec=s.object_seconds, grace=4.0, box=o["box"])

        if instant:
            for c in conditions.values():
                c["min_sec"] = 0.0

        new_events, active = self.incidents.update(t, conditions, frame_idx)

        sev_by_student = {}
        for ev in active:
            if ev["student"] is not None and SEV_ORDER[ev["severity"]] > SEV_ORDER[sev_by_student.get(ev["student"], "NONE")]:
                sev_by_student[ev["student"]] = ev["severity"]

        annotated = self._annotate(frame, persons, objects, sev_by_student, yaw_map)
        for ev in new_events:
            self.incidents.attach_evidence(ev, annotated)
        return annotated, {"students": len(persons), "active": active, "new": new_events, "persons": persons}

    def _annotate(self, frame, persons, objects, sev_by_student, yaw_map):
        out = frame.copy()
        scale = max(0.5, out.shape[1] / 1280 * 0.7)
        th = max(2, int(scale * 3))
        for p in persons:
            sid = p["id"]
            sev = sev_by_student.get(sid, "NONE")
            color = SEV_COLOR[sev]
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, th if sev != "NONE" else 2)
            text = f"#{sid}" + (f" {sev}" if sev != "NONE" else "")
            if self.s.show_debug and yaw_map.get(sid) is not None:
                text += f"  yaw {yaw_map[sid]:+.2f}"
            cv2.putText(out, text, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)
        for o in objects:
            x1, y1, x2, y2 = [int(v) for v in o["box"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), th)
            owner = f" -> #{o['owner']}" if o["owner"] is not None else ""
            cv2.putText(out, f"{o['label']} {o['conf']:.2f}{owner}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), th)
        return out


# ----------------------------------------------------------------------------
# Model Loader & Dataset Scanners
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading detection models...")
def load_model(name, fallback=None):
    if not YOLO_AVAILABLE:
        return None, "ultralytics is not installed"
    err = None
    for candidate in (name, fallback):
        if not candidate:
            continue
        try:
            return YOLO(candidate), None
        except Exception as exc:
            err = str(exc)
    return None, err


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


def resolve_url(url):
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
            return None, None, f"Could not open link: {exc}"

    tmp = tempfile.mkdtemp()
    try:
        import yt_dlp
        opts = {
            "format": "best[height<=720][ext=mp4]/best",
            "outtmpl": os.path.join(tmp, "video.%(ext)s"),
            "quiet": True, "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info), tmp, None
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        return None, None, f"Could not download link ({exc})."


# ----------------------------------------------------------------------------
# UI Live Rendering & Reports
# ----------------------------------------------------------------------------
def alarm_html():
    t_val = time.time()
    return f"""
    <audio autoplay data-t="{t_val}">
        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
    </audio>
    """


def render_live(widgets, info, monitor, current_student_count):
    active = info["active"]
    top = max((SEV_ORDER[e["severity"]] for e in monitor.incidents.events), default=0)
    top_name = {0: "None", 1: "Low", 2: "Medium", 3: "High"}[top]
    
    temp_events = list(monitor.incidents.events)
    consolidate_student_ids(temp_events)
    
    flagged_students = {e["student"] for e in temp_events if e["student"] is not None and SEV_ORDER[e["severity"]] >= 2}

    with widgets["metrics"].container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Students detected", current_student_count)
        c2.metric("Active incidents", len(active))
        c3.metric("Students flagged (cheating)", len(flagged_students))
        c4.metric("Highest risk so far", top_name)
        
    with widgets["alerts"].container():
        if active:
            for ev in sorted(active, key=lambda e: -SEV_ORDER[e["severity"]]):
                who = f"Student #{ev['student']}" if ev["student"] is not None else "Exam area"
                st.error(f"{SEV_ICON[ev['severity']]} - **{who}** - {ev['event']}")
        else:
            st.success("No active incidents. Exam environment is clean.")


def build_result(monitor, frames, source_name, final_student_count):
    consolidate_student_ids(monitor.incidents.events)
    return {
        "events": list(monitor.incidents.events),
        "evidence": list(monitor.incidents.evidence),
        "frames": frames,
        "source": source_name,
        "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_students": final_student_count,
    }


def analyze_video(source, source_name, monitor_args, perf):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Could not open video source.")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    max_frames = int(perf.max_seconds * fps) if perf.max_seconds else 0
    monitor = ExamMonitor(*monitor_args)
    frame_slot, progress, status = st.empty(), st.progress(0.0), st.empty()
    widgets = {"metrics": st.empty(), "alerts": st.empty()}
    sound_slot = st.empty()
    frame_idx, processed, last_sound, started = 0, 0, 0.0, time.time()
    
    latest_student_count = 0

    while True:
        if not cap.grab():
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break
        if frame_idx % perf.frame_skip:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        h, w = frame.shape[:2]
        if w > perf.max_width:
            frame = cv2.resize(frame, (perf.max_width, int(h * perf.max_width / w)), interpolation=cv2.INTER_AREA)

        annotated, info = monitor.process(frame, frame_idx / fps, frame_idx, track=True)
        processed += 1
        
        latest_student_count = get_stable_student_count(info.get("persons", []))

        if not perf.turbo and (processed % perf.preview_every == 0 or info["new"]):
            frame_slot.image(annotated, channels="BGR", output_format="JPEG", use_container_width=True)
            render_live(widgets, info, monitor, latest_student_count)
        if total > 0 and processed % 5 == 0:
            progress.progress(min(frame_idx / total, 1.0))

        if perf.sound and info["new"] and time.time() - last_sound > 3:
            sound_slot.markdown(alarm_html(), unsafe_allow_html=True)
            last_sound = time.time()

    cap.release()
    progress.progress(1.0)
    elapsed = max(time.time() - started, 1e-6)
    status.success(f"Analysis completed in {elapsed:.1f}s - {processed} frames processed.")
    return build_result(monitor, frame_idx, source_name, latest_student_count)


def analyze_image(image, source_name, monitor_args):
    monitor = ExamMonitor(*monitor_args)
    annotated, info = monitor.process(image, 0.0, 1, track=False, instant=True)
    st.image(annotated, channels="BGR", output_format="JPEG", use_container_width=True)
    exact_count = get_stable_student_count(info.get("persons", []))
    render_live({"metrics": st.empty(), "alerts": st.empty()}, info, monitor, exact_count)
    return build_result(monitor, 1, source_name, exact_count)


def show_report(result):
    st.markdown("---")
    st.header("Comprehensive Exam Analytics & Incident Report")
    st.caption(f"Source: {result['source']}  |  Total Students Detected: {result.get('total_students', 0)}  |  Finished: {result['finished']}")

    events = result["events"]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", result.get("total_students", 0))
    c2.metric("Total Incidents", len(events))

    if not events:
        c3.metric("High Severity", 0)
        c4.metric("Medium Severity", 0)
        st.success("Clean exam session! No cheating incidents or prohibited items were recorded in this video.")
        return

    df = pd.DataFrame(events)
    df["student"] = df["student"].apply(lambda v: f"#{int(v)}" if pd.notna(v) else "Exam area")
    df["start"] = df["start_sec"].apply(fmt_time)
    df["severity_icon"] = df["severity"].map(lambda s: f"{SEV_ICON.get(s, '')} {s}")

    high, medium = int((df["severity"] == "HIGH").sum()), int((df["severity"] == "MEDIUM").sum())
    c3.metric("High Severity", high)
    c4.metric("Medium Severity", medium)
    
    flagged_df = df[(df["student"] != "Exam area") & (df["severity"].isin(["HIGH", "MEDIUM"]))]
    flagged_students_list = sorted(flagged_df["student"].unique().tolist())
    flagged = len(flagged_students_list)

    if flagged:
        st.error(f"Alert: {flagged} student(s) flagged for cheating violations: {', '.join(str(s) for s in flagged_students_list)}")
    else:
        st.info("Only minor observations recorded.")

    st.subheader("Advanced Student Risk Analytics")
    rank = df.groupby("student").agg(
        total_incidents=("id", "count"),
        high_severity=("severity", lambda s: int((s == "HIGH").sum())),
        medium_severity=("severity", lambda s: int((s == "MEDIUM").sum())),
        total_duration_sec=("duration_sec", "sum"),
        violation_types=("event", lambda s: " | ".join(sorted(set(s)))),
    ).reset_index()
    
    rank["risk_score"] = rank["high_severity"] * 3 + rank["medium_severity"] * 2
    rank = rank.sort_values(by=["risk_score", "total_duration_sec"], ascending=False)
    st.dataframe(rank, use_container_width=True, hide_index=True)

    st.subheader("Incident Timeline")
    cols = ["id", "start", "duration_sec", "student", "severity_icon", "category", "event"]
    st.dataframe(df[cols].sort_values("id"), use_container_width=True, hide_index=True, height=320)

    shots = [(row, result["evidence"][int(row["evidence"])]) for _, row in df.iterrows()
             if pd.notna(row["evidence"]) and int(row["evidence"]) < len(result["evidence"])]
    if shots:
        st.subheader("Evidence Snapshots")
        grid = st.columns(3)
        for i, (row, jpg) in enumerate(shots):
            grid[i % 3].image(jpg, caption=f"#{row['id']} {row['severity']} - {row['student']} @ {row['start']}")

    export = df.drop(columns=["evidence", "severity_icon"], errors="ignore")
    st.download_button("Download Analytics Report (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name=f"exam_analytics_report_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")


# ----------------------------------------------------------------------------
# Streamlit Interface
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Universal Exam Cheating Detection System", page_icon="🎓", layout="wide")
st.title(f"Universal Exam Cheating Detection & Analytics System - {APP_VERSION}")
st.caption("Production-ready pipeline supporting universal video formats, robust confidence tuning, and sequential frame counter.")

st.sidebar.header("Speed Preset")
preset = st.sidebar.selectbox("Speed preset", list(PRESETS), index=0)
PS = PRESETS[preset]

st.sidebar.header("Detection Engines")
use_custom = st.sidebar.checkbox("1. Custom trained model (best.pt)", value=False)
custom_path = st.sidebar.text_input("Weights path", "weights/best.pt")
custom_upload = st.sidebar.file_uploader("...or upload best.pt", type=["pt"])
use_pose = st.sidebar.checkbox("2. Pose model (Head tracking)", value=True)
pose_name = st.sidebar.selectbox("Pose weights", ["yolov8n-pose.pt", "yolo11n-pose.pt"])
use_objects = st.sidebar.checkbox("3. Generic objects (Phones / Books)", value=True)
obj_name = st.sidebar.selectbox("Object weights", ["yolov8s.pt", "yolo11s.pt"])
imgsz = st.sidebar.select_slider("Inference size", options=[480, 640, 800, 960], value=PS["imgsz"])

custom_model, custom_err, custom_classes = None, None, []
if use_custom:
    weights_file = None
    if custom_upload is not None:
        weights_file = os.path.join(tempfile.gettempdir(), f"custom_{custom_upload.size}_{custom_upload.name}")
        if not os.path.exists(weights_file):
            with open(weights_file, "wb") as fh:
                fh.write(custom_upload.getvalue())
    elif custom_path and os.path.exists(custom_path):
        weights_file = custom_path
    if weights_file:
        custom_model, custom_err = load_model(weights_file)
if custom_model is not None and hasattr(custom_model, "names"):
    all_names = list(custom_model.names.values())
    default_cls = [n for n in all_names if not any(w in n.lower() for w in NORMAL_WORDS)]
    custom_classes = st.sidebar.multiselect("Trained classes counted as cheating", all_names, default=default_cls)

pose_model, pose_err = load_model(pose_name, "yolov8n-pose.pt") if use_pose else (None, None)
obj_model, obj_err = load_model(obj_name, "yolov8s.pt") if use_objects else (None, None)

st.sidebar.header("Detection Rules & Confidence")
object_conf = st.sidebar.slider("Generic item confidence (conf)", 0.10, 0.90, 0.50, 0.05)
person_conf = st.sidebar.slider("Student confidence", 0.10, 0.90, 0.35, 0.05)
custom_conf = st.sidebar.slider("Trained model confidence", 0.10, 0.90, 0.50, 0.05)
yaw_threshold = st.sidebar.slider("Head-turn threshold", 0.10, 0.80, 0.30, 0.05)
head_seconds = st.sidebar.slider("Head turn min seconds", 0.0, 5.0, 1.0, 0.5)
object_seconds = st.sidebar.slider("Item visible min seconds", 0.0, 5.0, 0.5, 0.5)
min_frame_count = st.sidebar.slider("Min consecutive frames for alert", 5, 60, 30, 5)
neighbor_reach = st.sidebar.slider("Neighbor reach", 1.0, 6.0, 2.5, 0.5)
prohibited = st.sidebar.multiselect("Prohibited items", list(OBJECT_INFO), default=list(OBJECT_INFO))
check_head = st.sidebar.checkbox("Detect head turn / looking at neighbor", value=True)
show_debug = st.sidebar.checkbox("Show debug values", value=False)

st.sidebar.header("Performance")
frame_skip = st.sidebar.slider("Process every Nth frame", 1, 10, PS["frame_skip"])
object_every = st.sidebar.slider("Run item detectors every Nth frame", 1, 5, PS["object_every"])
max_width = st.sidebar.select_slider("Max frame width", options=[480, 640, 960, 1280], value=PS["max_width"])
turbo = st.sidebar.checkbox("Turbo mode (no live video)", value=False)
preview_every = st.sidebar.slider("Live preview every Nth frame", 1, 20, PS["preview_every"])
max_seconds = st.sidebar.number_input("Analyze first N seconds (0 = all)", min_value=0, value=0, step=10)
sound_on = st.sidebar.checkbox("Alarm sound", value=True)
perf = Settings(frame_skip=frame_skip, max_width=max_width, turbo=turbo, preview_every=preview_every,
                max_seconds=max_seconds, sound=sound_on)

st.sidebar.markdown("---")
st.sidebar.header("Video Source (Data & External)")
source_option = st.sidebar.selectbox("Choose source:", ("Upload from device", "Internet link", "Dataset folders (Kaggle / Local)", "Device camera"))

settings = Settings(
    imgsz=imgsz, person_conf=person_conf, object_conf=object_conf, custom_conf=custom_conf,
    yaw_threshold=yaw_threshold, object_every=object_every, head_seconds=head_seconds, object_seconds=object_seconds,
    min_frame_count=min_frame_count, neighbor_reach=neighbor_reach, prohibited=prohibited,
    custom_classes=custom_classes, check_head=check_head, show_debug=show_debug,
)
monitor_args = (pose_model, obj_model, custom_model, settings)

def store(result):
    if result is not None:
        st.session_state["report"] = result

if source_option == "Upload from device":
    st.markdown("### Upload exam video from device")
    # تم فتح جميع الصيغ الشائعة هنا لضمان قبول الـ webm وغيرها
    uploaded = st.file_uploader("Upload video (MP4, AVI, MOV, MKV, WEBM, FLV, WMV)", type=[e.strip(".") for e in VIDEO_EXTENSIONS])
    if uploaded and st.button("Start analysis", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            store(analyze_video(tmp_path, uploaded.name, monitor_args, perf))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

elif source_option == "Internet link":
    st.markdown("### Analyze video from internet link")
    url = st.text_input("Video link:")
    if url and st.button("Start analysis", type="primary"):
        with st.spinner("Downloading and preparing video..."):
            source, tmp_dir, err = resolve_url(url)
        if err:
            st.error(err)
        else:
            try:
                store(analyze_video(source, url, monitor_args, perf))
            finally:
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

elif source_option == "Dataset folders (Kaggle / Local)":
    st.markdown("### Analyze files from Kaggle Input / Local Dataset")
    dirs_text = st.sidebar.text_area("Dataset folders", value="\n".join(DEFAULT_DATASET_DIRS), height=110)
    search_dirs = tuple(d for d in dirs_text.splitlines() if d.strip())
    videos, images = find_dataset_files(search_dirs)
    st.write(f"Found **{len(videos)}** video(s) and **{len(images)}** image(s).")

    kind = st.radio("File type", ["Videos", "Images"], horizontal=True)
    files = videos if kind == "Videos" else images
    if files:
        selected = st.selectbox("Select a file:", files)
        if kind == "Videos":
            if st.button("Start analysis", type="primary"):
                store(analyze_video(selected, Path(selected).name, monitor_args, perf))
        else:
            img = cv2.imread(selected)
            if img is not None and st.button("Analyze image", type="primary"):
                store(analyze_image(img, Path(selected).name, monitor_args))

else:
    st.markdown("### Device camera snapshot")
    snapshot = st.camera_input("Take a picture for verification")
    if snapshot is not None:
        image = cv2.imdecode(np.frombuffer(snapshot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        store(analyze_image(image, "Camera snapshot", monitor_args))

if "report" in st.session_state:
    show_report(st.session_state["report"])
