"""
Exam Cheating Detection & Analytics System  -  v16.0 Final Perfect & Indentation-Fixed (Single-file Streamlit app)
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

APP_VERSION = "v16.0 FINAL PERFECT"

# ----------------------------------------------------------------------------
# Constants & Settings
# ----------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET_DIRS = ["samples", "data", "dataset", "/kaggle/input"]
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
SEV_ICON = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴"}

PRESETS = {
    "⚡ Fast": dict(imgsz=640, frame_skip=5, object_every=3, max_width=640, preview_every=5),
    "⚖️ Balanced": dict(imgsz=800, frame_skip=3, object_every=2, max_width=960, preview_every=3),
    "🎯 Accurate": dict(imgsz=960, frame_skip=2, object_every=1, max_width=1280, preview_every=2),
}


# ----------------------------------------------------------------------------
# Precise Detection & Box Fusion Helpers
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
    return inter / union > thr or inter / max(min(area_a, area_b), 1e-6) > 0.8


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
# Incident Manager
# ----------------------------------------------------------------------------
class IncidentManager:
    def __init__(self, grace=1.5):
        self.grace = grace
        self.state = {}
        self.events = []
        self.evidence = []

    def update(self, t, conditions, frame_idx):
        new_events = []
        for key, c in conditions.items():
            st_ = self.state.setdefault(key, {"first": t, "last": t, "event": None})
            st_["last"] = t
            st_["grace"] = c.get("grace", self.grace)
            if st_["event"] is None:
                if t - st_["first"] >= c["min_sec"]:
                    ev = {
                        "id": len(self.events) + 1,
                        "student": c["student"],
                        "severity": c["severity"],
                        "category": c["category"],
                        "event": c["label"],
                        "start_sec": round(st_["first"], 2),
                        "end_sec": round(t, 2),
                        "duration_sec": round(t - st_["first"], 2),
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
            if key not in conditions and t - self.state[key]["last"] > self.state[key].get("grace", self.grace):
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


def filter_exact_cheaters(events):
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
        self.incidents = IncidentManager()
        self.first_frame = True
        self.last_error = None
        self.yaw_hist = {}
        self.n = 0
        self.cached_objects = []
        self.obj_tracks = []
        self.next_obj_id = 1
        self.obj_ids = []
        self.custom_ids = []
        if obj_model is not None:
            self.obj_ids = [i for i, n in obj_model.names.items() if n in set(settings.prohibited)]
        if custom_model is not None:
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
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
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
                        label=f"Looking {direction} at Student #{nb} (possible copying/passing notes)", 
                        min_sec=s.head_seconds, box=p["box"])
                else:
                    conditions[("head", sid)] = dict(
                        student=sid, severity="MEDIUM", category="Head turn",
                        label=f"Head turned {direction} (looking away)", 
                        min_sec=s.head_seconds + 0.5, box=p["box"])

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
                text = f"{o['name']} (trained model) near {where}"
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
# Cache & Source Loader
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
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
# UI Rendering & Live Metrics
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
    top_name = {0: "None ✅", 1: "Low 🟡", 2: "Medium 🟠", 3: "High 🔴"}[top]
    
    temp_events = list(monitor.incidents.events)
    consolidate_student_ids(temp_events)
    temp_events = filter_exact_cheaters(temp_events)
    
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
                st.error(f"{SEV_ICON[ev['severity']]} **{who}** - {ev['event']}")
        else:
            st.success("✨ No active incidents.")


def build_result(monitor, frames, source_name, final_student_count):
    consolidate_student_ids(monitor.incidents.events)
    monitor.incidents.events = filter_exact_cheaters(monitor.incidents.events)
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
        st.error("Could not open video.")
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
