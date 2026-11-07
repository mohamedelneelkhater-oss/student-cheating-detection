"""
Exam Cheating Detection & Analytics System  -  v15.0 Multi-Student Pro (Single-file Streamlit app)
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

APP_VERSION = "v15.0 MULTI-STUDENT PRO"

# ----------------------------------------------------------------------------
# Constants & Settings
# ----------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET_DIRS = ["samples", "data", "dataset", "/kaggle/input"]
PAGE_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "facebook.com", "tiktok.com", "dailymotion.com")

KP_CONF = 0.35
MIN_PERSON_AREA = 700   # مساحة مثالية لضمان رصد جميع الطلاب في الصفوف الأمامية والخلفية
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
    # تم تعديل هذه الدالة للإبقاء على جميع الطلاب الذين ارتكبوا مخالفات حقيقية وعدم حذف أي منهم
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
