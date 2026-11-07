"""
Exam Cheating Detection System  -  v3.0  (single-file Streamlit app)

Three detection engines that work together
  1. Custom model : your own trained YOLO weights (best.pt from the Roboflow cheating dataset)
  2. Pose model   : every student gets a tracked ID + head / hand keypoints (head turn, neighbor, hands)
  3. Object model : generic COCO detector for phone / book / laptop / other devices

Video sources: device upload, internet link (direct file or YouTube etc.), original dataset folders.
Automated alerts are indicators for human review, not proof of cheating.
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

APP_VERSION = "v3.6 FAST"

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET_DIRS = ["samples", "data", "dataset", "/kaggle/input"]
PAGE_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "facebook.com", "tiktok.com", "dailymotion.com")

KP_CONF = 0.35
HEAD_DOWN_RATIO = 0.25   # (shoulder line - nose) / shoulder width below this = head down
STANDING_RATIO = 1.35    # box height vs the student's own baseline height

# Speed presets: inference size, frame skip, run object detectors every Nth processed frame,
# max frame width before inference, live-preview interval
PRESETS = {
    "⚡ Fast": dict(imgsz=640, frame_skip=5, object_every=3, max_width=640, preview_every=5),
    "⚖️ Balanced": dict(imgsz=800, frame_skip=3, object_every=2, max_width=960, preview_every=3),
    "🎯 Accurate": dict(imgsz=960, frame_skip=2, object_every=1, max_width=1280, preview_every=2),
}
MIN_PERSON_AREA = 1500
MAX_EVIDENCE = 60
NORMAL_WORDS = ("normal", "not_cheat", "not cheat", "no_cheat", "non_cheat", "non-cheat", "honest", "clean")

OBJECT_INFO = {
    "cell phone": ("Mobile phone", "HIGH"),
    "book": ("Book / notes", "HIGH"),
    "laptop": ("Laptop / tablet", "HIGH"),
    "remote": ("Unknown electronic device", "MEDIUM"),
}
SEV_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
SEV_COLOR = {"NONE": (0, 200, 0), "LOW": (0, 220, 255), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}
SEV_ICON = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴"}


# ----------------------------------------------------------------------------
# Pure geometry / rule helpers
# ----------------------------------------------------------------------------
def inside(box, x, y):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def analyze_head(kp, yaw_threshold, min_conf=KP_CONF):
    """Head turn from COCO keypoints -> (yaw, direction, face_visible).

    yaw = nose offset from the eye midpoint / eye distance (about 0 when frontal).
    direction is "LEFT"/"RIGHT" in image coordinates, or None.
    """
    nose, l_eye, r_eye = kp[0], kp[1], kp[2]
    if nose[2] < min_conf:
        return None, None, False
    l_ok, r_ok = l_eye[2] >= min_conf, r_eye[2] >= min_conf
    yaw = None
    if l_ok and r_ok:
        eye_dist = abs(r_eye[0] - l_eye[0])
        if eye_dist > 1e-3:
            yaw = float((nose[0] - (l_eye[0] + r_eye[0]) / 2) / eye_dist)
    elif l_ok != r_ok:  # profile view: one eye visible
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
    """Nearest student on the side the person is looking toward (or None)."""
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
        if abs(dx) > reach * w or abs(dy) > 0.8 * h:
            continue
        if best is None or abs(dx) < best[1]:
            best = (o["id"], abs(dx))
    return best[0] if best else None


def assign_owner(obj_box, persons):
    """Closest student to the object, among students whose box contains it or is near it."""
    cx, cy = (obj_box[0] + obj_box[2]) / 2, (obj_box[1] + obj_box[3]) / 2
    best, best_d = None, None
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        d = (((x1 + x2) / 2 - cx) ** 2 + ((y1 + y2) / 2 - cy) ** 2) ** 0.5
        if inside(p["box"], cx, cy) or d < (x2 - x1) * 1.2:
            if best_d is None or d < best_d:
                best, best_d = p["id"], d
    return best


def same_item(a, b, thr=0.5):
    """True when two boxes cover the same physical item (high IoU, or one mostly inside the other)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return False
    return inter / union > thr or inter / max(min(area_a, area_b), 1e-6) > 0.8


def dedupe_objects(objects):
    """One detection per physical item: drops overlapping boxes (book vs laptop, trained vs generic model)."""
    kept = []
    for o in sorted(objects, key=lambda d: -d["conf"]):
        if not any(same_item(o["box"], k["box"]) for k in kept):
            kept.append(o)
    return kept


def fmt_time(seconds):
    seconds = max(float(seconds), 0.0)
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


# ----------------------------------------------------------------------------
# Incident manager
# ----------------------------------------------------------------------------
class IncidentManager:
    """Turns per-frame conditions into sustained incidents with start/end/duration."""

    def __init__(self, grace=1.2):
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


# ----------------------------------------------------------------------------
# Monitor (runs the three engines + rules)
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
        self.base_h = {}
        self.obj_ids = []
        self.custom_ids = []
        if obj_model is not None:
            self.obj_ids = [i for i, n in obj_model.names.items() if n in set(settings.prohibited)]
        if custom_model is not None:
            self.custom_ids = [i for i, n in custom_model.names.items() if n in set(settings.custom_classes)]

    # ---- engines -----------------------------------------------------------
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
            if (x2 - x1) * (y2 - y1) >= MIN_PERSON_AREA:
                persons.append({"id": int(pid), "box": (x1, y1, x2, y2), "kp": kp})
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

    def _extra_rules(self, p, sid, conditions):
        """Looking backward, head down, standing up, hand-to-ear (phone-call posture)."""
        s, kp = self.s, p["kp"]
        ok = lambda i: kp[i][2] >= KP_CONF
        x1, y1, x2, y2 = p["box"]
        shoulders = ok(5) and ok(6)
        sw = abs(kp[5][0] - kp[6][0]) if shoulders else (x2 - x1) * 0.5

        if s.check_backward and shoulders and not (ok(0) or ok(1) or ok(2)):
            conditions[("back", sid)] = dict(
                student=sid, severity="MEDIUM", category="Looking backward",
                label="Turned away / looking backward (face not visible)", min_sec=s.backward_seconds)

        if s.check_head_down and shoulders and ok(0) and sw > 1:
            ratio = ((kp[5][1] + kp[6][1]) / 2 - kp[0][1]) / sw
            if ratio < HEAD_DOWN_RATIO:
                conditions[("down", sid)] = dict(
                    student=sid, severity="LOW", category="Head down",
                    label="Head down for a long time (possible hidden phone / notes)", min_sec=8.0)

        if s.check_standing:
            hist = self.base_h.setdefault(sid, [])
            height = y2 - y1
            if len(hist) < 10:
                hist.append(height)  # the first frames define the seated height of this student
            elif height > float(np.median(hist)) * STANDING_RATIO:
                conditions[("stand", sid)] = dict(
                    student=sid, severity="MEDIUM", category="Standing / leaving seat",
                    label="Student standing up or leaving seat", min_sec=1.5)

        if s.check_phone_pose:
            for wi in (9, 10):
                if kp[wi][2] < 0.5:
                    continue
                for fi in (0, 3, 4):  # nose, ears
                    if ok(fi) and ((kp[wi][0] - kp[fi][0]) ** 2 + (kp[wi][1] - kp[fi][1]) ** 2) ** 0.5 < 0.45 * sw:
                        conditions[("phonepose", sid)] = dict(
                            student=sid, severity="MEDIUM", category="Phone-use posture",
                            label="Hand held to ear / face (possible phone call)", min_sec=2.0)

    def _track_objects(self, objects):
        """Give every physical item a stable id by position, so one book = one incident."""
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

    # ---- main entry --------------------------------------------------------
    def process(self, frame, t, frame_idx, track=True, instant=False):
        s = self.s
        persons = self._detect_people(frame, track)
        # Object detectors are the slowest part: run them every Nth processed frame and reuse the result
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
                yaw = float(np.median(hist))  # smoothing against keypoint jitter
            yaw_map[sid] = yaw
            direction = None
            if yaw is not None:
                direction = "LEFT" if yaw < -s.yaw_threshold else ("RIGHT" if yaw > s.yaw_threshold else None)

            if direction and s.check_head:
                nb = find_neighbor(p, direction, persons, s.neighbor_reach)
                if nb is not None:
                    conditions[("head", sid)] = dict(
                        student=sid, severity="HIGH", category="Copying from neighbor",
                        label=f"Looking {direction} at Student #{nb} (possible copying)", min_sec=s.head_seconds)
                else:
                    conditions[("head", sid)] = dict(
                        student=sid, severity="MEDIUM", category="Head turn",
                        label=f"Head turned {direction}", min_sec=s.head_seconds)

            self._extra_rules(p, sid, conditions)

            if s.check_hands:
                for wi in (9, 10):
                    wx, wy, wc = p["kp"][wi]
                    if wc < 0.5 or inside(p["box"], wx, wy):
                        continue
                    for o in persons:
                        if o["id"] != sid and inside(o["box"], wx, wy):
                            conditions[("hand", sid)] = dict(
                                student=sid, severity="MEDIUM", category="Hand toward neighbor",
                                label=f"Hand reaching toward Student #{o['id']} (possible passing items)",
                                min_sec=s.hand_seconds)
                            break

        for o in objects:
            owner = assign_owner(o["box"], persons)
            where = f"Student #{owner}" if owner is not None else "the exam area"
            o["owner"] = owner
            if o["source"] == "coco":
                label, sev = OBJECT_INFO.get(o["name"], (o["name"], "MEDIUM"))
                category = "Prohibited item"
                text = f"{label} near {where}"
            else:
                label, sev, category = o["name"], "HIGH", "Trained-model detection"
                text = f"{o['name']} (trained model) near {where}"
            o["label"] = label
            conditions[("obj", o["track"])] = dict(
                student=owner, severity=sev, category=category, label=text,
                min_sec=s.object_seconds, grace=4.0)

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
        return annotated, {"students": len(persons), "active": active, "new": new_events}

    # ---- drawing -----------------------------------------------------------
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
            for ki in range(5):
                kx, ky, kc = p["kp"][ki]
                if kc >= KP_CONF:
                    cv2.circle(out, (int(kx), int(ky)), 3, (255, 255, 0), -1)
        for o in objects:
            x1, y1, x2, y2 = [int(v) for v in o["box"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), th)
            owner = f" -> #{o['owner']}" if o["owner"] is not None else ""
            cv2.putText(out, f"{o['label']} {o['conf']:.2f}{owner}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), th)
        return out


# ----------------------------------------------------------------------------
# Cached resources & source helpers
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
    """Return (playable_source, temp_dir_or_None, error_or_None)."""
    url = url.strip()
    lower = url.lower().split("?")[0]
    if lower.endswith(tuple(VIDEO_EXTENSIONS)) and not any(h in url for h in PAGE_HOSTS):
        cap = cv2.VideoCapture(url)
        ok = cap.isOpened()
        cap.release()
        if ok:
            return url, None, None
        tmp = tempfile.mkdtemp()  # some hosts need a real download
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
        opts = {
            "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
            "outtmpl": os.path.join(tmp, "video.%(ext)s"),
            "quiet": True, "noplaylist": True, "max_filesize": 300 * 1024 * 1024,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info), tmp, None
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        return None, None, (f"Could not download this link ({exc}). "
                            "Sites like YouTube often block cloud servers - use a direct .mp4 link or upload the file.")


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def alarm_html():
    return (f'<audio autoplay data-t="{time.time()}">'
            '<source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" '
            'type="audio/mpeg"></audio>')


def render_live(widgets, info, monitor):
    active = info["active"]
    top = max((SEV_ORDER[e["severity"]] for e in active), default=0)
    top_name = {0: "None ✅", 1: "Low 🟡", 2: "Medium 🟠", 3: "High 🔴"}[top]
    with widgets["metrics"].container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Students detected", info["students"])
        c2.metric("Active incidents", len(active))
        flagged_live = {e["student"] for e in monitor.incidents.events
                        if e["student"] is not None and SEV_ORDER[e["severity"]] >= 2}
        c3.metric("Students flagged (cheating)", len(flagged_live))
        c4.metric("Current risk level", top_name)
    with widgets["alerts"].container():
        if active:
            for ev in sorted(active, key=lambda e: -SEV_ORDER[e["severity"]]):
                who = f"Student #{ev['student']}" if ev["student"] is not None else "Exam area"
                st.error(f"{SEV_ICON[ev['severity']]} **{who}** - {ev['event']}")
        else:
            st.success("✨ No active incidents.")


def build_result(monitor, frames, source_name):
    return {
        "events": list(monitor.incidents.events),
        "evidence": list(monitor.incidents.evidence),
        "frames": frames,
        "source": source_name,
        "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def analyze_video(source, source_name, monitor_args, perf):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Could not open the video source. Check the file path or link.")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    max_frames = int(perf.max_seconds * fps) if perf.max_seconds else 0
    monitor = ExamMonitor(*monitor_args)
    frame_slot, progress, status = st.empty(), st.progress(0.0), st.empty()
    widgets = {"metrics": st.empty(), "alerts": st.empty()}
    sound_slot = st.empty()
    frame_idx, processed, last_sound, started = 0, 0, 0.0, time.time()

    while True:
        if not cap.grab():  # grab() skips decoding work for frames we do not analyze
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
        if w > perf.max_width:  # smaller frames = faster inference, faster drawing, faster upload to browser
            frame = cv2.resize(frame, (perf.max_width, int(h * perf.max_width / w)), interpolation=cv2.INTER_AREA)

        annotated, info = monitor.process(frame, frame_idx / fps, frame_idx, track=True)
        processed += 1

        if not perf.turbo and (processed % perf.preview_every == 0 or info["new"]):
            frame_slot.image(annotated, channels="BGR", output_format="JPEG", use_container_width=True)
            render_live(widgets, info, monitor)
        if total > 0 and processed % 5 == 0:
            progress.progress(min(frame_idx / total, 1.0))
        if processed % 10 == 0:
            speed = processed / max(time.time() - started, 1e-6)
            status.caption(f"Analyzed {processed} frames | {speed:.1f} fps | incidents so far: {len(monitor.incidents.events)}")

        if perf.sound and info["new"] and time.time() - last_sound > 3:
            sound_slot.markdown(alarm_html(), unsafe_allow_html=True)
            last_sound = time.time()

    cap.release()
    progress.progress(1.0)
    elapsed = max(time.time() - started, 1e-6)
    status.success(f"Done in {elapsed:.1f}s - {processed} frames analyzed ({processed / elapsed:.1f} fps)")
    if monitor.last_error:
        st.warning(monitor.last_error)
    return build_result(monitor, frame_idx, source_name)


def analyze_image(image, source_name, monitor_args):
    monitor = ExamMonitor(*monitor_args)
    annotated, info = monitor.process(image, 0.0, 1, track=False, instant=True)
    st.image(annotated, channels="BGR", output_format="JPEG", use_container_width=True)
    render_live({"metrics": st.empty(), "alerts": st.empty()}, info, monitor)
    if monitor.last_error:
        st.warning(monitor.last_error)
    return build_result(monitor, 1, source_name)


def show_report(result):
    st.markdown("---")
    st.header("📋 Incident Report")
    st.caption(f"Source: {result['source']}  |  Frames read: {result['frames']}  |  Finished: {result['finished']}")
    st.caption("Automated alerts are indicators for human review, not proof of cheating.")

    events = result["events"]
    if not events:
        st.success("No incidents were recorded.")
        return

    df = pd.DataFrame(events)
    df["student"] = df["student"].apply(lambda v: f"#{int(v)}" if pd.notna(v) else "Exam area")
    df["start"] = df["start_sec"].apply(fmt_time)
    df["severity_icon"] = df["severity"].map(lambda s: f"{SEV_ICON[s]} {s}")

    high, medium = int((df["severity"] == "HIGH").sum()), int((df["severity"] == "MEDIUM").sum())
    flagged_df = df[(df["student"] != "Exam area") & (df["severity"].isin(["HIGH", "MEDIUM"]))]
    flagged = flagged_df["student"].nunique()
    unassigned = int((df["student"] == "Exam area").sum())

    if flagged:
        st.error(f"⚠️ {flagged} student(s) flagged for possible cheating.")
    elif high or medium:
        st.warning("Suspicious items were detected but not linked to a specific student - review the timeline.")
    else:
        st.info("Only low-severity observations were recorded.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cheating cases (students)", flagged)
    c2.metric("Total incidents", len(df))
    c3.metric("High severity", high)
    c4.metric("Medium severity", medium)
    if unassigned:
        st.caption(f"{unassigned} incident(s) are not linked to a specific student (shown as 'Exam area').")

    st.subheader("Breakdown by type")
    kinds = df.groupby("category").agg(
        incidents=("id", "count"),
        details=("event", lambda s: " | ".join(sorted(set(s)))),
    ).reset_index()
    st.dataframe(kinds, use_container_width=True, hide_index=True)

    st.subheader("Cheating cases per student")
    rank = df.groupby("student").agg(
        incidents=("id", "count"),
        high=("severity", lambda s: int((s == "HIGH").sum())),
        medium=("severity", lambda s: int((s == "MEDIUM").sum())),
        total_seconds=("duration_sec", "sum"),
        violations=("event", lambda s: " | ".join(sorted(set(s)))),
    ).reset_index()
    rank["score"] = rank["high"] * 3 + rank["medium"] * 2 + (rank["incidents"] - rank["high"] - rank["medium"])
    st.dataframe(rank.sort_values("score", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Incident timeline")
    cols = ["id", "start", "duration_sec", "student", "severity_icon", "category", "event"]
    st.dataframe(df[cols].sort_values("id"), use_container_width=True, hide_index=True, height=320)

    shots = [(row, result["evidence"][int(row["evidence"])]) for _, row in df.iterrows()
             if pd.notna(row["evidence"]) and int(row["evidence"]) < len(result["evidence"])]
    if shots:
        st.subheader("Evidence snapshots")
        grid = st.columns(3)
        for i, (row, jpg) in enumerate(shots):
            grid[i % 3].image(jpg, caption=f"#{row['id']} {row['severity']} - {row['student']} @ {row['start']} - {row['event']}")

    export = df.drop(columns=["evidence", "severity_icon"])
    st.download_button("⬇️ Download report (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name=f"cheating_report_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")


# ----------------------------------------------------------------------------
# Page + sidebar
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Exam Cheating Detection System", page_icon="🎓", layout="wide")
st.title(f"🎓 Exam Cheating Detection & Monitoring System  ·  {APP_VERSION}")
st.caption("Trained model + pose tracking + object detection | per-student incident reports")

st.sidebar.success(f"App version {APP_VERSION}")

# --- Speed preset (applies defaults to the sliders below)
st.sidebar.header("⚡ Speed")
preset = st.sidebar.selectbox("Speed preset", list(PRESETS), index=0)
PS = PRESETS[preset]

# --- Engines
st.sidebar.header("🧠 Detection engines")
use_custom = st.sidebar.checkbox("1. Custom trained model (your best.pt)", value=True)
custom_path = st.sidebar.text_input("Weights path in repo", "weights/best.pt")
custom_upload = st.sidebar.file_uploader("...or upload best.pt", type=["pt"])
use_pose = st.sidebar.checkbox("2. Pose model (students, head turn, hands)", value=True)
pose_name = st.sidebar.selectbox("Pose weights", ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolov8n-pose.pt", "yolov8s-pose.pt"])
use_objects = st.sidebar.checkbox("3. Generic objects (phone / book / laptop)", value=True)
obj_name = st.sidebar.selectbox("Object weights", ["yolo11s.pt", "yolo11m.pt", "yolo11n.pt", "yolov8s.pt"])
imgsz = st.sidebar.select_slider("Inference size (higher = small items, slower)", options=[640, 800, 960, 1280],
                                 value=PS["imgsz"], key=f"imgsz_{preset}")

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
    else:
        st.sidebar.info("No custom weights found. Put your trained file at weights/best.pt or upload it above.")
if custom_model is not None:
    all_names = list(custom_model.names.values())
    default_cls = [n for n in all_names if not any(w in n.lower() for w in NORMAL_WORDS)]
    custom_classes = st.sidebar.multiselect("Trained classes counted as cheating", all_names, default=default_cls)
elif custom_err:
    st.sidebar.error(f"Custom model error: {custom_err}")

pose_model, pose_err = load_model(pose_name, "yolov8n-pose.pt") if use_pose else (None, None)
obj_model, obj_err = load_model(obj_name, "yolov8s.pt") if use_objects else (None, None)
if use_pose and pose_model is None:
    st.sidebar.error(f"Pose model unavailable: {pose_err}")
if use_objects and obj_model is None:
    st.sidebar.error(f"Object model unavailable: {obj_err}")

# --- Rules
st.sidebar.header("🎯 Detection rules")
person_conf = st.sidebar.slider("Student confidence", 0.10, 0.90, 0.35, 0.05)
object_conf = st.sidebar.slider("Generic item confidence", 0.10, 0.90, 0.25, 0.05)
custom_conf = st.sidebar.slider("Trained model confidence", 0.10, 0.90, 0.35, 0.05)
yaw_threshold = st.sidebar.slider("Head-turn threshold (lower = more sensitive)", 0.10, 0.80, 0.30, 0.05)
head_seconds = st.sidebar.slider("Head turn must last (sec)", 0.0, 5.0, 1.5, 0.5)
object_seconds = st.sidebar.slider("Item must be visible for (sec)", 0.0, 5.0, 0.5, 0.5)
neighbor_reach = st.sidebar.slider("Neighbor reach (x student width)", 1.0, 5.0, 2.5, 0.5)
prohibited = st.sidebar.multiselect("Prohibited generic items", list(OBJECT_INFO), default=list(OBJECT_INFO))
check_head = st.sidebar.checkbox("Detect head turn / looking at a neighbor", value=True)
check_hands = st.sidebar.checkbox("Detect hand reaching toward a neighbor", value=True)
check_backward = st.sidebar.checkbox("Looking backward / turned away", value=True)
check_standing = st.sidebar.checkbox("Standing up / leaving seat", value=True)
check_phone_pose = st.sidebar.checkbox("Hand held to ear / face (phone-call posture)", value=True)
check_head_down = st.sidebar.checkbox("Head down for a long time (experimental)", value=False)
backward_seconds = st.sidebar.slider("Looking backward must last (sec)", 1.0, 6.0, 2.0, 0.5)
show_debug = st.sidebar.checkbox("Show head-turn values on video (for tuning)", value=False)

# --- Performance
st.sidebar.header("🚀 Performance")
frame_skip = st.sidebar.slider("Process every Nth frame", 1, 10, PS["frame_skip"], key=f"skip_{preset}")
object_every = st.sidebar.slider("Run item detectors every Nth analyzed frame", 1, 6, PS["object_every"], key=f"objevery_{preset}")
max_width = st.sidebar.select_slider("Max frame width before analysis", options=[480, 640, 960, 1280],
                                     value=PS["max_width"], key=f"width_{preset}")
turbo = st.sidebar.checkbox("Turbo: no live video, report only (fastest)", value=False)
preview_every = st.sidebar.slider("Live preview every Nth analyzed frame", 1, 20, PS["preview_every"], key=f"prev_{preset}")
max_seconds = st.sidebar.number_input("Analyze only the first N seconds (0 = whole video)", min_value=0, value=0, step=10)
sound_on = st.sidebar.checkbox("Alarm sound", value=True)
perf = Settings(frame_skip=frame_skip, max_width=max_width, turbo=turbo, preview_every=preview_every,
                max_seconds=max_seconds, sound=sound_on)

# --- Source
st.sidebar.markdown("---")
st.sidebar.header("🎥 Video source")
source_option = st.sidebar.selectbox(
    "Choose a source:",
    ("Upload from device", "Internet link", "Original dataset (videos / images)", "Device camera"),
)

settings = Settings(
    imgsz=imgsz, person_conf=person_conf, object_conf=object_conf, custom_conf=custom_conf,
    yaw_threshold=yaw_threshold, object_every=object_every, head_seconds=head_seconds, object_seconds=object_seconds,
    hand_seconds=1.0, neighbor_reach=neighbor_reach, prohibited=prohibited,
    custom_classes=custom_classes, check_head=check_head, check_hands=check_hands, check_backward=check_backward, check_standing=check_standing, check_phone_pose=check_phone_pose,
    check_head_down=check_head_down, backward_seconds=backward_seconds,
    show_debug=show_debug,
)
monitor_args = (pose_model, obj_model, custom_model, settings)

if pose_model is None and obj_model is None and custom_model is None:
    st.error("No detection engine is active. Enable at least one engine in the sidebar.")
    st.stop()


def store(result):
    if result is not None:
        st.session_state["report"] = result


# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------
if source_option == "Upload from device":
    st.markdown("### 📁 Analyze a video from your device")
    uploaded = st.file_uploader("Upload a video (MP4, AVI, MOV, MKV, WEBM)", type=[e.strip(".") for e in VIDEO_EXTENSIONS])
    if uploaded and st.button("▶️ Start analysis", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            store(analyze_video(tmp_path, uploaded.name, monitor_args, perf))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

elif source_option == "Internet link":
    st.markdown("### 🌐 Analyze a video from the internet")
    st.caption("Direct file links (.mp4 ...) work best. YouTube / Vimeo links are downloaded first (may be blocked on cloud servers).")
    url = st.text_input("Video link:")
    if url and st.button("▶️ Start analysis", type="primary"):
        with st.spinner("Preparing the video..."):
            source, tmp_dir, err = resolve_url(url)
        if err:
            st.error(err)
        else:
            try:
                store(analyze_video(source, url, monitor_args, perf))
            finally:
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

elif source_option == "Original dataset (videos / images)":
    st.markdown("### 🗂️ Analyze files from your original dataset")
    dirs_text = st.sidebar.text_area("Dataset folders (one per line)", value="\n".join(DEFAULT_DATASET_DIRS), height=110)
    search_dirs = tuple(d for d in dirs_text.splitlines() if d.strip())
    if st.sidebar.button("🔄 Rescan folders"):
        find_dataset_files.clear()
    videos, images = find_dataset_files(search_dirs)
    st.write(f"Found **{len(videos)}** video(s) and **{len(images)}** image(s).")

    kind = st.radio("File type", ["Videos", "Images"], horizontal=True)
    files = videos if kind == "Videos" else images
    if not files:
        st.warning(f"No {kind.lower()} found. Put files in a `samples/` folder in the repo or edit the folder list in the sidebar.")
    else:
        query = st.text_input("Filter by name (optional)")
        filtered = [f for f in files if query.lower() in f.lower()] if query else files
        if filtered:
            selected = st.selectbox("Select a file:", filtered, format_func=lambda p: f"{Path(p).name}  -  {p}")
            if kind == "Videos":
                with st.expander("Preview original video"):
                    st.video(selected)
                if st.button("▶️ Start analysis", type="primary"):
                    store(analyze_video(selected, Path(selected).name, monitor_args, perf))
            else:
                img = cv2.imread(selected)
                if img is None:
                    st.error("Could not read this image.")
                elif st.button("▶️ Analyze image", type="primary"):
                    store(analyze_image(img, Path(selected).name, monitor_args))
        else:
            st.info("No files match your filter.")

else:
    st.markdown("### 📷 Device camera snapshot")
    st.caption("A single photo has no duration, so every detected condition is reported immediately.")
    snapshot = st.camera_input("Take a picture for instant verification")
    if snapshot is not None:
        image = cv2.imdecode(np.frombuffer(snapshot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        store(analyze_image(image, "Camera snapshot", monitor_args))

# The report lives in session_state so it survives reruns (e.g. clicking the download button).
if "report" in st.session_state:
    show_report(st.session_state["report"])
