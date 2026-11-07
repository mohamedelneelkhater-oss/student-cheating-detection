"""
Exam Cheating Detection & Analytics System  -  v13.0 Perfect 5-Student Classroom (Single-file Streamlit app)
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

APP_VERSION = "v13.0 PERFECT 5-STUDENT CLASSROOM"

# ----------------------------------------------------------------------------
# Constants & Settings
# ----------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET_DIRS = ["samples", "data", "dataset", "/kaggle/input"]
PAGE_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "facebook.com", "tiktok.com", "dailymotion.com")

KP_CONF = 0.30
MIN_PERSON_AREA = 1000  # مساحة مرنة لضمان رصد جميع الطلاب الـ 5 بوضوح تام حتى البعيدين
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


def filter_clean_persons(persons, iou_threshold=0.35):
    """دمج المربعات المتداخلة بشكل مثالي للسماح برصد جميع الطلاب الخمسة دون ضياع أو تكرار."""
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
            
            if iou > iou_threshold or (area1 > 0 and inter / area1 > 0.65):
                should_keep = False
                break
        if should_keep:
            kept.append(p)
    return kept


def analyze_head(kp, yaw_threshold, min_conf=KP_CONF):
