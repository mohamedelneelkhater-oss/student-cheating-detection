from dataclasses import dataclass
import os
from pathlib import Path
from ultralytics import YOLO


@dataclass
class DetectorConfig:
    weights: str = "models/best.pt"
    fallback_weights: str = "yolo11n.pt"  # نموذج YOLO11 الافتراضي العام
    conf: float = 0.50
    screenshot_dir: str = "alerts/screenshots"
    db_path: str = "alerts/alerts.db"


def format_notification(event):
    return f"⚠️ تم رصد سلوك مشبوه: {event.get('behavior', 'Unknown')} بنسبة ثقة {event.get('confidence', 0):.2f}"


class CheatingMonitor:

    def __init__(self, cfg: DetectorConfig, db=None, on_alert=None):
        self.cfg = cfg
        self.db = db
        self.on_alert = on_alert
        self.model = None
        self._load_model()

    def _load_model(self):
        # محاولة تحميل نموذج YOLO11 المخصص أولاً إن وجد
        if os.path.exists(self.cfg.weights):
            try:
                self.model = YOLO(self.cfg.weights)
                print(f"Loaded custom YOLO11 model from {self.cfg.weights}")
                return
            except Exception as e:
                print(f"Error loading custom model: {e}")

        # إذا لم يكن موجوداً، تحميل نموذج YOLO11 العام تلقائياً
        try:
            print(f"Loading default YOLO11 model ({self.cfg.fallback_weights})...")
            self.model = YOLO(self.cfg.fallback_weights)
        except Exception as e:
            print(f"Error loading fallback YOLO11 model: {e}")
            self.model = None

    def stream_video(self, video_path, stop_event=None):
        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_index = 0

        os.makedirs(self.cfg.screenshot_dir, exist_ok=True)

        while cap.isOpened():
            if stop_event and stop_event.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            alerts_in_frame = []

            # تحليل الفريم باستخدام YOLO11
            if self.model:
                results = self.model(
                    frame, conf=self.cfg.conf, verbose=False
                )
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = (
                            self.model.names.get(cls_id, "Suspicious")
                            if hasattr(self.model, "names")
                            else "Suspicious"
                        )

                        alert_data = {
                            "behavior": class_name,
                            "confidence": conf,
                            "screenshot_path": None,
                        }
                        alerts_in_frame.append(alert_data)

                        if self.db:
                            self.db.insert_alert(
                                behavior=class_name,
                                confidence=conf,
                                screenshot_path=None,
                            )

                        if self.on_alert:
                            self.on_alert(alert_data)

            yield {
                "frame": frame,
                "frame_index": frame_index,
                "total_frames": total_frames,
                "alerts": alerts_in_frame,
            }

        cap.release()
