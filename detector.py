"""Cheating detection logic: YOLO object detection + MediaPipe head-turn analysis.

This module has no Streamlit dependency, so it can be reused in notebooks or scripts.
"""

import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


def load_yolo_model(weights="yolo11n.pt"):
    """Load YOLO weights, falling back to yolov8n.pt. Returns None if unavailable."""
    if not YOLO_AVAILABLE:
        return None
    for candidate in (weights, "yolov8n.pt"):
        try:
            return YOLO(candidate)
        except Exception:
            continue
    return None


class CheatingDetector:
    def __init__(self, model_path="yolo11n.pt", conf_threshold=0.45,
                 head_threshold=0.15, max_faces=5, model=None):
        self.model = model if model is not None else load_yolo_model(model_path)
        self.conf_threshold = conf_threshold
        self.head_threshold = head_threshold
        self.face_mesh = None

        if MEDIAPIPE_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=max_faces,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
            except Exception:
                self.face_mesh = None

    # ------------------------------------------------------------------
    def estimate_head_pose(self, frame):
        """Return (status_message, is_alert) from the nose offset between the eyes."""
        if self.face_mesh is None:
            return "Head analysis unavailable", False
        try:
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return "Face not visible or turned away completely", True

            for face in results.multi_face_landmarks:
                nose, left_eye, right_eye = face.landmark[1], face.landmark[33], face.landmark[263]
                eye_dist = abs(right_eye.x - left_eye.x)
                if eye_dist <= 0:
                    continue
                offset = (nose.x - (left_eye.x + right_eye.x) / 2) / eye_dist
                if offset < -self.head_threshold:
                    return "Student is looking LEFT (possible peeking at a neighbor)", True
                if offset > self.head_threshold:
                    return "Student is looking RIGHT (possible peeking at a neighbor)", True
        except Exception:
            pass
        return "Normal posture", False

    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """Run all checks on one BGR frame.

        Returns:
            frame  - annotated frame
            alerts - list of alert strings
            stats  - dict(persons, phone, material, head_alert, head_msg)
        """
        alerts = []
        stats = {"persons": 0, "phone": False, "material": False,
                 "head_alert": False, "head_msg": ""}

        if self.model is not None:
            try:
                for result in self.model(frame, verbose=False):
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf < self.conf_threshold:
                            continue
                        name = self.model.names[int(box.cls[0])]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        if name == "person" and (x2 - x1) * (y2 - y1) > 2500:
                            stats["persons"] += 1
                            self._draw(frame, (x1, y1, x2, y2), f"Person {conf:.2f}", (0, 255, 0), 2)
                        elif name in ("cell phone", "phone"):
                            stats["phone"] = True
                            self._draw(frame, (x1, y1, x2, y2), f"PHONE {conf:.2f}", (0, 0, 255), 3)
                        elif name in ("book", "laptop"):
                            stats["material"] = True
                            self._draw(frame, (x1, y1, x2, y2), f"{name.upper()} {conf:.2f}", (0, 165, 255), 3)
            except Exception:
                pass

        if stats["phone"]:
            alerts.append("🚨 Confirmed: mobile phone detected")
        if stats["material"]:
            alerts.append("📚 Confirmed: book / laptop / external material detected")

        head_msg, head_alert = self.estimate_head_pose(frame)
        stats["head_msg"], stats["head_alert"] = head_msg, head_alert
        if head_alert:
            alerts.append(f"👀 {head_msg}")

        if stats["persons"] > 1:
            alerts.append(f"👥 Group alert: {stats['persons']} people in camera view")

        return frame, alerts, stats

    # ------------------------------------------------------------------
    @staticmethod
    def _draw(frame, box, label, color, thickness):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
