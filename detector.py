import cv2
import numpy as np
from ultralytics import YOLO

class CheatingDetector:
    def __init__(self, model_path="yolov8s.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # عدادات الإطارات المتتالية لكل طالب/حالة لتجنب الفالس آلارم
        self.head_turn_counter = 0
        self.phone_counter = 0
        self.alert_threshold = 30  # أكثر من 30 إطاراً متتالياً (حوالي ثانية إلى ثانيتين)

    def detect_objects(self, frame):
        """كشف الأجسام المحظورة مثل الهاتف والكتاب باستخدام YOLO مع conf=0.5"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detected_items = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_id]
                
                # فئات افتراضية في COCO مثل cell phone (67) أو كتاب (book)
                # يمكنك تخصيصها حسب نموذجك المدرب (مثل 'cheating' أو 'phone')
                if class_name in ['cell phone', 'book', 'cheating']:
                    detected_items.append((class_name, conf, box.xyxy[0].tolist()))
                    
        return detected_items

    def estimate_head_pose(self, frame):
        """تقدير التفات الرأس وحركة الوجه البسيطة باستخدام المعالم أو تحليل الإطار"""
        # نموذج مبسط ومستقر يعتمد على اكتشاف الوجه وزواياه أو العودة لحالة الالتفات
        # في بيئة الإنتاج يفضل استخدام MediaPipe Face Mesh، وهنا نضع منطقاً آمناً متكاملاً:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        is_turning = False
        for (x, y, w, h) in faces:
            # إذا كان الوجة مائلاً أو غير متمركز بشكل حاد (تقدير تقريبي للالتفات)
            # يمكن ربطه بإحداثيات مركز الوجه بالنسبة لإطار الكاميرا
            frame_center_x = frame.shape[1] / 2
            face_center_x = x + w / 2
            if abs(face_center_x - frame_center_x) > (frame.shape[1] * 0.25):
                is_turning = True
                
        return is_turning, faces

    def process_frame(self, frame):
        """معالجة الإطار وتطبيق العداد المتتالي (Frame Counter) لمنع الإنذارات الكاذبة"""
        h, w, _ = frame.shape
        objects = self.detect_objects(frame)
        is_turning, faces = self.estimate_head_pose(frame)
        
        alert_triggered = False
        alert_message = ""

        # فحص الهاتف أو الأجسام المحظورة مع العداد
        phone_detected_in_frame = any(item[0] in ['cell phone', 'cheating'] for item in objects)
        if phone_detected_in_frame:
            self.phone_counter += 1
        else:
            self.phone_counter = max(0, self.phone_counter - 1)

        # فحص التفات الرأس مع العداد
        if is_turning:
            self.head_turn_counter += 1
        else:
            self.head_turn_counter = max(0, self.head_turn_counter - 1)

        # إطلاق الإنذار فقط إذا تجاوز التراكم 30 إطاراً متتالياً
        if self.phone_counter >= self.alert_threshold:
            alert_triggered = True
            alert_message = "انذار: تم رصد الهاتف المحمول لفترة طويلة!"
        elif self.head_turn_counter >= self.alert_threshold:
            alert_triggered = True
            alert_message = "إنذار: تم رصد التفات متواصل للرأس!"

        # رسم الصناديق والمؤشرات على الإطار (أخضر للاعتيادي، أحمر عند الغش/الإنذار)
        box_color = (0, 0, 255) if alert_triggered else (0, 255, 0)
        
        for (x, y, w_box, h_box) in faces:
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), box_color, 2)
            
        for name, conf, coords in objects:
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} ({conf:.2f})", (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame, alert_triggered, alert_message, self.phone_counter, self.head_turn_counter
