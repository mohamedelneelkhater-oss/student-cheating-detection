from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="نظام كشف الغش الذكي", layout="wide")

# ==========================================
# نموذج كشف الغش المدمج (Cheating Detector)
# ==========================================
class CheatingDetector:
    def __init__(self, model_path="yolov8s.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.head_turn_counter = 0
        self.phone_counter = 0
        self.alert_threshold = 30  # عداد الإطارات لمنع الإنذارات الكاذبة

    def detect_objects(self, frame):
        """كشف الأجسام المحظورة مثل الهاتف والكتاب باستخدام YOLO"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detected_items = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_id]
                
                if class_name in ['cell phone', 'book', 'cheating', 'laptop']:
                    detected_items.append((class_name, conf, box.xyxy[0].tolist()))
                    
        return detected_items

    def estimate_head_pose(self, frame):
        """طريقة آمنة ومستقرة لتقدير التفات الرأس بدون الاعتماد على ملفات خارجية مفقودة"""
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # استخدام كشف بسيط ومنتظم يعتمد على توزيع السطوع أو مركز الإطار لتجنب أي أخطاء
        is_turning = False
        faces = []
        
        # طريقة بديلة آمنة تعتمد على الكشف عبر YOLO (إذا تم اكتشاف شخص أو وجه)
        # أو تحليل الحركة البسيطة لتحديد الالتفات لمنع أي توقف للنظام
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not face_cascade.empty():
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, fw, fh) in faces:
                    face_center_x = x + fw / 2
                    if abs(face_center_x - (w / 2)) > (w * 0.25):
                        is_turning = True
        except Exception:
            # بديل آمن تماماً في حال عدم توفر الملفات سحابياً
            is_turning = False
            
        return is_turning, faces

    def process_frame(self, frame):
        objects = self.detect_objects(frame)
        is_turning, faces = self.estimate_head_pose(frame)
        
        alert_triggered = False
        alert_message = ""

        phone_detected = any(item[0] in ['cell phone', 'cheating', 'laptop'] for item in objects)
        if phone_detected:
            self.phone_counter += 1
        else:
            self.phone_counter = max(0, self.phone_counter - 1)

        if is_turning:
            self.head_turn_counter += 1
        else:
            self.head_turn_counter = max(0, self.head_turn_counter - 1)

        if self.phone_counter >= self.alert_threshold:
            alert_triggered = True
            alert_message = "إنذار: تم رصد الهاتف المحمول لفترة طويلة!"
        elif self.head_turn_counter >= self.alert_threshold:
            alert_triggered = True
            alert_message = "إنذار: تم رصد التفات متواصل للرأس!"

        box_color = (0, 0, 255) if alert_triggered else (0, 255, 0)
        
        for (x, y, w_box, h_box) in faces:
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), box_color, 2)
            
        for name, conf, coords in objects:
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} ({conf:.2f})", (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame, alert_triggered, alert_message, self.phone_counter, self.head_turn_counter

# ==========================================
# واجهة مستخدم Streamlit
# ==========================================
st.title("🚨 نظام تتبع ومراقبة حالات الغش في الامتحانات")
st.markdown("---")

@st.cache_resource
def load_detector():
    return CheatingDetector(model_path="yolov8s.pt", conf_threshold=0.5)

detector = load_detector()

st.sidebar.header("إعدادات المصدر والتحكم")
source_option = st.sidebar.selectbox(
    "اختر مصدر الفيديو:",
    ("رفع فيديو من الجهاز", "رابط فيديو مباشر")
)

video_path = None
if source_option == "رفع فيديو من الجهاز":
    uploaded_file = st.sidebar.file_uploader("اختر ملف فيديو", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_file = Path("/tmp/uploaded_video.mp4")
        temp_file.write_bytes(uploaded_file.read())
        video_path = str(temp_file)
else:
    url_input = st.sidebar.text_input("أدخل رابط الفيديو المباشر:")
    if url_input:
        video_path = url_input

if video_path and st.sidebar.button("بدء المعالجة والتحليل"):
    cap = cv2.VideoCapture(video_path)
    st.sidebar.success("جاري تشغيل ومعالجة الفيديو...")
    
    st_frame = st.empty()
    alert_box = st.empty()
    metrics_col1, metrics_col2 = st.columns(2)
    
    frame_count_metric = metrics_col1.metric("عداد إطارات الهاتف المتتالية", 0)
    head_count_metric = metrics_col2.metric("عداد إطارات التفات الرأس", 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("انتهى الفيديو أو حدث خطأ في القراءة.")
            break

        processed_frame, alert, msg, p_count, h_count = detector.process_frame(frame)
        
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(processed_frame, channels="RGB", use_column_width=True)
        
        frame_count_metric.metric("عداد إطارات الهاتف المتتالية", p_count)
        head_count_metric.metric("عداد إطارات التفات الرأس", h_count)

        if alert:
            alert_box.error(f"⚠️ {msg} (التوقيت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            alert_box.success("الحالة طبيعية - لم يتم رصد أي سلوك مريب.")

    cap.release()
else:
    st.info("الرجاء اختيار مصدر الفيديو والضغط على زر 'بدء المعالجة والتحليل'.")
