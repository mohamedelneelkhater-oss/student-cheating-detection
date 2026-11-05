import cv2
import numpy as np
import streamlit as st
import tempfile
import os

# محاولة تحميل مكتبات الذكاء الاصطناعي بشكل آمن
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

st.set_page_config(page_title="نظام كشف الغش الذكي", layout="wide")

st.title("🎓 نظام كشف الغش الذكي (YOLO + Head Pose + Streamlit)")

# الشريط الجانبي لاختيار مصدر الفيديو (الكاميرا، ملفات البيانات، أو الإنترنت)
st.sidebar.header("إدارة مصادر الفيديو")
source_option = st.sidebar.selectbox(
    "اختر مصدر الفيديو للتحليل:",
    ("كاميرا الجهاز (Camera)", "ملف فيديو من الجهاز (Data)", "رابط فيديو من الإنترنت (URL)")
)

# تحميل نموذج YOLOv8 لكشف الأشياء (الهواتف، الكتب، الأشخاص)
@st.cache_resource
def load_yolo_model():
    if YOLO_AVAILABLE:
        # سيقوم النموذج بتحميل الملف تلقائياً عند أول تشغيل
        return YOLO('yolov8n.pt')
    return None

yolo_model = load_yolo_model()

class CheatingDetectorSystem:
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            self.face_mesh = None

    def estimate_head_pose(self, frame):
        """كود تتبع التفات الرأس وحركة الوجه"""
        pose_status = "طبيعي"
        is_alert = False

        if self.face_mesh is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # ملامح الأنف والعينين لتقدير الاتجاه الأفقي
                    nose = face_landmarks.landmark[1]
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]
                    
                    if nose.x < left_eye.x - 0.02:
                        pose_status = "تحذير: الطالب ملتفت يميناً!"
                        is_alert = True
                    elif nose.x > right_eye.x + 0.02:
                        pose_status = "تحذير: الطالب ملتفت يساراً!"
                        is_alert = True
            else:
                pose_status = "تحذير: الوجه مختفي عن الكاميرا!"
                is_alert = True
        return pose_status, is_alert

    def process_frame_with_yolo(self, frame):
        """كود كشف الأجسام المحظورة (هاتف، كتاب) وتتبع الأشخاص باستخدام YOLO"""
        alerts = []
        person_count = 0
        phone_count = 0
        book_count = 0

        # تحليل الأجسام عبر YOLO
        if yolo_model is not None:
            results = yolo_model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = yolo_model.names[cls_id]
                    
                    if conf > 0.35: # عتبة الثقة
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = coords

                        # 1. كشف الأشخاص
                        if class_name == 'person':
                            person_count += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 2. كشف الهواتف المحمولة
                        elif class_name in ['cell phone', 'phone']:
                            phone_count += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, f"⚠️ PHONE DETECTED!", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            alerts.append("⚠️ تنبيه خطير: تم اكتشاف هاتف محمول محظور!")

                        # 3. كشف الكتب أو الأوراق الخارجية
                        elif class_name in ['book', 'laptop']:
                            book_count += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                            cv2.putText(frame, f"⚠️ BOOK/MATERIAL", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            alerts.append("⚠️ تنبيه: تم اكتشاف كتاب أو مادة خارجية!")

        # فحص التفات الرأس
        head_msg, head_alert = self.estimate_head_pose(frame)
        if head_alert:
            alerts.append(f"⚠️ {head_msg}")

        # فحص تعدد الأشخاص
        if person_count > 1:
            alerts.append(f"⚠️ تنبيه: يوجد {person_count} أشخاص في نطاق الاختبار!")

        return frame, alerts, person_count, phone_count, book_count

detector = CheatingDetectorSystem()

# واجهة مستخدم Streamlit لعرض النتائج والإنذارات
if source_option == "كاميرا الجهاز (Camera)":
    st.markdown("### 📷 البث المباشر من كاميرا الجهاز")
    img_file_buffer = st.camera_input("التقط صورة للتحقق الفوري")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        processed_frame, alerts, p_count, phone_count, book_count = detector.process_frame_with_yolo(cv2_img)

        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("عدد الأشخاص", p_count)
        col2.metric("الهواتف المكتشفة", phone_count)
        col3.metric("الكتب والمواد", book_count)

        if alerts:
            for alert in alerts:
                st.error(alert)
        else:
            st.success("الوضع سليم: لم يتم رصد أي محاولات غش.")

elif source_option == "ملف فيديو من الجهاز (Data)":
    st.markdown("### 📁 تحليل ملف فيديو مسجل (من البيانات)")
    uploaded_file = st.file_uploader("اختر ملف فيديو (MP4, AVI)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st.video(uploaded_file)
        
        success, frame = cap.read()
        if success:
            processed_frame, alerts, p_count, phone_count, book_count = detector.process_frame_with_yolo(frame)
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("عدد الأشخاص", p_count)
            col2.metric("الهواتف المكتشفة", phone_count)
            col3.metric("الكتب والمواد", book_count)

            if alerts:
                for alert in alerts:
                    st.error(alert)
            else:
                st.success("الوضع سليم في الفيديو المحدد.")

elif source_option == "رابط فيديو من الإنترنت (URL)":
    st.markdown("### 🌐 جلب وتحليل فيديو من الإنترنت")
    video_url = st.text_input("أدخل رابط الفيديو المباشر:")

    if video_url:
        st.video(video_url)
        cap = cv2.VideoCapture(video_url)
        success, frame = cap.read()

        if success:
            processed_frame, alerts, p_count, phone_count, book_count = detector.process_frame_with_yolo(frame)
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("عدد الأشخاص", p_count)
            col2.metric("الهواتف المكتشفة", phone_count)
            col3.metric("الكتب والمواد", book_count)

            if alerts:
                for alert in alerts:
                    st.error(alert)
            else:
                st.success("الوضع سليم.")
