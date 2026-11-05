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

st.set_page_config(page_title="نظام كشف الغش الذكي - YOLOv11", layout="wide")

st.title("🎓 نظام كشف الغش الذكي (YOLOv11 + Head Pose + جرس إنذار)")

# الشريط الجانبي لاختيار مصادر الفيديو
st.sidebar.header("إدارة مصادر الفيديو")
source_option = st.sidebar.selectbox(
    "اختر مصدر الفيديو للتحليل:",
    ("ملف فيديو من بياناتي (Data Set)", "رابط فيديو من الإنترنت (URL)", "كاميرا الجهاز (Camera)")
)

@st.cache_resource
def load_yolo_model():
    if YOLO_AVAILABLE:
        try:
            # تحميل أحدث نموذج YOLOv11 (سيتم تحميله تلقائياً عند التشغيل)
            return YOLO('yolo11n.pt')
        except Exception:
            try:
                # كحل بديل في حال لم يتوفر نموذج 11 النسخة الناانو
                return YOLO('yolov8n.pt')
            except Exception:
                return None
    return None

yolo_model = load_yolo_model()

class CheatingDetectorSystem:
    def __init__(self):
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            try:
                if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                    self.mp_face_mesh = mp.solutions.face_mesh
                    self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            except Exception:
                self.face_mesh = None

    def estimate_head_pose(self, frame):
        """تتبع التفات الرأس وحركة الوجه بشكل آمن"""
        pose_status = "طبيعي"
        is_alert = False

        if self.face_mesh is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
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
            except Exception:
                pass
        return pose_status, is_alert

    def process_frame_with_yolo(self, frame):
        """كشف الأجسام المحظورة باستخدام YOLOv11 وتوليد الإنذارات الدقيقة"""
        alerts = []
        person_count = 0
        phone_count = 0
        book_count = 0

        if yolo_model is not None:
            try:
                results = yolo_model(frame, verbose=False)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = yolo_model.names[cls_id]
                        
                        # رفع عتبة الثقة إلى 0.60 لمنع الإنذارات الكاذبة وعد الأجسام الوهمية
                        if conf > 0.60:
                            coords = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = coords

                            # كشف الأشخاص بشرط الحجم لتجنب العد الخاطئ للأطراف والأوراق
                            if class_name == 'person':
                                area = (x2 - x1) * (y2 - y1)
                                if area > 4000:
                                    person_count += 1
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # كشف الهواتف المحمولة
                            elif class_name in ['cell phone', 'phone']:
                                phone_count += 1
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, f"⚠️ PHONE DETECTED!", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                alerts.append("🚨 [تنبيه خطر]: تم اكتشاف هاتف محمول محظور!")

                            # كشف الكتب أو اللابتوب
                            elif class_name in ['book', 'laptop']:
                                book_count += 1
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                                cv2.putText(frame, f"⚠️ BOOK/MATERIAL", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                alerts.append("⚠️ [تنبيه]: تم اكتشاف كتاب أو مادة خارجية!")
            except Exception:
                pass

        head_msg, head_alert = self.estimate_head_pose(frame)
        if head_alert:
            alerts.append(f"⚠️ {head_msg}")

        if person_count > 1:
            alerts.append(f"🚨 [تنبيه خطر]: يوجد {person_count} أشخاص في نطاق الاختبار!")

        return frame, alerts, person_count, phone_count, book_count

detector = CheatingDetectorSystem()

# دالة لتشغيل جرس الإنذار الصوتي عبر متصفح الويب (HTML Audio Bell)
def play_alarm_sound():
    alarm_html = """
        <audio autoplay>
          <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
        </audio>
    """
    st.markdown(alarm_html, unsafe_allow_html=True)

# --- 1. خيار ملف فيديو من بياناتي (Data Set) ---
if source_option == "ملف فيديو من بياناتي (Data Set)":
    st.markdown("### 📁 تحليل ملف فيديو من مجموعة البيانات الخاصة بك (Data Set)")
    uploaded_file = st.file_uploader("قم برفع ملف فيديو (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st.video(uploaded_file)
        
        if st.button("بدء تحليل الفيديو من البيانات"):
            st.write("جاري معالجة الفيديو عبر YOLOv11...")
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, alerts, p_count, phone_count, book_count = detector.process_frame_with_yolo(frame)
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                if alerts:
                    play_alarm_sound()
            cap.release()

# --- 2. خيار رابط فيديو من الإنترنت (URL) ---
elif source_option == "رابط فيديو من الإنترنت (URL)":
    st.markdown("### 🌐 جلب وتحليل فيديو من الإنترنت عبر الرابط")
    video_url = st.text_input("أدخل رابط الفيديو المباشر (رابط ملف فيديو مباشر MP4):")

    if video_url:
        st.video(video_url)
        if st.button("بدء تحليل فيديو الإنترنت"):
            cap = cv2.VideoCapture(video_url)
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("انتهى الفيديو أو تعذر الوصول للرابط.")
                    break
                
                processed_frame, alerts, p_count, phone_count, book_count = detector.process_frame_with_yolo(frame)
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                if alerts:
                    play_alarm_sound()
            cap.release()

# --- 3. خيار كاميرا الجهاز (Camera) ---
elif source_option == "كاميرا الجهاز (Camera)":
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
            play_alarm_sound()  # تشغيل جرس الإنذار الصوتي الفوري
            for alert in alerts:
                st.error(alert)
        else:
            st.success("الوضع سليم: لم يتم رصد أي محاولات غش.")
