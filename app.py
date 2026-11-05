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
            return YOLO('yolo11n.pt')
        except Exception:
            try:
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
                    self.face_mesh = self.mp_face_mesh.FaceMesh(
                        max_num_faces=5,  # السماح بتتبع عدة وجوه في نفس الإطار
                        min_detection_confidence=0.4, 
                        min_tracking_confidence=0.4
                    )
            except Exception:
                self.face_mesh = None

    def estimate_head_pose(self, frame):
        """تتبع دقيق لحركة الوجه والتفات الرأس لجميع الطلاب في الإطار"""
        pose_status = "الوضع الطبيعي للوجه"
        is_alert = False
        alerts_list = []

        if self.face_mesh is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_idx = 1
                    for face_landmarks in results.multi_face_landmarks:
                        # نقاط الأنف، العين اليسرى، العين اليمنى، وحافة الوجه
                        nose = face_landmarks.landmark[1]
                        left_eye = face_landmarks.landmark[33]
                        right_eye = face_landmarks.landmark[263]
                        
                        # حساب المسافات لتقدير دوران الرأس أفقي وعمودي
                        eye_dist = abs(right_eye.x - left_eye.x)
                        if eye_dist > 0:
                            offset = (nose.x - (left_eye.x + right_eye.x) / 2) / eye_dist
                            
                            # إذا تجاوز الانحراف النسب الطبيعية (التفات جانبي أو التفتت للخلف)
                            if offset < -0.25:
                                alerts_list.append(f"⚠️ طالب #{face_idx}: ملتفت بيسار الرأس / أو التفات جانبي للخلف!")
                                is_alert = True
                            elif offset > 0.25:
                                alerts_list.append(f"⚠️ طالب #{face_idx}: ملتفت بيمين الرأس / أو التفات جانبي للخلف!")
                                is_alert = True
                        
                        face_idx += 1
                        
                    if is_alert:
                        pose_status = " | ".join(alerts_list)
                else:
                    pose_status = "⚠️ تحذير: الوجه غير واضح أو ملتفت بالكامل بعيداً عن الكاميرا!"
                    is_alert = True
            except Exception:
                pass
        return pose_status, is_alert

    def process_frame_with_yolo(self, frame):
        """كشف الأجسام المحظورة والأشخاص وتحديد التفات الرأس"""
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
                        
                        if conf > 0.50:
                            coords = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = coords

                            if class_name == 'person':
                                area = (x2 - x1) * (y2 - y1)
                                if area > 3000:
                                    person_count += 1
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            elif class_name in ['cell phone', 'phone']:
                                phone_count += 1
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, f"⚠️ PHONE DETECTED!", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                alerts.append("🚨 [تنبيه خطر]: تم اكتشاف هاتف محمول محظور!")

                            elif class_name in ['book', 'laptop']:
                                book_count += 1
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                                cv2.putText(frame, f"⚠️ BOOK/MATERIAL", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                alerts.append("⚠️ [تنبيه]: تم اكتشاف كتاب أو مادة خارجية!")
            except Exception:
                pass

        # فحص حركة الرأس لجميع الطلاب
        head_msg, head_alert = self.estimate_head_pose(frame)
        if head_alert:
            alerts.append(f"🚨 {head_msg}")

        if person_count > 1:
            alerts.append(f"🚨 [تنبيه خطر]: يوجد {person_count} أشخاص في نطاق الاختبار!")

        return frame, alerts, person_count, phone_count, book_count, head_msg

detector = CheatingDetectorSystem()

def play_alarm_sound():
    alarm_html = """
        <audio autoplay>
          <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
        </audio>
    """
    st.markdown(alarm_html, unsafe_allow_html=True)

# --- واجهة Streamlit ---
if source_option == "ملف فيديو من بياناتي (Data Set)":
    st.markdown("### 📁 تحليل ملف فيديو من مجموعة البيانات الخاصة بك (Data Set)")
    uploaded_file = st.file_uploader("قم برفع ملف فيديو (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        if st.button("بدء تحليل الفيديو وعرض النتائج المفصلة"):
            st.write("جاري معالجة الفيديو واستخراج تقارير الغش...")
            stframe = st.empty()
            
            metrics_placeholder = st.empty()
            head_status_placeholder = st.empty()
            alert_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, alerts, p_count, phone_count, book_count, head_msg = detector.process_frame_with_yolo(frame)
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("عدد الأشخاص", p_count)
                    col2.metric("الهواتف المكتشفة", phone_count)
                    col3.metric("الكتب والمواد", book_count)
                
                with head_status_placeholder.container():
                    st.info(f"👤 حالة حركة الوجه والالتفات: **{head_msg}**")

                with alert_placeholder.container():
                    if alerts:
                        play_alarm_sound()
                        for alert in alerts:
                            st.error(alert)
                    else:
                        st.success("الوضع سليم في هذا الإطار.")
            cap.release()

elif source_option == "رابط فيديو من الإنترنت (URL)":
    st.markdown("### 🌐 جلب وتحليل فيديو من الإنترنت عبر الرابط")
    video_url = st.text_input("أدخل رابط الفيديو المباشر (رابط ملف فيديو مباشر MP4):")

    if video_url:
        if st.button("بدء تحليل فيديو الإنترنت وعرض النتائج المفصلة"):
            cap = cv2.VideoCapture(video_url)
            stframe = st.empty()
            metrics_placeholder = st.empty()
            head_status_placeholder = st.empty()
            alert_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("انتهى الفيديو أو تعذر الوصول للرابط.")
                    break
                
                processed_frame, alerts, p_count, phone_count, book_count, head_msg = detector.process_frame_with_yolo(frame)
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("عدد الأشخاص", p_count)
                    col2.metric("الهواتف المكتشفة", phone_count)
                    col3.metric("الكتب والمواد", book_count)
                
                with head_status_placeholder.container():
                    st.info(f"👤 حالة حركة الوجه والالتفات: **{head_msg}**")

                with alert_placeholder.container():
                    if alerts:
                        play_alarm_sound()
                        for alert in alerts:
                            st.error(alert)
                    else:
                        st.success("الوضع سليم.")
            cap.release()

elif source_option == "كاميرا الجهاز (Camera)":
    st.markdown("### 📷 البث المباشر من كاميرا الجهاز")
    img_file_buffer = st.camera_input("التقط صورة للتحقق الفوري")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        processed_frame, alerts, p_count, phone_count, book_count, head_msg = detector.process_frame_with_yolo(cv2_img)

        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("عدد الأشخاص", p_count)
        col2.metric("الهواتف المكتشفة", phone_count)
        col3.metric("الكتب والمواد", book_count)

        st.info(f"👤 حالة حركة الوجه والالتفات: **{head_msg}**")

        if alerts:
            play_alarm_sound()
            for alert in alerts:
                st.error(alert)
        else:
            st.success("الوضع سليم: لم يتم رصد أي محاولات غش.")
