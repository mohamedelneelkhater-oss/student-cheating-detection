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

st.set_page_config(page_title="نظام كشف الغش الذكي الاحترافي", layout="wide")

st.title("🎓 نظام كشف الغش الذكي الاحترافي (YOLOv11 + تحليل الالتفات والغش)")

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
                        max_num_faces=5, 
                        min_detection_confidence=0.3, 
                        min_tracking_confidence=0.3
                    )
            except Exception:
                self.face_mesh = None

    def estimate_head_pose(self, frame):
        """تحليل دقيق لحركة الرأس والالتفات للكشف عن محاولات الغش أو النظر للزملاء"""
        pose_status = "وضع طبيعي ومنتظم"
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
                        
                        eye_dist = abs(right_eye.x - left_eye.x)
                        if eye_dist > 0:
                            offset = (nose.x - (left_eye.x + right_eye.x) / 2) / eye_dist
                            
                            if offset < -0.15:
                                pose_status = "🚨 تنبيه غش: الطالب ملتفت نحو اليسار (مشاهدة ورقة زميل أو محيط خارجي)!"
                                is_alert = True
                                break
                            elif offset > 0.15:
                                pose_status = "🚨 تنبيه غش: الطالب ملتفت نحو اليمين (مشاهدة ورقة زميل أو محيط خارجي)!"
                                is_alert = True
                                break
                else:
                    pose_status = "⚠️ تنبيه: وجه الطالب غير مرئي أو ملتفت بالكامل للوراء!"
                    is_alert = True
            except Exception:
                pass
        return pose_status, is_alert

    def process_frame_with_yolo(self, frame):
        """كشف الأجسام المحظورة والأشخاص باستخدام YOLOv11"""
        alerts = []
        person_count = 0
        phone_detected = False
        book_detected = False

        if yolo_model is not None:
            try:
                results = yolo_model(frame, verbose=False)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = yolo_model.names[cls_id]
                        
                        if conf > 0.45:
                            coords = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = coords

                            # كشف الأشخاص
                            if class_name == 'person':
                                area = (x2 - x1) * (y2 - y1)
                                if area > 2500:
                                    person_count += 1
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # كشف الهواتف (غش بالهاتف)
                            elif class_name in ['cell phone', 'phone']:
                                phone_detected = True
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, f"⚠️ PHONE CHEATING!", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                alerts.append("🚨 [حالة غش مؤكدة]: تم رصد استخدام هاتف محمول محظور!")

                            # كشف الكتب أو الأوراق الخارجية (غش بالكتاب)
                            elif class_name in ['book', 'laptop']:
                                book_detected = True
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                                cv2.putText(frame, f"⚠️ BOOK/MATERIAL DETECTED", (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                alerts.append("📚 [حالة غش مؤكدة]: تم رصد كتاب أو أوراق خارجية غير مسموحة!")
            except Exception:
                pass

        # فحص الالتفات وحركة الوجه
        head_msg, head_alert = self.estimate_head_pose(frame)
        if head_alert:
            alerts.append(f"👀 {head_msg}")

        # تنبيه تعدد الأشخاص (غش جماعي أو وجود شخص مساعد)
        if person_count > 1:
            alerts.append(f"👥 [تنبيه غش جماعي]: يوجد {person_count} أشخاص داخل نطاق الكاميرا!")

        return frame, alerts, person_count, phone_detected, book_detected, head_msg

detector = CheatingDetectorSystem()

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
    uploaded_file = st.file_uploader("قم برفع ملف فيديو لاختبار الغش (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        if st.button("بدء تحليل الفيديو واستخراج تقارير الغش الشاملة"):
            st.write("جاري معالجة الفيديو إطاراً بإطار ورصد حالات الغش...")
            stframe = st.empty()
            
            metrics_placeholder = st.empty()
            head_status_placeholder = st.empty()
            alert_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, alerts, p_count, phone_det, book_det, head_msg = detector.process_frame_with_yolo(frame)
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("عدد الأشخاص", p_count)
                    col2.metric("حالة الهاتف", "مكتشف 🚨" if phone_det else "سليم ✅")
                    col3.metric("حالة الكتب/المراجع", "مكتشف 🚨" if book_det else "سليم ✅")
                
                with head_status_placeholder.container():
                    if "تنبيه" in head_msg or "مشاهدة" in head_msg:
                        st.error(f"👤 تقرير حالة الالتفات: **{head_msg}**")
                    else:
                        st.success(f"👤 تقرير حالة الالتفات: **{head_msg}**")

                with alert_placeholder.container():
                    if alerts:
                        play_alarm_sound()
                        for alert in alerts:
                            st.error(alert)
                    else:
                        st.success("✨ الوضع سليم تماماً: لم يتم رصد أي محاولات غش في هذا الإطار.")
            cap.release()

# --- 2. خيار رابط فيديو من الإنترنت (URL) ---
elif source_option == "رابط فيديو من الإنترنت (URL)":
    st.markdown("### 🌐 جلب وتحليل فيديو من الإنترنت عبر الرابط")
    video_url = st.text_input("أدخل رابط الفيديو المباشر (رابط ملف فيديو مباشر MP4):")

    if video_url:
        if st.button("بدء تحليل فيديو الإنترنت واستخراج التقارير"):
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
                
                processed_frame, alerts, p_count, phone_det, book_det, head_msg = detector.process_frame_with_yolo(frame)
                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("عدد الأشخاص", p_count)
                    col2.metric("حالة الهاتف", "مكتشف 🚨" if phone_det else "سليم ✅")
                    col3.metric("حالة الكتب/المراجع", "مكتشف 🚨" if book_det else "سليم ✅")
                
                with head_status_placeholder.container():
                    if "تنبيه" in head_msg or "مشاهدة" in head_msg:
                        st.error(f"👤 تقرير حالة الالتفات: **{head_msg}**")
                    else:
                        st.success(f"👤 تقرير حالة الالتفات: **{head_msg}**")

                with alert_placeholder.container():
                    if alerts:
                        play_alarm_sound()
                        for alert in alerts:
                            st.error(alert)
                    else:
                        st.success("✨ الوضع سليم تماماً.")
            cap.release()

# --- 3. خيار كاميرا الجهاز (Camera) ---
elif source_option == "كاميرا الجهاز (Camera)":
    st.markdown("### 📷 البث المباشر من كاميرا الجهاز")
    img_file_buffer = st.camera_input("التقط صورة للتحقق الفوري")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        processed_frame, alerts, p_count, phone_det, book_det, head_msg = detector.process_frame_with_yolo(cv2_img)

        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("عدد الأشخاص", p_count)
        col2.metric("حالة الهاتف", "مكتشف 🚨" if phone_det else "سليم ✅")
        col3.metric("حالة الكتب/المراجع", "مكتشف 🚨" if book_det else "سليم ✅")

        if "تنبيه" in head_msg or "مشاهدة" in head_msg:
            st.error(f"👤 تقرير حالة الالتفات: **{head_msg}**")
        else:
            st.success(f"👤 تقرير حالة الالتفات: **{head_msg}**")

        if alerts:
            play_alarm_sound()
            for alert in alerts:
                st.error(alert)
        else:
            st.success("✨ الوضع سليم تماماً: لم يتم رصد أي محاولات غش.")
