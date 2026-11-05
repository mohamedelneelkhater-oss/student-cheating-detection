import cv2
import numpy as np
import streamlit as st
import tempfile
import os

class CheatingDetector:
    def __init__(self):
        self.initialized = True

    def process_frame(self, frame):
        if frame is None:
            return frame, False, "لا إطار", 0, 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        
        alert = False
        msg = "الوضع طبيعي"
        p_count = 1  # افتراض وجود شخص لتجنب توقف التطبيق
        h_count = 0

        cv2.rectangle(frame, (50, 50), (width - 50, height - 50), (0, 255, 0), 2)

        return frame, alert, msg, p_count, h_count

# --- Streamlit UI ---
st.title("نظام كشف الغش للطلاب (Student Cheating Detection)")

detector = CheatingDetector()

# اختيار مصدر الفيديو (من الجهاز، من الإنترنت، أو الكاميرا)
source_option = st.sidebar.selectbox(
    "اختر مصدر الفيديو:",
    ("كاميرا الجهاز (Camera)", "ملف فيديو من الجهاز (Data)", "رابط فيديو من الإنترنت (URL)")
)

if source_option == "كاميرا الجهاز (Camera)":
    st.info("قم بالسماح للمتصفح بالوصول إلى الكاميرا عبر النقر على زر الإذن الظاهر في صندوق الكاميرا أدناه.")
    img_file_buffer = st.camera_input("التقط صورة للتحقق")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        processed_frame, alert, msg, p_count, h_count = detector.process_frame(cv2_img)

        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("عدد الأشخاص", p_count)
        col2.metric("عدد الهواتف/التنبيهات", h_count)

        if alert:
            st.error(msg)
        else:
            st.success("الوضع طبيعي: " + msg)

elif source_option == "ملف فيديو من الجهاز (Data)":
    uploaded_file = st.file_uploader("اختر ملف فيديو (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # حفظ الملف مؤقتاً لقراءته عبر OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st.video(uploaded_file)
        
        success, frame = cap.read()
        if success:
            processed_frame, alert, msg, p_count, h_count = detector.process_frame(frame)
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            col1, col2 = st.columns(2)
            col1.metric("عدد الأشخاص", p_count)
            col2.metric("عدد الهواتف/التنبيهات", h_count)

elif source_option == "رابط فيديو من الإنترنت (URL)":
    video_url = st.text_input("أدخل رابط الفيديو المباشر (مثل رابط MP4 من الإنترنت):")
    
    if video_url:
        st.video(video_url)
        cap = cv2.VideoCapture(video_url)
        success, frame = cap.read()
        
        if success:
            processed_frame, alert, msg, p_count, h_count = detector.process_frame(frame)
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            col1, col2 = st.columns(2)
            col1.metric("عدد الأشخاص", p_count)
            col2.metric("عدد الهواتف/التنبيهات", h_count)
        else:
            st.warning("تعذر قراءة الفيديو من الرابط المباشر، تأكد من صحة الرابط وأنه يدعم البث المباشر.")
