import cv2
import numpy as np
import streamlit as st
import urllib.request
import os

class CheatingDetector:
    def __init__(self):
        # مسار تحميل ملف النموذج احتياطياً مباشرة من غيتهاب إذا لم يكن موجوداً
        cascade_path = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
            try:
                urllib.request.urlretrieve(url, cascade_path)
            except Exception:
                pass

        # محاولة التحميل بأكثر من طريقة لضمان عدم حدوث خطأ
        self.face_cascade = None
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade is None or self.face_cascade.empty():
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except Exception:
                pass

    def estimate_head_pose(self, face_crop):
        return "Normal"

    def process_frame(self, frame):
        if frame is None:
            return frame, False, "لا إطار", 0, 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
        else:
            faces = []

        alert = False
        msg = "طبيعي"
        p_count = 0
        h_count = 0

        if faces is not None and len(faces) > 0:
            p_count = len(faces)
            if p_count > 1:
                alert = True
                msg = "تحذير: يوجد أكثر من شخص في الشاشة!"
                
            for (x, y, fw, fh) in faces:
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        else:
            alert = True
            msg = "تحذير: لا يوجد شخص أمام الكاميرا!"

        return frame, alert, msg, p_count, h_count

# --- Streamlit UI ---
st.title("نظام كشف الغش للطلاب (Student Cheating Detection)")

detector = CheatingDetector()

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
