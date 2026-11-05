import cv2
import numpy as np
import streamlit as st

class CheatingDetector:
    def __init__(self):
        # تحميل كاشف الوجوه الإفتراضي من OpenCV (Haar Cascade)
        self.face_cascade = cv2.CascadeView = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def estimate_head_pose(self, face_crop):
        """طريقة آمنة ومستقرة لتقدير التفات الرأس"""
        # يمكن إضافة منطق تقدير الاتجاه هنا
        return "Normal"

    def process_frame(self, frame):
        if frame is None:
            return frame, False, "لا إطار", 0, 0

        # تحويل الصورة إلى الأبيض والأسود لكشف الوجوه
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # كشف الوجوه بشكل آمن
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        alert = False
        msg = "طبيعي"
        p_count = 0  # عدد الأشخاص
        h_count = 0  # عدد الهواتف / الأجهزة (يمكن توسعتها)

        # التحقق من أن faces ليست الخالية/غير المعرفة قبل التكرار
        if faces is not None and len(faces) > 0:
            p_count = len(faces)
            
            # إذا كان هناك أكثر من شخص، نطلق تنبيهاً
            if p_count > 1:
                alert = True
                msg = "تحذير: يوجد أكثر من شخص في الشاشة!"
                
            for (x, y, fw, fh) in faces:
                # رسم مربع حول الوجه
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                
                # اقتطاع الوجه واختبار التفات الرأس
                face_crop = frame[y:y+fh, x:x+fw]
                pose = self.estimate_head_pose(face_crop)
        else:
            alert = True
            msg = "تحذير: لا يوجد شخص أمام الكاميرا!"

        return frame, alert, msg, p_count, h_count

# --- Streamlit UI ---
st.title("نظام كشف الغش للطلاب (Student Cheating Detection)")

detector = CheatingDetector()

# استخدام كاميرا Streamlit
img_file_buffer = st.camera_input("التقط صورة للتحقق")

if img_file_buffer is not None:
    # تحويل الصورة الملتقطة إلى مصفوفة OpenCV
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # معالجة الإطار
    processed_frame, alert, msg, p_count, h_count = detector.process_frame(cv2_img)

    # عرض الصورة المعالجة (تحويل BGR إلى RGB للعرّض في Streamlit)
    st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    # عرض النتائج
    col1, col2 = st.columns(2)
    col1.metric("عدد الأشخاص", p_count)
    col2.metric("عدد الهواتف/التنبيهات", h_count)

    if alert:
        st.error(msg)
    else:
        st.success("الوضع طبيعي: " + msg)
