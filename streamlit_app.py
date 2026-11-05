from datetime import datetime
from pathlib import Path
import cv2
import streamlit as st
from src.detector import CheatingDetector

st.set_page_config(page_title="نظام كشف الغش الذكي", layout="wide")

st.title("🚨 نظام تتبع ومراقبة حالات الغش في الامتحانات (Exam Cheating Detection)")
st.markdown("---")

# إعداد النموذج
@st.cache_resource
def load_detector():
    return CheatingDetector(model_path="yolov8s.pt", conf_threshold=0.5)

detector = load_detector()

# الشريط الجانبي للتحكم وخيارات الفيديوهات
st.sidebar.header("إعدادات المصدر والتحكم")
source_option = st.sidebar.selectbox(
    "اختر مصدر الفيديو:",
    ("رفع فيديو من الجهاز", "الفيديوهات المتاحة في النظام (Dataset)", "رابط فيديو مباشر")
)

video_path = None
if source_option == "رفع فيديو من الجهاز":
    uploaded_file = st.sidebar.file_uploader("اختر ملف فيديو", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_file = Path("/tmp/uploaded_video.mp4")
        temp_file.write_bytes(uploaded_file.read())
        video_path = str(temp_file)

elif source_option == "الفيديوهات المتاحة في النظام (Dataset)":
    # البحث التلقائي عن الفيديوهات في مسارات كاجل أو مجلد samples
    sample_videos = list(Path("/kaggle/input").rglob("*.mp4")) + list(Path("samples").glob("*.mp4"))
    if sample_videos:
        selected_sample = st.sidebar.selectbox("اختر فيديو من الداتا:", [str(p) for p in sample_videos])
        video_path = selected_sample
    else:
        st.sidebar.warning("لم يتم العثور على فيديوهات تلقائية في /kaggle/input. يرجى رفع فيديو.")

else:
    url_input = st.sidebar.text_input("أدخل رابط الفيديو المباشر:")
    if url_input:
        video_path = url_input

# العرض الرئيسي والتشغيل
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

        # معالجة الإطار عبر نظام الكشف
        processed_frame, alert, msg, p_count, h_count = detector.process_frame(frame)
        
        # تحويل الألوان للعرض في Streamlit
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(processed_frame, channels="RGB", use_column_width=True)
        
        # تحديث العدادات على الواجهة
        frame_count_metric.metric("عداد إطارات الهاتف المتتالية", p_count)
        head_count_metric.metric("عداد إطارات التفات الرأس", h_count)

        if alert:
            alert_box.error(f"⚠️ {msg} (التوقيت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            alert_box.success("الحالة طبيعية - لم يتم رصد أي سلوك مريب.")

    cap.release()
else:
    st.info("الرجاء اختيار مصدر فيديو من الشريط الجانبي والضغط على 'بدء المعالجة والتحليل'.")
