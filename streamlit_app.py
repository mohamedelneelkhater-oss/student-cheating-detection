"""
Road Damage Detection System - Streamlit app
Model: YOLO26n (fine-tuned on a Roboflow road damage dataset)
"""

import io
from pathlib import Path

import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "weights" / "best.pt"
IMG_SIZE = 640


# --------------------------------------------------------------------------- #
# Model loading (cached so it loads only once)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading YOLO26n model...")
def load_model():
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{MODEL_PATH}'. "
            "Make sure 'weights/best.pt' is uploaded to the repository."
        )
    return YOLO(str(MODEL_PATH))


def run_detection(model, image: Image.Image, conf: float, iou: float):
    """Run inference and return the annotated image and per-class counts."""
    result = model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=conf,
        iou=iou,
        device="cpu",
        verbose=False,
    )[0]

    # result.plot() returns BGR, so reverse the channels to get RGB.
    annotated = Image.fromarray(result.plot()[..., ::-1])

    counts = {}
    for cls_id in result.boxes.cls.int().tolist():
        name = model.names[cls_id]
        counts[name] = counts.get(name, 0) + 1

    return annotated, counts


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🛣️ Road Damage Detection System")
st.caption("Powered by YOLO26n | Detects potholes, cracks, and other road defects")

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    conf = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05,
        help="Lower values detect more damage but may add false alarms.",
    )
    iou = st.slider(
        "NMS IoU threshold", 0.10, 0.95, 0.45, 0.05,
        help="Controls how overlapping boxes are merged.",
    )
    st.divider()
    input_mode = st.radio("Input source", ["Upload image", "Use camera"])
    st.divider()
    st.markdown(
        "**About**  \n"
        "Fine-tuned YOLO26n trained on a 7-class road damage dataset "
        "(Roboflow). Built with Ultralytics, PyTorch, and Streamlit."
    )

# --------------------------------------------------------------------------- #
# Load model (with a clear error message if something is wrong)
# --------------------------------------------------------------------------- #
try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
if input_mode == "Upload image":
    source = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])
else:
    source = st.camera_input("Take a photo of the road")

# --------------------------------------------------------------------------- #
# Inference and results
# --------------------------------------------------------------------------- #
if source is not None:
    image = Image.open(source).convert("RGB")

    with st.spinner("Detecting road damage..."):
        annotated, counts = run_detection(model, image, conf, iou)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detection result")
        st.image(annotated, use_container_width=True)

    st.divider()

    if not counts:
        st.info(
            "No road damage detected at the current confidence threshold. "
            "Try lowering the threshold from the sidebar."
        )
    else:
        st.subheader(f"Detected damage ({sum(counts.values())} total)")
        st.table({"Class": list(counts.keys()), "Count": list(counts.values())})

    # Download the annotated image
    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    st.download_button(
        label="⬇️ Download result",
        data=buffer.getvalue(),
        file_name="road_damage_result.png",
        mime="image/png",
    )
else:
    st.info("Upload an image or use your camera to start.")
