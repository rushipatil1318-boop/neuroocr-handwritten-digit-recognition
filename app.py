from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from src.inference import load_model, predict_image


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "neuroocr_cnn.npz"
SAMPLE_DIR = ROOT / "samples"


st.set_page_config(page_title="NeuroOCR", page_icon="🔢", layout="wide")
st.title("NeuroOCR: Handwritten Digit Recognition")
st.caption("A confidence-aware CNN for reading one handwritten digit from an image.")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


model = get_model()
threshold = st.sidebar.slider("Confidence threshold", 0.50, 0.99, 0.85, 0.01)
source = st.radio("Choose an input", ["Upload an image", "Use a sample"], horizontal=True)

image = None
if source == "Upload an image":
    uploaded = st.file_uploader("Upload one digit on a plain background", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded)
else:
    chosen = st.selectbox("Sample digit", list(range(10)))
    image = Image.open(SAMPLE_DIR / f"digit_{chosen}.png")

if image is not None:
    result = predict_image(model, image, confidence_threshold=threshold)
    left, middle, right = st.columns([1, 1, 1.5])
    with left:
        st.subheader("Input")
        st.image(image, width=250)
    with middle:
        st.subheader("Model input")
        st.image(result["processed_image"], width=250, clamp=True)
    with right:
        st.subheader("Decision")
        if result["accepted"]:
            st.success(f"Predicted digit: {result['prediction']}")
        else:
            st.warning("Uncertain result — manual review recommended")
        st.metric("Confidence", f"{result['confidence']:.2%}")

    probability_table = pd.DataFrame(
        {
            "digit": list(range(10)),
            "probability": result["probabilities"],
        }
    ).set_index("digit")
    st.bar_chart(probability_table)

    saliency = model.saliency(result["processed_image"], int(result["prediction"]))
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(result["processed_image"], cmap="gray_r", vmin=0, vmax=1)
    axis.imshow(saliency, cmap="magma", alpha=0.55, vmin=0, vmax=1)
    axis.axis("off")
    axis.set_title("Pixels that influenced the CNN")
    st.pyplot(figure)

st.info(
    "Prototype limitation: the model was trained on small centered 8×8 handwritten digits. "
    "It is not intended for identity, financial, or other high-stakes decisions."
)
