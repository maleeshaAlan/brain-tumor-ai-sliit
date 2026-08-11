import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE = Path(__file__).resolve().parent
MODEL = BASE / "model" / "final_brain_tumor_model.keras"
META = BASE / "model" / "model_metadata.json"

DEFAULT_CLASSES = ["glioma_tumor","meningioma_tumor","no_tumor","pituitary_tumor"]
DISPLAY = {
    "glioma_tumor":"Glioma Tumor",
    "meningioma_tumor":"Meningioma Tumor",
    "no_tumor":"No Tumor",
    "pituitary_tumor":"Pituitary Tumor"
}

st.set_page_config(page_title="Brain Tumor MRI AI", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor MRI Classification")
st.caption("SLIIT Year 2 Semester 1 — Educational AI Prototype")

if not MODEL.exists():
    st.error("Missing model/final_brain_tumor_model.keras")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL)

if META.exists():
    with open(META, "r", encoding="utf-8") as f:
        meta = json.load(f)
else:
    meta = {"class_names":DEFAULT_CLASSES,"image_size":[224,224],"model_name":"Brain Tumor Classifier"}

model = load_model()
classes = meta.get("class_names", DEFAULT_CLASSES)
image_size = meta.get("image_size", [224,224])

st.success(f"Model loaded: {meta.get('model_name','Brain Tumor Classifier')}")

uploaded = st.file_uploader("Upload an MRI image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)

    with left:
        st.image(image, caption="Uploaded MRI", use_container_width=True)

    resized = image.resize((int(image_size[1]), int(image_size[0])))
    arr = np.asarray(resized, dtype=np.float32)
    probs = model.predict(np.expand_dims(arr,0), verbose=0)[0]
    idx = int(np.argmax(probs))

    with right:
        st.subheader("AI Result")
        st.metric("Prediction", DISPLAY.get(classes[idx], classes[idx]))
        st.metric("Model Probability", f"{float(probs[idx])*100:.2f}%")
        chart = {DISPLAY.get(name,name): float(probs[i]) for i,name in enumerate(classes)}
        st.bar_chart(chart)
        for name,p in chart.items():
            st.write(f"**{name}:** {p*100:.2f}%")

    st.warning("Educational prototype only — not a medical diagnosis.")
