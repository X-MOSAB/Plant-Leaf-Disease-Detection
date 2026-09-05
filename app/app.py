from pathlib import Path
import sys

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# =========================
# Project Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT / "src"))

from dataset import class_names


# =========================
# Configuration
# =========================

MODEL_PATH = PROJECT_ROOT / "models" / "efficientnetb0_best.keras"

IMG_SIZE = (224, 224)


# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# =========================
# Prediction
# =========================

def predict_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    return [
        (class_names[index], float(predictions[index]))
        for index in top_indices
    ]


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Plant Disease Detection")
st.write(
    "Upload a plant leaf image and the model will predict "
    "the most likely disease."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing image..."):

            results = predict_image(image)

        st.subheader("Prediction")

        best_class, best_confidence = results[0]

        st.success(
            f"**{best_class}**"
        )

        st.metric(
            "Confidence",
            f"{best_confidence:.2%}"
        )

        st.subheader("Top 3 Predictions")

        for rank, (class_name, confidence) in enumerate(
            results,
            start=1
        ):
            st.write(
                f"**{rank}. {class_name}** — "
                f"{confidence:.2%}"
            )

            st.progress(
                confidence
            )