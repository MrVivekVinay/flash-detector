import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    # Functional models are much more stable across versions
    return tf.keras.models.load_model('flash_detection_model.keras', compile=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("📸 Flash vs. No-Flash Detector")

file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert('RGB')
    st.image(image, use_container_width=True)
    
    # Preprocessing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    score = float(prediction[0][0])

    if score > 0.5:
        st.error(f"No Flash Detected (Confidence: {score:.2%})")
    else:
        st.success(f"Flash Detected (Confidence: {1-score:.2%})")
