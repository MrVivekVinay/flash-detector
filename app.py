import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# 1. Page Config
st.set_page_config(page_title="Flash Detector", page_icon="📸")

# 2. Model Loading (with Error Handling)
@st.cache_resource
def load_model():
    model_path = 'flash_detection_model_v2_final.keras'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Checking directory...")
        st.write("Files found:", os.listdir("."))
        return None
    
    # compile=False is critical for cross-version compatibility
    return tf.keras.models.load_model(model_path, compile=False)

st.title("📸 Flash vs. No-Flash Detector")

model = load_model()

if model:
    file = st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocessing to match MobileNetV2 requirements
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner('Analyzing image...'):
            prediction = model.predict(img_array)
            score = float(prediction[0][0])

        st.divider()
        # Logic: 0 is usually 'Flash', 1 is 'No Flash' (check your training labels)
        if score < 0.5:
            st.success(f"**FLASH DETECTED** (Confidence: {100*(1-score):.1f}%)")
        else:
            st.error(f"**NO FLASH** (Confidence: {100*score:.1f}%)")

