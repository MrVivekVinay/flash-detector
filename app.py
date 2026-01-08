import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Load the model (Cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('flash_detection_model.keras')
    return model

model = load_model()

# 2. Define the page layout
st.title("📸 Flash vs. No-Flash Detector")
st.write("Upload an image to check if it was taken with a camera flash.")

# 3. File Uploader
file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file is not None:
    # Display the user's image
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 4. Preprocess the image to match model requirements
    # MobileNetV2 expects 224x224 and pixel values in [0, 1] usually handled by rescale
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    # Normalize (1./255) because your training used ImageDataGenerator(rescale=1./255)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 5. Make Prediction
    prediction = model.predict(img_array)
    score = prediction[0][0]

    # 6. Show Result
    st.write("---")
    if score > 0.5:
        st.error(f"**Result: NO FLASH DETECTED** ({score:.2%} confidence)")
    else:

        st.success(f"**Result: FLASH DETECTED** ({(1-score):.2%} confidence)")
