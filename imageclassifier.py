import streamlit as st
import numpy as np
import tensorflow as tf
from io import BytesIO
import PIL.Image as Image

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="X-Ray Pneumonia Detection",
    page_icon="🩺",
    layout="centered",
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
/* Global styling */
body {
    background-color: #f8fbfc;
    font-family: 'Open Sans', sans-serif;
}

/* Title */
h1, h2, h3, h4 {
    color: #006d77;
    font-weight: 700;
}

/* Subheader text */
.block-container {
    padding-top: 2rem;
}

/* Upload widget */
.stFileUploader {
    border: 2px dashed #83c5be;
    padding: 1rem;
    border-radius: 10px;
    background-color: #e6fffa20;
}

/* Button styling */
div.stButton > button {
    background-color: #0096c7;
    color: white;
    font-size: 1.1rem;
    font-weight: bold;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    border: none;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    transition: all 0.3s ease-in-out;
}
div.stButton > button:hover {
    background-color: #0077b6;
    transform: translateY(-2px);
}

/* Prediction result */
.prediction-box {
    background-color: #e6fffa;
    border-left: 6px solid #06d6a0;
    padding: 1rem;
    border-radius: 8px;
    font-size: 1.2rem;
    font-weight: bold;
    color: #073b4c;
    margin-top: 1rem;
}

/* Uploaded image styling */
img {
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ========== APP LOGIC ==========
img_size = 100
CATEGORIES = ["NORMAL", "Pneumonia"]

PRETRAINED_MODEL_PATH = r"G:\X-Ray Pneumonia Detection\X_Ray_Pneumonia_Detection\X_Ray_Classification.keras"
model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
print("Model Loaded")

def load_classifier():
    st.title("🩺 X-Ray Pneumonia Detection")
    st.write("This tool uses deep learning to assist in identifying **Pneumonia** from chest X-ray images. "
             "It is designed for educational purposes and **not** as a replacement for professional medical advice.")

    file = st.file_uploader("Upload a Chest X-Ray Image", type=['jpeg', 'jpg', 'png'])
    
    if file is not None:
        img = Image.open(BytesIO(file.read())).convert("RGB")  # Force 3 channels
        img = img.resize((img_size, img_size))
        new_array = tf.keras.preprocessing.image.img_to_array(img)
        new_array = np.expand_dims(new_array, axis=0)  # (1, 100, 100, 3)
        
        st.image(img, caption="Uploaded X-ray", use_column_width=True)

        if st.button("🔍 Predict"):
            prediction = model.predict(new_array / 255.0)
            label_index = int(round(prediction[0][0]))
            confidence = round(prediction[0][0] * 100, 2)
            preds = f"{CATEGORIES[label_index]} - {confidence}%"
            
            st.markdown(f"<div class='prediction-box'>{preds}</div>", unsafe_allow_html=True)

def main():
    load_classifier()

if __name__ == "__main__":
    main()
