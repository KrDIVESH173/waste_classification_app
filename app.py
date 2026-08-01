import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("waste_model_best.h5")

# Streamlit UI
st.title("♻️ Waste Classification App")
st.write("Upload an image to classify it as Organic or Recyclable")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    label = "Recyclable" if prediction > 0.5 else "Organic"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Show results
    st.image(img, caption=f"Prediction: {label}", use_column_width=True)
    st.write(f"Confidence: {confidence:.2f}")
