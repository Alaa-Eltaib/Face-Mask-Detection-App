# file: app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("face_mask_model.h5")

st.title("Face Mask Classifier 😷")
st.write("Upload an image and the model will tell you if a mask is present.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((128, 128))
    st.image(img, use_container_width=True)

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "✅ With Mask" if pred < 0.5 else "❌ Without Mask"
    confidence = (1 - pred) if pred < 0.5 else pred

    st.markdown(f"### Prediction: **{label}**")
  
