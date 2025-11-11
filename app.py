import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set title
st.title("🫁 Lung Cancer Detection using CNN")
st.write("Upload a lung or colon tissue histopathological image to classify it as Normal or Cancerous.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lung_cancer_cnn.h5")
    return model

model = load_model()

# Class labels (match training order)
classes = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)
    pred_class = classes[np.argmax(prediction)]
    
    st.markdown(f"### 🩸 Prediction: **{pred_class.upper()}**")
