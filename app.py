import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------------------------------
# 🧠 App Title and Description
# -------------------------------------------------------
st.set_page_config(page_title="Cancer Detection", page_icon="🩸", layout="centered")
st.title("🩸 Lung & Colon Cancer Detection using CNN")
st.write("""
Upload a **histopathological image** of lung or colon tissue.  
Our AI model will classify it as **Normal** or **Cancerous**.
""")

# -------------------------------------------------------
# 📦 Load Trained Model
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lung_cancer_cnn.h5")
    return model

model = load_model()

# -------------------------------------------------------
# 🏷️ Class Labels and Friendly Mapping
# -------------------------------------------------------
classes = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']

label_map = {
    "COLON_ACA": "Colon Adenocarcinoma (Cancerous)",
    "COLON_N": "Normal Colon Tissue",
    "LUNG_ACA": "Lung Adenocarcinoma (Cancerous)",
    "LUNG_SCC": "Lung Squamous Cell Carcinoma (Cancerous)",
    "LUNG_N": "Normal Lung Tissue"
}

# -------------------------------------------------------
# 📤 File Upload Section
# -------------------------------------------------------
uploaded_file = st.file_uploader("📸 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='🧬 Uploaded Image', use_column_width=True)

    # -------------------------------------------------------
    # 🧩 Preprocessing
    # -------------------------------------------------------
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------------------------------------
    # 🔍 Prediction
    # -------------------------------------------------------
    prediction = model.predict(img_array)
    pred_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    readable_label = label_map.get(pred_class.upper(), pred_class)

    # -------------------------------------------------------
    # 📊 Display Results
    # -------------------------------------------------------
    st.markdown("---")
    st.subheader("🔬 Prediction Result")
    st.markdown(f"**🩸 {readable_label}**")
    st.progress(confidence / 100)
    st.markdown(f"**Model Confidence:** {confidence:.2f}%")
    st.markdown("---")

    # -------------------------------------------------------
    # ℹ️ About Section
    # -------------------------------------------------------
    with st.expander("ℹ️ About this Model"):
        st.write("""
        - **Model Type:** Convolutional Neural Network (CNN)
        - **Input Size:** 128x128 RGB
        - **Classes:** Colon Adenocarcinoma, Colon Normal, Lung Adenocarcinoma, Lung SCC, Lung Normal
        - **Framework:** TensorFlow / Keras  
        - **Purpose:** Assist pathologists in identifying cancerous tissues
        """)

else:
    st.info("👆 Upload a tissue image to begin prediction.")
