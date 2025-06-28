import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

# Page configuration
st.set_page_config(page_title=" AI  Cats vs Dogs", page_icon="🐶", layout="centered")

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=100)
st.sidebar.title("🐾 DOG vs CAT Classifier")
st.sidebar.markdown("Upload an image of a pet or animal to see if it's a **Cat**, **Dog**, or something else.")

# Title
st.title("🐕‍🦺 Cats vs Dogs Classifier")
st.caption("🔍 Animal Classifier ")

# Load pretrained model
model = MobileNetV2(weights="imagenet")

uploaded_file = st.file_uploader("📂 Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        preview_image = image.copy()
        st.image(preview_image, caption="📷 Uploaded Image", use_container_width=True, channels="RGB")

        # Resize and preprocess for model
        resized = image.resize((224, 224), Image.LANCZOS)
        img_array = np.array(resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=3)[0]

        top_label = decoded[0][1].lower()
        confidence = decoded[0][2]

        if any(word in top_label for word in ["cat", "kitten", "siamese", "tabby"]):
            label = "Cat 🐱"
        elif any(word in top_label for word in ["dog", "puppy", "retriever", "shepherd", "terrier"]):
            label = "Dog 🐶"
        else:
            label = "Other Animal or Object 🐾"

        st.success(f"🎯 Prediction: **{label}**")
        st.info(f"📌 Top Match: **{decoded[0][1].replace('_', ' ').title()}** ({confidence * 100:.2f}%)")

        with st.expander("📊 See Top 3 Predictions"):
            for i, (id, name, prob) in enumerate(decoded):
                st.write(f"{i+1}. **{name.replace('_', ' ').title()}** — {prob * 100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown(
    'Made with ❤️ by <a href="https://www.linkedin.com/in/mohammed-hadi-abdulla-4033782b5" target="_blank">Hadi on LinkedIn</a>',
    unsafe_allow_html=True
)



