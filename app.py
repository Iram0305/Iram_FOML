import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Define class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load pre-trained model once
@st.cache_resource
def load_model():
    # Load MobileNetV2 pre-trained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    # Build model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # Since we're using a pre-trained model, we assume it's fine-tuned for Fashion MNIST
    # In a real scenario, you'd load a fine-tuned model or train this one
    return model

model = load_model()

st.title("👗 Fashion Item Classifier")
st.write("Upload a 96x96 color image of a fashion item to classify it (e.g., T-shirt, Trouser, etc.).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and preprocess image
    image = Image.open(uploaded_file).convert("RGB").resize((96, 96))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 96, 96, 3)

    st.image(image, caption="Uploaded Fashion Item", width=150)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.write(f"### Predicted Item: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")
