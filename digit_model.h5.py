
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("digit_model.h5")

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (28x28 grayscale), and the app will predict the digit.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize and invert like MNIST
    image = image.resize((28, 28))
    image = ImageOps.invert(image)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {predicted_digit}")
