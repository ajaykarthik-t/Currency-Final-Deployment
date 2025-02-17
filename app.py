import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from gtts import gTTS  # Google Text-to-Speech
import os

# Load the trained model
MODEL_PATH = "vgg16_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_labels = {
    '100_new': "You have 100 rupees note",
    '200_new': "You have 200 rupees note",
    '20_new': "You have 20 rupees note",
    '500_new': "You have 500 rupees note",
    '50_new': "You have 50 rupees note"
}

def preprocess_image(img):
    img = img.resize((224, 224))  # Resize
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_class(img):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    class_id = np.argmax(predictions)
    predicted_label = list(class_labels.keys())[class_id]
    return class_labels[predicted_label]

def generate_audio(text):
    """Generate an audio file from text using gTTS"""
    tts = gTTS(text=text, lang="en")
    audio_file = "prediction.mp3"
    tts.save(audio_file)
    return audio_file

# Streamlit UI
st.title("Currency Note Classification for the Visually Impaired")
st.write("Upload an image of a currency note, and the app will predict its class and provide an audio response.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
    
    predicted_class = predict_class(image_data)
    st.write(f"**Predicted Class:** {predicted_class}")
    
    # Generate and provide an audio file for download
    audio_file = generate_audio(predicted_class)
    with open(audio_file, "rb") as file:
        st.download_button(label="🔊 Download Prediction Audio", data=file, file_name="prediction.mp3", mime="audio/mp3")
