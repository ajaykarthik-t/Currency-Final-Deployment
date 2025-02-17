import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import pyttsx3  # For text-to-speech

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

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("Currency Note Classification for the Visually Impaired")
st.write("Upload an image of a currency note, and the app will predict its class and announce it.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
    
    predicted_class = predict_class(image_data)
    st.write(f"**Predicted Class:** {predicted_class}")
    
    speak(predicted_class)
