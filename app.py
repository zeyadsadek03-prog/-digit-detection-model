# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('model/handwritten.keras')

st.title("Handwritten Digit Recognizer")
uploaded = st.file_uploader("Upload a digit image", type=["png","jpg"])

if uploaded:
    img = Image.open(uploaded).convert('L')
    img = np.array(img)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    img = tf.keras.utils.normalize(img, axis=1)
    img = np.array([img])
    
    prediction = model.predict(img)
    st.write(f"## Predicted Digit: {np.argmax(prediction)}")