import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

model = tf.keras.models.load_model('model/handwritten.keras')

def predict(image):
    img = np.array(image.convert('L'))
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    img = tf.keras.utils.normalize(img, axis=1)
    img = np.array([img])
    prediction = model.predict(img)
    return f"Predicted Digit: {np.argmax(prediction)}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Handwritten Digit Recognizer"
)

demo.launch()
