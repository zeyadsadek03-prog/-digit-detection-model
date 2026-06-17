import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

model = tf.keras.models.load_model('handwritten.keras')

def predict(image):
    if image is None:
        return {}
    if isinstance(image, dict):
        image = image['composite']
    img = Image.fromarray(image).convert('L')
    img = ImageOps.invert(img)

    # Crop to bounding box, add padding
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        img = ImageOps.expand(img, border=40, fill=0)

    img = img.resize((28, 28))
    arr = np.array(img)
    arr = tf.keras.utils.normalize(arr, axis=1)  # match Colab preprocessing
    arr = np.array([arr])  # shape (1, 28, 28) — no channel dim
    pred = model.predict(arr)
    return {str(i): float(pred[0][i]) for i in range(10)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(label="Draw a digit", type="numpy", image_mode="RGB"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Handwritten Digit Recognizer",
    live=True
)

demo.launch()