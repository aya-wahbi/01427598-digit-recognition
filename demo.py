import gradio as gr
import tensorflow as tf
import numpy as np

# Load your trained model (update the path as needed)
model = tf.keras.models.load_model("saved_models/final_refined_cnn.h5")

def preprocess_image(image):
    # If the input is a dict (e.g., from gr.ImageEditor), extract the composite image.
    if isinstance(image, dict):
        image = image.get("composite", image.get("image", None))
        if image is None:
            raise ValueError("Could not extract image from dictionary input!")
    # If the image has an alpha channel, discard it.
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    # Convert RGB to grayscale if necessary.
    if image.ndim == 3 and image.shape[-1] == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    # Resize the image to 28x28, add a channel dimension, normalize and add batch dimension.
    image = tf.image.resize(image[..., np.newaxis], [28, 28]).numpy()
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_digit(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return f"Predicted Digit: {predicted_class}\nConfidence: {confidence:.2f}"

# Create a default black canvas (28x28 with 3 channels)
default_canvas = np.zeros((28, 28, 3), dtype="uint8")

# Create the Gradio Interface with a bigger output textbox
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(
        canvas_size=(28, 28),
        type="numpy",
        value=default_canvas,
        brush=gr.Brush(default_size=1, default_color="black")
    ),
    outputs=gr.Textbox(lines=3, placeholder="predictions"),
    title="MNIST Digit Recognition",
    description="Draw a digit on the canvas to see the model's prediction.",
    
)

if __name__ == "__main__":
    demo.launch()