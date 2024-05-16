import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st



# Set working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/files/plantLeafDiseasesModel.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
class_indices = json.load(open(f"{working_dir}/files/class_indices.json"))


# Function to load and preprocess image using PIL
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to predict the class of an image using trained model, the image, and the class index list
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    # class_indices.json has the classes as strings, not integers
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# GUI: Streamlit app:

# Title
st.title('Plant Leaf Disease Predictor')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Once an image is uploaded, open the image, resize and preprocess the image, predict the class of the image
# Two columns are used, 1st displays image, 2nd has prediction
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    # Resize image to display for user
    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Predict'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')