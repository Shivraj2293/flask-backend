import os
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import google.generativeai as genai

# Load and Download Model

# Load and Download Model
# Load and Download Model
def load_model_from_drive(output_file):
    model_url = "https://drive.google.com/uc?id=1qA1xQlbTHUiWm3JVY2vLTt4HKSxi8Kez"
    if not os.path.exists(output_file):
        gdown.download(model_url, output_file, quiet=False, fuzzy=True)
    return tf.keras.models.load_model(output_file)



# Class labels for prediction
class_labels = [
    "burger", "butter_naan", "chai", "chapati", "chole_bhature",
    "dal_makhani", "dhokla", "fried_rice", "idli", "jalebi",
    "kaathi_rolls", "kadhai_paneer", "kulfi", "masala_dosa", "momos",
    "pani_puri", "pakode", "pav_bhaji", "pizza", "samosa"
]

# Predict food from image
def predict_food_from_image(model, image):
    try:
        IMG_SIZE = (150, 150)
        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        
        if predictions is None or len(predictions) == 0:
            return "unknown", 0.0

        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        
        predicted_food = class_labels[predicted_index] if predicted_index < len(class_labels) else "unknown"
        return predicted_food, confidence
    except Exception as e:
        print("Error in prediction:", e)
        return "unknown", 0.0

# Prepare image data for Gemini API
def input_image_setup(uploaded_file):
    try:
        mime_type = uploaded_file.mimetype
        if mime_type not in ["image/jpeg", "image/png"]:
            mime_type = "image/jpeg"
        return [{"mime_type": mime_type, "data": uploaded_file.read()}]
    except Exception as e:
        print("Error setting up image:", e)
        return []

# Get response from Gemini API
def get_gemini_response(food_item, image_data, prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = gemini_model.generate_content([food_item, image_data[0], prompt])
        return response.text
    except Exception as e:
        print("Error from Gemini API:", e)
        return f"Error retrieving nutritional details: {str(e)}"
