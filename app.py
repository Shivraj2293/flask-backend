from flask import Flask, request, jsonify
from PIL import Image
from dotenv import load_dotenv
import os

# Import only the necessary utility functions (removed model-related functions)
from utils.model_utils import (
    input_image_setup,
    get_gemini_response
)

# Load environment variables
load_dotenv(dotenv_path='instance/.env')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise Exception("Google API key not found. Set it in your .env file.")

# Flask App Setup
app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "pong"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    try:
        # Open and convert the image to RGB
        img = Image.open(image.stream).convert('RGB')
        
        # Since the TensorFlow model is removed, we no longer perform food prediction.
        # Instead, we set the predicted food to a generic label.
        predicted_food = "this food item"
        
        # Reset the stream and prepare the image data for the Gemini API.
        image.seek(0)
        image_data = input_image_setup(image)
        
        # Use the same prompt as before to analyze nutritional content.
        prompt = (
            "Identify the food item and analyze its nutritional content. "
            "Provide an estimated breakdown of Calories, Protein (g), Carbohydrates (g), "
            "Fats (g), Fiber (g), and Sugars (g). Additionally, include dietary recommendations tailored to Indian cuisine."
        )
        
        nutritional_response = get_gemini_response(predicted_food, image_data, prompt, GOOGLE_API_KEY)
        return jsonify({
            "predicted_food": predicted_food,
            "nutritional_info": nutritional_response
        }), 200

    except Exception as e:
        print("Error processing upload:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)