from flask import Flask, request, jsonify
from PIL import Image
from dotenv import load_dotenv
import os

# Import utility functions
from utils.model_utils import (
    load_model_from_drive,
    predict_food_from_image,
    input_image_setup,
    get_gemini_response
)

# Load environment variables
load_dotenv(dotenv_path='instance/.env')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise Exception("Google API key not found. Set it in your .env file.")

# Download and load the model
# Download and load the model
model = load_model_from_drive("food_classifier.h5")
model.summary()

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
        img = Image.open(image.stream).convert('RGB')
        print(f"📸 Received image: {image.filename}")
        print(f"🖼 Image format: {img.format}, Size: {img.size}, Mode: {img.mode}")
        # Predict food item
        predicted_food, confidence = predict_food_from_image(model, img)
        confidence_threshold = 0.7
        print(f"✅ Predicted Food: {predicted_food}")
        print(f"🔍 Confidence: {confidence:.2f}")
        # Prepare image data for Gemini
        image.seek(0)
        image_data = input_image_setup(image)
        
        
        # Generate prompt for Gemini API
        if predicted_food == "unknown" or confidence < confidence_threshold:
            predicted_food = "this food item"
            prompt = (
                "Identify the food item and analyze its nutritional content. "
                "Provide an estimated breakdown of Calories, Protein (g), Carbohydrates (g), "
                "Fats (g), Fiber (g), and Sugars (g). Additionally, include dietary recommendations tailored to Indian cuisine."
            )
        else:
            prompt = (
                f"Analyze the nutritional content of {predicted_food}. "
                "Provide an estimated breakdown of Calories, Protein (g), Carbohydrates (g), "
                "Fats (g), Fiber (g), and Sugars (g), along with dietary recommendations."
            )
        
        nutritional_response = get_gemini_response(predicted_food, image_data, prompt, GOOGLE_API_KEY)
        return jsonify({
            "predicted_food": predicted_food,
            "confidence": confidence,
            "nutritional_info": nutritional_response
        }), 200

    except Exception as e:
        print("Error processing upload:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)