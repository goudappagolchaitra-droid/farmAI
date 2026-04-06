from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io

app = FastAPI(title="FarmAI - Crop Disease Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model and class names
model = tf.keras.models.load_model('crop_disease_model.keras')
with open('class_names.json') as f:
    class_names = json.load(f)

@app.get("/")
def home():
    return {"message": "FarmAI Crop Disease Detector is running! 🌾"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100
    
    return {
        "disease": predicted_class,
        "confidence": f"{confidence:.1f}%",
        "is_healthy": "healthy" in predicted_class.lower()
    }
# ============================================
# MEMBER 2 - FEATURE 1: CROP PRICE PREDICTOR
# ============================================

crop_prices = {
    "rice": {"min": 1800, "max": 2200, "unit": "per quintal"},
    "wheat": {"min": 2000, "max": 2400, "unit": "per quintal"},
    "maize": {"min": 1500, "max": 1900, "unit": "per quintal"},
    "tomato": {"min": 800, "max": 2000, "unit": "per quintal"},
    "potato": {"min": 600, "max": 1200, "unit": "per quintal"},
    "onion": {"min": 700, "max": 1500, "unit": "per quintal"},
    "cotton": {"min": 5500, "max": 6500, "unit": "per quintal"},
    "sugarcane": {"min": 280, "max": 350, "unit": "per quintal"},
    "soybean": {"min": 3800, "max": 4400, "unit": "per quintal"},
    "groundnut": {"min": 4500, "max": 5500, "unit": "per quintal"},
}

@app.get("/predict-price")
def predict_price(crop: str, season: str = "kharif"):
    crop = crop.lower().strip()
    
    if crop not in crop_prices:
        return {
            "error": f"Crop '{crop}' not found",
            "available_crops": list(crop_prices.keys())
        }
    
    price_data = crop_prices[crop]
    avg_price = (price_data["min"] + price_data["max"]) // 2
    
    if season.lower() == "rabi":
        avg_price = int(avg_price * 1.1)
    elif season.lower() == "zaid":
        avg_price = int(avg_price * 0.95)
    
    return {
        "crop": crop.capitalize(),
        "season": season,
        "min_price": price_data["min"],
        "max_price": price_data["max"],
        "avg_price": avg_price,
        "unit": price_data["unit"],
        "recommendation": "Good time to sell! 📈" if avg_price > 2000 else "Wait for better price 📉"
    }
# ============================================
# MEMBER 2 - FEATURE 1: CROP PRICE PREDICTOR
# ============================================

crop_prices = {
    "rice": {"min": 1800, "max": 2200, "unit": "per quintal"},
    "wheat": {"min": 2000, "max": 2400, "unit": "per quintal"},
    "maize": {"min": 1500, "max": 1900, "unit": "per quintal"},
    "tomato": {"min": 800, "max": 2000, "unit": "per quintal"},
    "potato": {"min": 600, "max": 1200, "unit": "per quintal"},
    "onion": {"min": 700, "max": 1500, "unit": "per quintal"},
    "cotton": {"min": 5500, "max": 6500, "unit": "per quintal"},
    "sugarcane": {"min": 280, "max": 350, "unit": "per quintal"},
    "soybean": {"min": 3800, "max": 4400, "unit": "per quintal"},
    "groundnut": {"min": 4500, "max": 5500, "unit": "per quintal"},
}

@app.get("/predict-price")
def predict_price(crop: str, season: str = "kharif"):
    crop = crop.lower().strip()
    
    if crop not in crop_prices:
        return {
            "error": f"Crop '{crop}' not found",
            "available_crops": list(crop_prices.keys())
        }
    
    price_data = crop_prices[crop]
    avg_price = (price_data["min"] + price_data["max"]) // 2
    
    if season.lower() == "rabi":
        avg_price = int(avg_price * 1.1)
    elif season.lower() == "zaid":
        avg_price = int(avg_price * 0.95)
    
    return {
        "crop": crop.capitalize(),
        "season": season,
        "min_price": price_data["min"],
        "max_price": price_data["max"],
        "avg_price": avg_price,
        "unit": price_data["unit"],
        "recommendation": "Good time to sell! 📈" if avg_price > (price_data["min"] + price_data["max"]) // 2 else "Wait for better price 📉"
    }