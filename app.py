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