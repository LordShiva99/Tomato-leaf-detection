import io
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path

# Initialize the FastAPI app
app = FastAPI(
    title="Plant Disease Classification API",
    description="REST API endpoint for diagnosing tomato crop health using MobileNetV2.",
    version="1.0"
)

# Project paths
PROJECT_DIR = Path(r"C:\python_work\computer_vision\PlantDiseaseProject")
MODEL_PATH = PROJECT_DIR / "plant_disease_model.keras"
CLASS_FILE_PATH = PROJECT_DIR / "class_names.json"

# Load the trained model
print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Load the exact class names dynamically from disk to prevent label mixing
if CLASS_FILE_PATH.exists():
    with open(CLASS_FILE_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
    print(f"Loaded {len(CLASS_NAMES)} class names successfully.")
else:
    # Fallback default list if json file is missing
    CLASS_NAMES = [
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Healthy",
        "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus"
    ]

@app.get("/")
def home_route():
    return {"message": "Plant Disease Classification API is running!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((160, 160))

        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)

        # Run inference
        predictions = model.predict(image_array)
        score = predictions[0]

        # Get the correct winning index and map it using our synchronized list
        predicted_index = int(np.argmax(score))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence_percentage = float(100 * np.max(score))

        return JSONResponse(content={
            "filename": file.filename,
            "predicted_disease": predicted_class,
            "confidence_score_percent": round(confidence_percentage, 2)
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
