from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import gdown

app = FastAPI()
#11XxB4c0rpvRJQ0w5yI6VBnr3f4MyOZLe
MODEL_PATH = "RCC.keras"

if not os.path.exists(MODEL_PATH):

    url = "https://drive.google.com/uc?id=1gT4CALHrUCPQbV1YlgKRYATqCbLsPqVN"

    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(
    MODEL_PATH,
    compile=False
)

classes = [
    "Grade-0",
    "Grade-1",
    "Grade-2",
    "Grade-3",
    "Grade-4"
]

def preprocess(image):
    image = image.resize((224,224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    img = preprocess(image)

    pred = model.predict(img)

    class_index = np.argmax(pred)

    return {
        "prediction": classes[class_index],
        "confidence": float(np.max(pred))
    }
