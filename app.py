from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

model = load_model("RCC.keras", compile=False)

classes = [
    "Grade-0",
    "Grade-1",
    "Grade-2",
    "Grade-3",
    "Grade-4"
]

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "RCC API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(
        io.BytesIO(await file.read())
    ).convert("RGB")

    img = preprocess(image)

    pred = model.predict(img)

    class_index = np.argmax(pred)

    return {
        "prediction": classes[class_index],
        "confidence": float(np.max(pred))
    }
