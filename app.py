from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG19
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import gdown
import os

app = FastAPI()

# ---------------- DOWNLOAD WEIGHTS ----------------

FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"

if not os.path.exists("RCC.weights.h5"):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, "RCC.weights.h5", quiet=False)

# ---------------- BUILD MODEL ----------------

def build_vgg19(input_shape=(224,224,3), num_classes=5):

    base_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model

model = build_vgg19()

# ---------------- LOAD WEIGHTS ----------------

model.load_weights("RCC.weights.h5")

# ---------------- CLASS NAMES ----------------

classes = [
    "Grade-0",
    "Grade-1",
    "Grade-2",
    "Grade-3",
    "Grade-4"
]

# ---------------- PREPROCESS ----------------

def preprocess(image):

    image = image.resize((224,224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    return image

# ---------------- ROUTES ----------------

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
