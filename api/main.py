import numpy as np
from fastapi import FastAPI, UploadFile
from keras import models
import uvicorn
from PIL import Image
from numpy import asarray
from io import BytesIO
app = FastAPI()

loaded_model = models.load_model("../models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def read_img_file(data) -> np.ndarray:
    img = Image.open(BytesIO(data))
    numpy_array = asarray(img)
    return numpy_array


@app.get("/ping")
async def ping():
    return "Hi"


@app.post("/predict")
async def predict(
    file: UploadFile
):
    # Read the image file and convert to numpy array
    numpy_array = read_img_file(await file.read())

    # Predict and return the result
    numpy_array = np.expand_dims(numpy_array, 0)
    scores = loaded_model.predict(numpy_array)
    predicted_class = CLASS_NAMES[np.argmax(scores[0])]
    confidence = np.max(scores[0])
    return {
        "predicted_batch": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
