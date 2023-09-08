from google.cloud import storage
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import os

model = None
BUCKET_NAME = "hyw-bucket"
SOURCE_BLOB_NAME = "models/potato_predict_model.h5"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
destination_file_name = "/tmp/potatoes.h5"


# Read model from Google Cloud bucket
def read_model():
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(SOURCE_BLOB_NAME)
    blob.download_to_filename(destination_file_name)


def predict(request):
    global model
    if model is None:
        read_model()
        model = tf.keras.models.load_model(destination_file_name)
        os.remove(destination_file_name)

    # Read the image from the request
    incoming = request.files["file"].read()
    image = np.array(Image.open(BytesIO(incoming)).convert("RGB").resize((256, 256)))
    image_array = np.expand_dims(image, 0)
    # Predict and return the result
    scores = model.predict(image_array)
    predicted_class = CLASS_NAMES[np.argmax(scores[0])]
    confidence = np.max(scores[0])

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence)
    }
