import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')

validation_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

class_names = validation_set.class_names

def predict_disease(img_array):
    """Predict disease from the given image array."""
    input_arr = np.array([img_array])  # Convert single image to batch
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions)
    return class_names[result_index]

# API route to handle image uploads and return predictions
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    """Receive an uploaded image, process it, and return the disease prediction."""
    # Read image file
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))

    # Convert image to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize the image to match the input size of the model
    img = img.resize((128, 128))

    # Convert image to numpy array
    img_array = keras_image.img_to_array(img)

    # Get the prediction
    model_prediction = predict_disease(img_array)

    # Return prediction as JSON response
    return JSONResponse(content={"prediction": model_prediction})

# Run the FastAPI app using Uvicorn (in terminal):
# uvicorn your_script_name:app --reload
