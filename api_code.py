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

# validation_set = tf.keras.utils.image_dataset_from_directory(
#     'dataset/valid',
#     labels="inferred",
#     label_mode="categorical",
#     class_names=None,
#     color_mode="rgb",
#     batch_size=32,
#     image_size=(128, 128),
#     shuffle=True
# )

# class_names = validation_set.class_names
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

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
