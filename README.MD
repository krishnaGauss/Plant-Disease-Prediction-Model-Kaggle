# Crop Disease Prediction API

This API allows users to upload an image of a crop leaf and receive a prediction of the disease (if any) affecting the crop. The API is powered by a pre-trained Convolutional Neural Network (CNN) model, using **TensorFlow** and **FastAPI**.

## Features

- Upload an image of a crop leaf.
- The model predicts the disease (or healthy status) of the plant based on the uploaded image.
- Supports multiple crop types and diseases.

## Requirements

Make sure you have the following installed on your system:

- Python 3.8+
- Pip (Python package manager)

### Required Python Libraries

Install the following dependencies by running the command below:

```bash
pip install fastapi uvicorn tensorflow pillow opencv-python numpy
```

## How to Run the API

### 1. Clone the Repository

First, clone the (or place the crop_disease_api.py file into your project directory):

```bash
git clone https://github.com/krishnaGauss/Plant-Disease-Prediction-Model-Kaggle.git
cd <project-directory>
```

### 2. Set Up Your Python Environment

You can optionally create a virtual environment to isolate your project's dependencies:

```bash
python -m venv venv
# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Add Your Model File

Make sure your pre-trained Keras model (in this case trained_plant_disease_model.keras) is placed in the project directory, or update the path in the script if needed.

### 5. Download dataset.

Download the dataset from the given url [https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset]
Place the extracted test, train and valid folder in a parent directory named dataset.

### 6. Run the FastAPI Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn 'api_name':app --reload
```

Once the server is running, you should see something like this in the terminal:

```vbnet
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [XYZ] using watchgod
INFO:     Started server process [XYZ]
INFO:     Waiting for application startup.
```

### 7. Testing the API with Postman

To test the API, you can use Postman or cURL to upload an image and get a prediction.

#### Using Postman:

1.Open Postman and create a new POST request.
2.Set the URL to

```arduino
http://127.0.0.1:8000/predict/
```

3.Under the Body tab, select form-data.
4.Add a key named image, set its type to File, and upload an image file (e.g., a photo of a crop leaf).
5.Click Send to submit the request.
6.The response will contain the predicted disease in JSON format.

### 8. Example Response

After uploading an image, you will receive a response with the predicted disease (or healthy status) in the following format:

```json
{
  "prediction": "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
}
```

## Dictionary Structure

```graphql
.
├── crop_disease_api.py            # The main FastAPI app
├── trained_plant_disease_model.keras  # Pre-trained Keras model file
├── README.md                      # This file
├── dataset/                       # Folder containing test images (for manual testing)
```

---

This `README.md` file provides clear instructions on how to set up and run the crop disease prediction API, including testing it with Postman or cURL. Let me know if you need any more adjustments!
