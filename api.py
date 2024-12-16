import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Load the saved model
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Instantiate FastAPI app
app = FastAPI(
    title="Logistic Regression Classifier API",
    description="An API for predicting breast cancer malignancy (benign or malignant) using logistic regression.",
    version="1.0.0"
)


# Define the input data structure
class PredictionInput(BaseModel):
    clump_thickness: int
    uniformity_of_cell_size: int
    uniformity_of_cell_shape: int
    marginal_adhesion: int
    single_epithelial_cell_size: int
    bare_nuclei: int
    bland_chromatin: int
    normal_nucleoli: int
    mitoses: int


@app.post(
    "/predict",
    summary="Predict Class Label",
    description="Predict whether the tumor is benign (2) or malignant (4).",
    tags=["Prediction"]
)
def predict(input_data: PredictionInput):
    """
    :param input_data:
    - Clump Thickness
    - Uniformity of Cell Size
    - Uniformity of Cell Shape
    - Marginal Adhesion
    - Single Epithelial Cell Size
    - Bare Nuclei
    - Bland Chromatin
    - Normal Nucleoli
    - Mitoses

    :return:
    - Prediction: 2 for benign, 4 for malignant.
    - Label: String representation of the prediction (Benign/Malignant).
    """
    # prepare the data for prediction
    data = np.array([[
        input_data.clump_thickness,
        input_data.uniformity_of_cell_size,
        input_data.uniformity_of_cell_shape,
        input_data.marginal_adhesion,
        input_data.single_epithelial_cell_size,
        input_data.bare_nuclei,
        input_data.bland_chromatin,
        input_data.normal_nucleoli,
        input_data.mitoses
    ]])

    # Scale the input data
    scaled_data = scaler.transform(data)

    # Make the prediction
    prediction = model.predict(scaled_data)[0]
    label = "Benign" if prediction == 2 else "Malignant"

    return {
        "prediction": int(prediction),
        "label": label
    }


# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
