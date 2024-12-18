# Breast Cancer Prediction App

This repository contains the code for a web application built with Streamlit that predicts whether a tumor is benign or malignant based on specific medical features. The app uses a pre-trained machine learning model to classify tumors based on user input and provides real-time predictions.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Information](#model-information)
- [License](#license)

## Overview

The Breast Cancer Prediction App is a simple and intuitive web interface for predicting the malignancy of a tumor. By entering values for a set of medical features such as `Clump Thickness`, `Uniformity of Cell Size`, and others, the model predicts whether the tumor is benign or malignant.

### Key Features:
- Interactive form for entering tumor features.
- Real-time prediction for tumor classification.
- Beautiful animations and user feedback with Streamlit.
- Ability to run the app locally and integrate it with a backend API (FastAPI).

## Technologies Used

- **Python**: The app is built using Python.
- **Streamlit**: For creating the web-based user interface.
- **Scikit-learn**: For model training and predictions.
- **Joblib**: For loading the pre-trained model and scaler.
- **Numpy**: For numerical operations and input handling.
- **Streamlit Extras**: For the rain animation effect when showing results.

## Installation

To run this app on your local machine, follow the instructions below:

### Prerequisites:
1. **Python 3.7+**: Make sure you have Python 3.7 or later installed.

2. **Install the required packages**:
    ```bash
    pip install streamlit scikit-learn numpy joblib requests
    pip install streamlit-extras
    ```

### Cloning the repository:
You can clone the repository using the following command:

```bash
git clone https://github.com/yourusername/breast-cancer-prediction.git
cd breast-cancer-prediction
```

### Model Files:
Ensure that the `model.pkl` and `scaler.pkl` files (trained model and scaler) are placed in a folder named `model/`. These files should be available from a previous model training process.

## Usage

1. Navigate to the folder containing your project in the terminal.
2. Run the following command to start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3. The app will open in your default web browser. Enter the values for each feature:
    - Clump Thickness
    - Uniformity of Cell Size
    - Uniformity of Cell Shape
    - Marginal Adhesion
    - Single Epithelial Cell Size
    - Bare Nuclei
    - Bland Chromatin
    - Normal Nucleoli
    - Mitoses

4. Click the **"Predict"** button to generate a prediction.

5. The app will display whether the tumor is **Benign** or **Malignant**.

## Features

- **Interactive UI**: Allows users to input feature values using sliders for each feature.
- **Prediction**: After clicking the "Predict" button, the app processes the input, scales the data, and displays the result.
- **Rain Animation**: A fun animation appears after the prediction is made, thanks to the `streamlit_extras` package.
- **Model Integration**: The app uses a pre-trained machine learning model and scaler loaded via `joblib`.

### Optional FastAPI Integration:
The code also includes commented-out lines to make the app work with an API (FastAPI). This can be enabled if you wish to use a separate backend for the model inference:

# Prepare the data for API request
```
input_data = { ... }  # Feature values here
```

# Make the API request
response = requests.post("http://127.0.0.1:8001/predict", json=input_data)

if response.status_code == 200:
    prediction = response.json()
    st.write(f"The prediction is {prediction['label']} (Class: {prediction['prediction']})")
else:
    print(response)
    st.write("Error: Could not retrieve prediction. Please try again.")

## Model Information

The model used in this application is a machine learning classifier trained on the breast cancer dataset. It predicts whether a tumor is benign (labeled as 2) or malignant (labeled as 4) based on features like clump thickness, uniformity of cell size, and others.

- **Model Type**: Classification (likely a Random Forest, SVM, or similar model)
- **Feature Set**: 9 features related to the cell structure of breast cancer cells.
- **Output**: Predicts either "Benign" or "Malignant" based on the input features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by submitting issues or pull requests. If you have any questions or need help setting up the app, please open an issue or contact me directly.
