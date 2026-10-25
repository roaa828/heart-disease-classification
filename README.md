# Heart Disease Classification

A machine learning project that uses clinical features to classify heart disease status with a trained Random Forest model. The project includes exploratory data analysis, model evaluation, and an interactive web application built with Streamlit.

> **Disclaimer:** This project is for educational purposes only. It is not a medical diagnostic tool and must not be used to make healthcare decisions.

## Project Overview

The goal of this project is to explore clinical data, train a classification model, evaluate its performance, and make predictions through a user-friendly interface.

Users can enter patient feature values and view the model's predicted class, its output probabilities, and a summary of the entered data.

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing and machine learning
- Random Forest classification
- Model evaluation using Accuracy, Precision, Recall, and F1-score
- Feature importance visualization
- Model comparison charts
- Interactive Streamlit application
- Individual predictions with model output probabilities

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Model Performance

The previously recorded Random Forest test-set results are:

| Metric | Score |
|---|---:|
| Accuracy | 91.8% |
| Precision | 87.1% |
| Recall | 96.4% |
| F1-score | 91.5% |

These scores describe the model's performance on the evaluated test set. They do not guarantee the same performance on new data.

**Evaluation note:** The recorded cross-validation results differ from the test-set results. The evaluation pipeline and data splits should be reviewed before making conclusions about the model's generalization performance.

## Project Structure

```text
heart-disease-classification/
├── app.py
├── test_prediction.py
├── requirements.txt
├── data/
│   └── raw/
│       └── processed.cleveland.data
├── models/
│   └── heart_disease_random_forest.joblib
├── notebooks/
│   └── 01_heart_disease_eda.ipynb
├── reports/
│   ├── model_comparison.png
│   └── feature_importance.png
└── src/
    └── predict.py
```

## Dataset

This project uses the processed Cleveland heart disease dataset.

The model uses 13 input features, including age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, exercise-induced angina, and other clinical test features.

The target is represented as a binary classification:

- `0`: No disease detected in the dataset label
- `1`: Disease detected in the dataset label

Dataset labels and model predictions are not equivalent to a clinical diagnosis.

## Installation

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment.

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

4. Install the required dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Application

From the project root, run:

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local URL to open the application in your browser.

## Run the Prediction Test

```powershell
python test_prediction.py
```

## Visualizations

The project includes the following analysis outputs:

- `reports/model_comparison.png` — comparison of model evaluation metrics.
- `reports/feature_importance.png` — feature importance from the Random Forest model.

## Future Improvements

- Review and strengthen the model evaluation process.
- Add input validation and more automated tests.
- Improve interpretability of individual predictions.
- Deploy the application online.
- Document the full preprocessing and training workflow.

## Author

Information Technology Engineering student interested in data analysis and machine learning.
