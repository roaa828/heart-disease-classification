
import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_random_forest.joblib"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]


def load_model():
    """Load the saved machine learning pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_heart_disease(patient_data):
    """
    Predict the binary target for one patient.
    Educational demonstration only, not a medical diagnosis.
    """
    model = load_model()

    patient_df = pd.DataFrame(
        [patient_data],
        columns=FEATURES
    )

    prediction = int(model.predict(patient_df)[0])

    result = {
        "prediction": prediction,
        "label": (
            "Disease detected by model"
            if prediction == 1
            else "No disease detected by model"
        ),
        "probabilities": None
    }

    # Retrieve model probabilities when supported.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(patient_df)[0]
        classes = model.classes_

        result["probabilities"] = {
            int(class_label): float(probability)
            for class_label, probability
            in zip(classes, probabilities)
        }

    return result
