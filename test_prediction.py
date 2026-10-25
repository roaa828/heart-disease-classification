
from src.predict import predict_heart_disease

# Example record using the dataset's feature order
example_patient = {
    "age": 55,
    "sex": 1,
    "cp": 1,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 3
}

result = predict_heart_disease(example_patient)

print("Model prediction:", result["label"])
print("Binary output:", result["prediction"])
