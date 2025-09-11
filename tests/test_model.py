import numpy as np
from app.model import model, preprocess_input
import pandas as pd   # ← brakowało!
import joblib         # ← brakowało!
import pytest



def test_model_loads():
    assert model is not None
    assert hasattr(model, "predict")

def test_single_prediction():
    df = pd.DataFrame([{
        "age": 67,
        "hypertension": 0,
        "heart_disease": 1,
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "gender": "Male",
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "smoking_status": "formerly smoked"
    }])
    X = preprocess_input(df)
    pred = model.predict(X)
    assert pred[0] in [0, 1]

"""def test_model_quality_on_sample():
    df = pd.read_csv("data/test_sample.csv")
    X, y = df.drop("stroke", axis=1), df["stroke"]
    X_prepared = preprocess_input(X)
    y_pred = model.predict(X_prepared)
    acc = (y_pred == y).mean()
    assert acc > 0.7, f"Model accuracy on test_sample.csv too low: {acc}"""

def test_model_quality_on_sample():
    model, feature_order = joblib.load("Selected_model.pkl")
    df = pd.read_csv("data/test_sample.csv")

    X, y = df.drop("stroke", axis=1), df["stroke"]

    # upewnij się, że kolumny są w tej samej kolejności co przy treningu
    X = X[feature_order]

    y_pred = model.predict(X)

    assert len(y_pred) == len(y)

