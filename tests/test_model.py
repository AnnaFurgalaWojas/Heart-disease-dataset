import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.model import preprocess_input, predict_stroke, feature_order

@pytest.fixture
def sample_input():
    return pd.DataFrame([{
        "age": 45,
        "hypertension": 0,
        "heart_disease": 0,
        "avg_glucose_level": 95.5,
        "bmi": 22.4,
        "gender": "Male",
        "ever_married": "No",
        "work_type": "Private",
        "Residence_type": "Urban",
        "smoking_status": "never smoked"
    }])

def test_preprocess_input_columns(sample_input):
    df_ready = preprocess_input(sample_input)
    assert isinstance(df_ready, pd.DataFrame)
    assert all(col in df_ready.columns for col in feature_order)

def test_predict_stroke_output(sample_input):
    y_pred = predict_stroke(sample_input)
    assert isinstance(y_pred, dict)
    assert "Probability" in y_pred and "Prediction" in y_pred
    assert y_pred["Prediction"] in [0, 1]
    assert 0.0 <= y_pred["Probability"] <= 1.0
