import pandas as pd
import numpy as np
import pytest
from app.model import preprocess_input, predict_stroke

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
    # wszystkie kolumny z feature_order muszą być
    from app.model import feature_order
    assert all(col in df_ready.columns for col in feature_order)

def test_predict_stroke_output(sample_input):
    y_pred = predict_stroke(sample_input)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (1,)
    assert y_pred[0] in [0, 1]
