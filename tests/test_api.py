import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app
client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Stroke Prediction API is working!"}

def test_predict_valid_input():
    payload = {
        "age": 90,
        "hypertension": 1,
        "heart_disease": 0,
        "avg_glucose_level": 220.5,
        "bmi": 38.7,
        "gender": "Female",
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "smoking_status": "never smoked"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "stroke" in result
    assert result["stroke"] in [0, 1]

def test_predict_invalid_input():
    # brak pola wymagane przez Pydantic
    payload = {
        "age": 60,
        "hypertension": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # validation error


