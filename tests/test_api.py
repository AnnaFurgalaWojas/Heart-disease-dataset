from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_endpoint():
    sample = {
        "age": 67.0,
        "hypertension": 0,
        "heart_disease": 1,
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "gender_Female": 0,
        "gender_Male": 1,
        "gender_Other": 0,
        "ever_married_No": 0,
        "ever_married_Yes": 1,
        "work_type_Govt_job": 0,
        "work_type_Never_worked": 0,
        "work_type_Private": 1,
        "work_type_Self-employed": 0,
        "work_type_children": 0,
        "Residence_type_Rural": 0,
        "Residence_type_Urban": 1,
        "smoking_status_Unknown": 0,
        "smoking_status_formerly smoked": 1,
        "smoking_status_never smoked": 0,
        "smoking_status_smokes": 0
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0,1]
