import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def load_model():
    return joblib.load("Selected_model.pkl")

def test_model_loads():
    model = load_model()
    assert model is not None, "Model nie wczytał się poprawnie!"

def test_single_prediction():
    model = load_model()
    # przykładowe dane w takiej kolejności jak w X
    sample = np.array([[67.0, 0, 1, 228.69, 36.6,  # numeryczne cechy
                        0,1,0,  # gender_Female, gender_Male, gender_Other
                        0,1,    # ever_married_No, ever_married_Yes
                        0,1,0,0,1,0,  # work_type one-hot
                        0,1,0,0]])   # Residence + smoking_status one-hot
    pred = model.predict(sample)
    assert pred[0] in [0,1], "Model zwrócił niepoprawną klasę!"
    
def test_model_quality_on_sample():
    model = load_model()
    df = pd.read_csv("data/test_sample.csv")  # musisz mieć test.csv
    X, y = df.drop("stroke", axis=1), df["stroke"]
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    assert f1 > 0.9, f"F1-score za niski: {f1}"
    assert acc > 0.9, f"Accuracy za niskie: {acc}"
