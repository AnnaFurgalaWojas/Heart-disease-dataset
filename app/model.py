import pandas as pd
import numpy as np
import joblib
from catboost import Pool


# Upload of model and necessary variables
data = joblib.load("Selected_model.pkl")

model = data["model"]
feature_order = data["feature_order"]
threshold = data["threshold"]
cat_columns = data["cat_columns"]

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df[feature_order].copy()
    for col in cat_columns:
        df[col] = df[col].astype(str)
    return df

def predict_stroke(df: pd.DataFrame) -> dict:
    df_ready = preprocess_input(df)
    
    pool = Pool(df_ready, cat_features=cat_columns)
    y_prob = model.predict_proba(pool)[:, 1]
    y_pred = (y_prob > threshold).astype(int)
    return {"Probability": float(y_prob[0]), "Prediction": int(y_pred[0])}

# przykładowe dane wejściowe
"""data_dict = {
    "age": [95],
    "hypertension": [1],
    "heart_disease": [1],
    "avg_glucose_level": [205.5],
    "bmi": [40.3],
    "gender": ["Female"],
    "ever_married": ["Yes"],
    "work_type": ["Private"],
    "Residence_type": ["Urban"],
    "smoking_status": ["never smoked"]
}

df = pd.DataFrame(data_dict)

# teraz df jest zdefiniowane
df_ready = preprocess_input(df)
y_prob = cat_model.predict_proba(df_ready)[:,1]
print(y_prob)"""