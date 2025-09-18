import pandas as pd
import joblib
import numpy as np

# Wczytanie wytrenowanego modelu CatBoost, listy kolumn i threshold
cat_model, feature_order, threshold = joblib.load("Selected_model.pkl")

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Przygotowanie danych wejściowych:
    - one-hot encoding dla zmiennych kategorycznych
    - dodanie brakujących kolumn z zerami
    - ustawienie kolumn w kolejności feature_order
    """
    df_processed = pd.get_dummies(
        df,
        columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
        dtype=int
    )

    # Dodanie brakujących kolumn
    for col in feature_order:
        if col not in df_processed.columns:
            df_processed[col] = 0

    # Ustawienie kolejności kolumn
    df_processed = df_processed[feature_order]

    return df_processed

def predict_stroke(df: pd.DataFrame) -> np.ndarray:
    """
    Zwraca predykcję 0/1 na podstawie modelu i threshold
    """
    df_ready = preprocess_input(df)
    y_prob = cat_model.predict_proba(df_ready)[:, 1]
    y_pred = (y_prob > threshold).astype(int)
    return y_pred


# przykładowe dane wejściowe
data_dict = {
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
print(y_prob)

#['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Female', 'gender_Male', 
 #'gender_Other', 'ever_married_No', 'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked', 
 #'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'Residence_type_Rural', 'Residence_type_Urban',
 #  'smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
