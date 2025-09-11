import pandas as pd
import joblib

# wczytanie modelu i kolejności kolumn
model, feature_order = joblib.load("Selected_model.pkl")

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zamienia dane wejściowe na format zgodny z treningiem:
    - one-hot encoding dla kolumn kategorycznych
    - dodanie brakujących kolumn z zerami
    - ustawienie kolumn w kolejności feature_order
    """
    df_processed = pd.get_dummies(
        df,
        columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
        dtype=int
    )

    # dodaj brakujące kolumny
    for col in feature_order:
        if col not in df_processed:
            df_processed[col] = 0

    # ustaw kolejność kolumn
    df_processed = df_processed[feature_order]

    return df_processed

def predict_stroke(df: pd.DataFrame):
    df_prepared = preprocess_input(df)
    return model.predict(df_prepared)




#['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Female', 'gender_Male', 
 #'gender_Other', 'ever_married_No', 'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked', 
 #'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'Residence_type_Rural', 'Residence_type_Urban',
 #  'smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
