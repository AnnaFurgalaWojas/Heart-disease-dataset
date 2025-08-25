import joblib
import pandas as pd


model = joblib.load("Selected_model.pkl")


MODEL_COLUMNS = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "gender_Female",
    "gender_Male",
    "gender_Other",
    "ever_married_No",
    "ever_married_Yes",
    "work_type_children",
    "work_type_Govt_job",
    "work_type_Never_worked",
    "work_type_Private",
    "work_type_Self-employed",
    "residence_type_Rural",
    "residence_type_Urban",
    "smoking_status_formerly smoked",
    "smoking_status_never smoked",
    "smoking_status_smokes",
    "smoking_status_Unknown"
]

#  preprocess_input
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
   
   
    df_encoded = pd.get_dummies(df)

    
    for col in MODEL_COLUMNS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    
    df_encoded = df_encoded[MODEL_COLUMNS]

    return df_encoded

# Predyction
def predict_stroke(df: pd.DataFrame):
    df_preprocessed = preprocess_input(df)
    prediction = model.predict(df_preprocessed)
    return prediction
