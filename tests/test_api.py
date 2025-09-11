import pandas as pd
import joblib

# wczytanie modelu i kolejności kolumn
model, feature_order = joblib.load("Selected_model.pkl")

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:

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

