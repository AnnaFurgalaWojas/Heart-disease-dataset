import pandas as pd
import joblib

# 1️⃣ Załaduj wytrenowany model
model = joblib.load("Selected_model.pkl")

# 2️⃣ Pobierz dokładne kolumny użyte podczas trenowania
MODEL_COLUMNS = model.feature_names_in_

# 3️⃣ Funkcja preprocess_input
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:


    # One-hot encoding
    df_encoded = pd.get_dummies(df)

    # Dopasowanie nazwy kolumny residence_type do liter w modelu
    if "residence_type" in df.columns:
        df_encoded.rename(columns={
            "residence_type_Rural": "Residence_type_Rural",
            "residence_type_Urban": "Residence_type_Urban"
        }, inplace=True)

    # Dodaj brakujące kolumny zerami
    for col in MODEL_COLUMNS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ustaw kolejność kolumn zgodnie z model.columns
    df_encoded = df_encoded[MODEL_COLUMNS]

    # Wszystkie kolumny jako int (0/1)
    df_encoded = df_encoded.astype(int)

    return df_encoded

# 4️⃣ Funkcja predykcji
def predict_stroke(df: pd.DataFrame):
    df_preprocessed = preprocess_input(df)
    prob = model.predict_proba(df_preprocessed)[:, 1]  # prawdopodobieństwo klasy 1
    return prob

# 5️⃣ Test predykcji
if __name__ == "__main__":
    test_data = {
        "age": 42,
        "hypertension": 0,
        "heart_disease": 0,
        "avg_glucose_level": 70,
        "bmi": 22,
        "gender": "Male",
        "ever_married": "Yes",
        "work_type": "Govt_job",
        "residence_type": "Urban",
        "smoking_status": "smokes"
    }

    df = pd.DataFrame([test_data])

    prob = predict_stroke(df)
    print("Prawdopodobieństwo udaru:", prob[0])
