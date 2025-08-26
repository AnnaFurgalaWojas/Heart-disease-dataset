from fastapi import FastAPI
import pandas as pd
from app.schemas import StrokeInput, StrokeOutput
from app.model import predict_stroke

app = FastAPI(
    title="Stroke Prediction API",
    description="API for predicting stroke based on patient's features",
    version="1.0"
)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Stroke Prediction API is working!"}

# Endpoint predykcji
@app.post("/predict", response_model=StrokeOutput)
def predict(input_data: StrokeInput):
    # Zamiana danych wejściowych z Pydantic na DataFrame
    df = pd.DataFrame([input_data.dict()])

    # Predykcja
    prediction = predict_stroke(df)

    return {"stroke": int(prediction[0])}  # 0 lub 1
