from fastapi import FastAPI
import pandas as pd

from app.schemas import StrokeInput, StrokeOutput
from app.model import predict_stroke

# Inicjalization of FastAPI app
app = FastAPI(
    title="Stroke Prediction API",
    description="API for prediction of stroke base on patient's feautures.",
    version="1.0.0"
)

# Endpoint testowy
@app.get("/")
def root():
    return {"message": "Stroke Prediction API is working "}

# Endpoint predykcji
@app.post("/predict", response_model=StrokeOutput)
def predict(data: StrokeInput):
    #  DataFrame
    df = pd.DataFrame([data.dict()])

    # Predykcja
    prediction = predict_stroke(df)

    return {"stroke": int(prediction[0])}