from fastapi import FastAPI
import pandas as pd
from app.schemas import StrokeInput, StrokeOutput
from app.model import predict_stroke

app = FastAPI(
    title="Stroke Prediction API",
    description="API for predicting stroke based on patient's features",
    version="1.0"
)

@app.get("/")
def read_root():
    return {"message": "Stroke Prediction API is working!"}

@app.post("/predict", response_model=StrokeOutput)
def predict(input_data: StrokeInput):
    # Konwersja inputu do DataFrame
    df = pd.DataFrame([input_data.dict()])

    try:
        prediction = predict_stroke(df)
        # Z model.py dostajemy dict {"Probability": ..., "Prediction": ...}
        stroke_result = prediction["Prediction"]
    except Exception as e:
        # W przypadku błędu zwracamy None (opcjonalnie można też raise HTTPException)
        return {"stroke": None}

    return {"stroke": stroke_result}

