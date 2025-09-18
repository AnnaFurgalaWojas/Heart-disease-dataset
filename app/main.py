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
    """
    Przyjmuje dane pacjenta w formacie Pydantic i zwraca predykcję 0/1
    """
    # Konwersja danych wejściowych do DataFrame
    df = pd.DataFrame([input_data.dict()])

    # Predykcja
    try:
        prediction = predict_stroke(df)
        stroke_result = int(prediction[0])
    except Exception as e:
        return {"stroke": None, "error": str(e)}

    return {"stroke": stroke_result}
