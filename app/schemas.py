from pydantic import BaseModel
from typing import Literal

class StrokeInput(BaseModel):
    age: int  # Wiek pacjenta
    hypertension: int  # 0 = brak nadciśnienia, 1 = występuje nadciśnienie
    heart_disease: int  # 0 = brak choroby serca, 1 = występuje choroba serca
    avg_glucose_level: float  # Średni poziom glukozy we krwi
    bmi: float  # Body Mass Index

    gender: Literal["Male", "Female", "Other"]
    ever_married: Literal["Yes", "No"]
    work_type: Literal["children", "Govt_job", "Never_worked", "Private", "Self-employed"]
    Residence_type: Literal["Urban", "Rural"]
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]

class StrokeOutput(BaseModel):
    stroke: int  # 0 = brak udaru, 1 = wystąpił udar

