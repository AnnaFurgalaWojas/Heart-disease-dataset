from pydantic import BaseModel
from typing import Literal

class StrokeInput(BaseModel):
    age: int  # Age of patient
    hypertension: int  # 0 = No hypertension, 1 = Present hypertension
    heart_disease: int  # 0 = No heart disease, 1 = Present heart disease
    avg_glucose_level: float  # Average glucose level in blood
    bmi: float  # Body Mass Index

    gender: Literal["Male", "Female", "Other"]
    ever_married: Literal["Yes", "No"]
    work_type: Literal["children", "Govt_job", "Never_worked", "Private", "Self-employed"]
    Residence_type: Literal["Urban", "Rural"]
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]

class StrokeOutput(BaseModel):
    stroke: int  # 0 = No stoke, 1 = Present stroke

