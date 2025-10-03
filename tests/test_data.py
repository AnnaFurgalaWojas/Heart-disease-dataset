import pandas as pd  
import joblib        
import pytest


dataset_train = pd.read_csv('/Users/annafurgala-wojas/Projects/Heart-disease-dataset/data/train_data.csv')
dataset_test = pd.read_csv('/Users/annafurgala-wojas/Projects/Heart-disease-dataset/data/test_data.csv')

def test_missing_values(dataset_train):
    assert dataset_train.isnull().sum().sum() == 0, "There are missing values in the dataset."

def test_columns_types(dataset_train):
    expected_types = {"gender": "object",
    "age": "float64",
    "hypertension": "int64",
    "heart_disease": "int64",
    "ever_married": "object",
    "work_type": "object",
    "Residence_type": "object",
    "avg_glucose_level": "float64",
    "bmi": "float64",
    "smoking_status": "object",
    "stroke": "int64"}
    for col, dtype in expected_types.items():
        assert dataset_train[col].dtype == dtype, f"Column {col} has incorrect type. Expected {dtype}, got {dataset_train[col].dtype}."

def test_data_structure_in_columns(dataset_train):
    assert set(dataset_train['gender'].unique()).issubset({"Male", "Female", "Other"}), "Unexpected values in gender columns!"
    assert set(dataset_train['ever married'].unique()).issubset({'Yes' ,'No'}), "Unexpected values in ever married columns!"
    assert set(dataset_train['work_type'].unique()).issubset({'Private','Self-employed' ,'Govt_job', 'children', 'Never_worked'}), "Unexpected values in work type columns!"
    assert set(dataset_train['Residence_type'].unique()).issubset({'Urban', 'Rural'}), "Unexpected values in Residence_type columns!"
    assert set(dataset_train['smoking_status'].unique()).issubset({'formerly smoked' ,'never smoked', 'smokes', 'Unknown'}), "Unexpected values in smoking_status columns!"

def test_no_data_leakage(dataset_train, dataset_test):
    overlap = pd.merge(dataset_train, dataset_test, how='inner')
    assert overlap.empty, "Data leakage detected between training and testing sets!"

def test_synthetic_data(dataset_train):
    df = pd.DataFrame({
    "gender": ["Male", "Female", "Male", "Other", "Female"],
    "age": [45.0, 84.5, 63.0, 67.2, 54.1],
    "hypertension": [0, 1, 0, 1, 0],
    "heart_disease": [0, 1, 0, 1, 0],
    "ever_married": ["Yes", "No", "Yes", "Yes", "No"],
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    "Residence_type":  ["Urban", "Rural", "Urban", "Rural", "Urban"],
    "avg_glucose_level": [105.92, 234.56, 98.76, 150.34, 120.45],
    "bmi": [25.4, 30.1, 22.5, 28.3, 26.7],
    "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown", "never smoked"]
    })