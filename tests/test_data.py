import pandas as pd

def test_data_integrity():
    df = pd.read_csv("data/test_sample.csv")
    #  NaN

    assert not df.isnull().values.any(), "W danych są NaN-y!"


    # columns
    expected_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender_Female', 'gender_Male',
                      'gender_Other', 'ever_married_No', 'ever_married_Yes', 'work_type_Govt_job', 
                      'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children', 
                      'Residence_type_Rural', 'Residence_type_Urban', 'smoking_status_Unknown', 
                      'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']
    assert list(df.columns) == expected_cols, "Kolumny w danych się nie zgadzają!"
