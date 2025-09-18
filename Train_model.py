import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.metrics import classification_report, precision_recall_curve
import numpy as np

# --- 1. Wczytanie danych ---
data = pd.read_csv('/Users/annafurgala-wojas/New_project/Heart-disease-dataset/healthcare-dataset-stroke-data.csv')

# --- 2. Uzupełnianie brakujących wartości w bmi ---
data['bmi'] = data.groupby("stroke")['bmi'].transform(lambda x: x.fillna(x.median()))

# --- 3. Usunięcie kolumny id ---
data = data.drop(columns=["id"])

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE

# --- Dane: bardzo niezbalansowane (1% pozytywna klasa) ---
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    weights=[0.99, 0.01],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- SMOTE: oversampling klasy rzadkiej ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

feature_order = [f"X{i}" for i in range(X.shape[1])]

# --- scale_pos_weight ---
scale_pos_weight = sum(y_train_res == 0) / sum(y_train_res == 1)

def find_threshold_for_recall(y_true, y_prob, min_recall=0.7):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # znajdź pierwszy threshold, gdzie recall >= min_recall
    idxs = np.where(recall >= min_recall)[0]
    if len(idxs) == 0:
        return None, precision, recall, None  # brak progu spełniającego warunek
    best_idx = idxs[-1]  # weź próg z najwyższą precyzją przy tym recall
    return thresholds[best_idx], precision, recall, best_idx

# --- 3. CatBoost ---
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    eval_metric='Recall',
    class_weights=[1, scale_pos_weight],
    random_state=42,
    verbose=False
)

cat_model.fit(X_train_res, y_train_res)
y_prob_cat = cat_model.predict_proba(X_test)[:, 1]

thr_rec_cat, _, _, idx_rec_cat = find_threshold_for_recall(y_test, y_prob_cat, min_recall=0.7)
y_pred_rec_cat = (y_prob_cat > thr_rec_cat).astype(int) if thr_rec_cat else None


print("\n=== CatBoost ===")
if y_pred_rec_cat is not None:
    print("Threshold dla recall>=0.7:", round(thr_rec_cat, 4))
    print(classification_report(y_test, y_pred_rec_cat, digits=4))

import joblib

# Zapis modelu wraz z feature_order i threshold
feature_order = [f"X{i}" for i in range(X.shape[1])]
threshold = 0.5  # lub threshold wyliczony na zbiorze testowym
joblib.dump((cat_model, feature_order, threshold), "Selected_model.pkl")