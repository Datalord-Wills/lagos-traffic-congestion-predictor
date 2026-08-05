"""
Reproduces the exact model training pipeline from Model.ipynb so that
model.pkl matches what was evaluated in the capstone notebook
(Random Forest, F1-macro ~0.947 on the held-out test set).

This is NOT a new/different model -- it's the same wrangle(), same
train/test split, same RandomizedSearchCV grid and random_state, on the
same dataset, producing the same trained pipeline.
"""

import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from category_encoders import OneHotEncoder
from sklearn.pipeline import make_pipeline


# ----------------------------------------------------------------------
# Wrangle function (identical to Model.ipynb)
# ----------------------------------------------------------------------
def wrangle(URL):
    df = pd.read_csv(URL)
    # Data preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_month'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Is_weekend'] = df['Day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
    df['hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Drop columns that are not needed
    df.drop(columns=['Time', 'hour', 'Date', 'State'], inplace=True)

    # Change the order of the "target" to ensure an ordinal progression and split
    order = {'Heavy': 4, 'High': 3, 'Normal': 2, 'Low': 1}
    df['Traffic_Situation'] = df['Traffic_Situation'].str.split("-").str[-1]
    df['Traffic_Situation'] = df['Traffic_Situation'].map(order)

    # Convert "bool" columns to integer
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Drop leakages
    df.drop(columns=["Total", "Is_Raining"], inplace=True)

    return df


# ----------------------------------------------------------------------
# Load + split (identical to Model.ipynb)
# ----------------------------------------------------------------------
traffic_data = wrangle("lagos_synthetic_traffic_dataset.csv")
print("Wrangled shape:", traffic_data.shape)
print("Feature columns:", [c for c in traffic_data.columns if c != 'Traffic_Situation'])

X = traffic_data.drop(columns=['Traffic_Situation'])
y = traffic_data['Traffic_Situation']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------------------
# Random Forest + RandomizedSearchCV (identical grid/params to Model.ipynb)
# ----------------------------------------------------------------------
clf_rf = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    RandomForestClassifier(random_state=42, class_weight="balanced")
)

param_grid_rf = {
    'randomforestclassifier__n_estimators': [100, 200, 300, 400, 500],
    'randomforestclassifier__max_depth': [None, 5, 10, 20],
    'randomforestclassifier__min_samples_split': [2, 5, 10],
    'randomforestclassifier__min_samples_leaf': [1, 2, 4]
}

kfold = KFold(n_splits=5, random_state=42, shuffle=True)

random_search_rf = RandomizedSearchCV(
    estimator=clf_rf,
    param_distributions=param_grid_rf,
    n_iter=10,
    cv=kfold,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_result_rf = random_search_rf.fit(X_train, y_train)

print("Best CV score:", random_result_rf.best_score_)
print("Best parameters:", random_result_rf.best_params_)

model = random_result_rf.best_estimator_

# ----------------------------------------------------------------------
# Evaluation (matches Model.ipynb reporting)
# ----------------------------------------------------------------------
y_pred = model.predict(X_test)
test_f1_macro = f1_score(y_test, y_pred, average='macro')
train_f1_macro = f1_score(y_train, model.predict(X_train), average='macro')

print(f"\nTrain F1-macro: {train_f1_macro:.3f}")
print(f"Test F1-macro:  {test_f1_macro:.3f}")
print("\nClassification report (test):")
print(classification_report(y_test, y_pred, target_names=['Low', 'Normal', 'High', 'Heavy']))

# ----------------------------------------------------------------------
# Save segment metadata lookup (used by the Streamlit app to auto-fill
# Road_Type / Lanes once a user picks a Segment)
# ----------------------------------------------------------------------
segment_meta = (
    traffic_data.merge(
        pd.read_csv("lagos_synthetic_traffic_dataset.csv")[['Segment', 'Road_Type', 'Lanes']].drop_duplicates(),
        left_index=False, right_index=False, how="cross"
    ) if False else None
)
# Simpler: pull directly from the raw CSV (Segment/Road_Type/Lanes are static per segment)
raw = pd.read_csv("lagos_synthetic_traffic_dataset.csv")
segment_meta = raw[['Segment', 'Road_Type', 'Lanes']].drop_duplicates().sort_values('Segment').reset_index(drop=True)
segment_meta.to_csv("segment_metadata.csv", index=False)
print("\nSegment metadata:")
print(segment_meta)

# ----------------------------------------------------------------------
# Save model (matches Model.ipynb's joblib.dump(model, "model.pkl"))
# ----------------------------------------------------------------------
joblib.dump(model, "model.pkl")
print("\nSaved model.pkl")
print("Model expects columns (in order):", list(model.named_steps["onehotencoder"].feature_names_in_))
