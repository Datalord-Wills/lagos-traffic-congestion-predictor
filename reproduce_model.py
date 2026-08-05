"""
Same model as reproduce_model.py (identical wrangle(), features, target,
Random Forest hyperparameter search) -- the ONLY change is swapping
category_encoders.OneHotEncoder for sklearn's own OneHotEncoder inside a
ColumnTransformer. This avoids cross-environment pickle compatibility
issues that category_encoders can run into on platforms like Streamlit
Community Cloud.
"""

import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


# ----------------------------------------------------------------------
# Wrangle function (identical to Model.ipynb)
# ----------------------------------------------------------------------
def wrangle(URL):
    df = pd.read_csv(URL)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_of_month'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Is_weekend'] = df['Day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
    df['hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df.drop(columns=['Time', 'hour', 'Date', 'State'], inplace=True)

    order = {'Heavy': 4, 'High': 3, 'Normal': 2, 'Low': 1}
    df['Traffic_Situation'] = df['Traffic_Situation'].str.split("-").str[-1]
    df['Traffic_Situation'] = df['Traffic_Situation'].map(order)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    df.drop(columns=["Total", "Is_Raining"], inplace=True)

    return df


# ----------------------------------------------------------------------
# Load + split (identical to Model.ipynb)
# ----------------------------------------------------------------------
traffic_data = wrangle("lagos_synthetic_traffic_dataset.csv")
print("Wrangled shape:", traffic_data.shape)

X = traffic_data.drop(columns=['Traffic_Situation'])
y = traffic_data['Traffic_Situation']

CATEGORICAL_COLS = ['Day_of_week', 'Segment', 'Road_Type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ----------------------------------------------------------------------
# Random Forest + sklearn OneHotEncoder (via ColumnTransformer)
# ----------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False,
)
preprocessor.set_output(transform="pandas")

clf_rf = make_pipeline(
    preprocessor,
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
# Evaluation
# ----------------------------------------------------------------------
y_pred = model.predict(X_test)
test_f1_macro = f1_score(y_test, y_pred, average='macro')
train_f1_macro = f1_score(y_train, model.predict(X_train), average='macro')

print(f"\nTrain F1-macro: {train_f1_macro:.3f}")
print(f"Test F1-macro:  {test_f1_macro:.3f}")
print("\nClassification report (test):")
print(classification_report(y_test, y_pred, target_names=['Low', 'Normal', 'High', 'Heavy']))

# ----------------------------------------------------------------------
# Segment metadata + model save
# ----------------------------------------------------------------------
raw = pd.read_csv("lagos_synthetic_traffic_dataset.csv")
segment_meta = raw[['Segment', 'Road_Type', 'Lanes']].drop_duplicates().sort_values('Segment').reset_index(drop=True)
segment_meta.to_csv("segment_metadata.csv", index=False)

joblib.dump(model, "model.pkl", compress=3)
print("\nSaved model.pkl (sklearn OneHotEncoder version)")

encoded_features = model.named_steps["columntransformer"].get_feature_names_out()
print("Model expects columns (post-encoding):", list(encoded_features))
print("Model expects raw input columns (in order):", list(X.columns))
