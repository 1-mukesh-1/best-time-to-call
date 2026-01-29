"""Train the Best Time to Call model."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import joblib
import os

DATA_PATH = "data/carInsurance_train.csv"
MODEL_DIR = "model"

CAT_FEATURES = ['Job', 'Marital', 'Education', 'Communication', 'LastContactMonth', 'Outcome']
NUM_FEATURES = ['Age', 'Balance', 'HHInsurance', 'CarLoan', 'Default', 
                'LastContactDay', 'NoOfContacts', 'DaysPassed', 'PrevAttempts', 'CallHour']


def load_data(filepath):
    """Load CSV and extract call hour."""
    df = pd.read_csv(filepath)
    df['CallHour'] = pd.to_datetime(df['CallStart'], format='%H:%M:%S').dt.hour
    return df


def encode_features(df, label_encoders=None, fit=True):
    """Encode categorical features."""
    df = df.copy()
    
    for col in CAT_FEATURES:
        df[col] = df[col].fillna('Unknown').astype(str)
    
    if fit:
        label_encoders = {}
        for col in CAT_FEATURES:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
    else:
        for col in CAT_FEATURES:
            le = label_encoders[col]
            df[col + '_encoded'] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    feature_cols = [col + '_encoded' for col in CAT_FEATURES] + NUM_FEATURES
    return df, feature_cols, label_encoders


def train_model(df, feature_cols):
    """Train XGBoost classifier."""
    X = df[feature_cols]
    y = df['CarInsurance']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1,
        verbosity=1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob):.3f}")
    
    return model


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    
    print("Encoding features...")
    df, feature_cols, label_encoders = encode_features(df, fit=True)
    
    print("Training model...")
    model = train_model(df, feature_cols)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/xgb_model.joblib")
    joblib.dump(label_encoders, f"{MODEL_DIR}/label_encoders.joblib")
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.joblib")
    
    categories = {col: list(label_encoders[col].classes_) for col in CAT_FEATURES}
    joblib.dump(categories, f"{MODEL_DIR}/categories.joblib")
    
    print(f"Model saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
