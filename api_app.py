import numpy as np
import pandas as pd
from fastapi import FastAPI
import joblib
import pandas as pd

# --- Exact make_features copied from notebook/main.py ---

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Interaction/Rate Features
    df["meeting_ratio"] = df["meetings_held"] / (df["days_open"] + 1)
    df["actions_per_meeting"] = df["actions_completed"] / (df["meetings_held"] + 1)
    df["actions_per_day"] = df["actions_completed"] / (df["days_open"] + 1)
    df["sales_velocity"] = df["opportunity_size"] / (df["days_open"] + 1)

    # Opportunity Value Transformations
    df["log_opportunity_size"] = np.log1p(df["opportunity_size"])
    df["opp_per_meeting"] = df["opportunity_size"] / (df["meetings_held"] + 1)

    # Pipeline Stage Features
    df["stage_squared"] = df["stage"] ** 2
    df["stage_meeting_interaction"] = df["stage"] * df["meetings_held"]

    # Time-based Features
    median_days = df["days_open"].median() if "days_open" in df else 0
    df["fast_deal"] = (df["days_open"] < median_days).astype(int)
    df["deal_age_bucket"] = pd.cut(
        df["days_open"], bins=[-1, 30, 60, 90, 180, np.inf], labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Binary Indicators
    median_size = df["opportunity_size"].median() if "opportunity_size" in df else 0
    median_actions = df["actions_completed"].median() if "actions_completed" in df else 0
    df["high_value"] = (df["opportunity_size"] > median_size).astype(int)
    df["high_activity"] = (df["actions_completed"] > median_actions).astype(int)

    return df

# --- Load artifacts ---
model = joblib.load("sales_win_model.pkl")
scaler = joblib.load("scaler.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")

app = FastAPI(title="Sales Opportunity Win Predictor (RF)")

# Helper to transform raw input into full feature set

def preprocess_payload(payload: dict) -> pd.DataFrame:
    raw = pd.DataFrame([payload])
    # Ensure required base fields exist
    required = [
        "opportunity_size",
        "meetings_held",
        "actions_completed",
        "days_open",
        "stage",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    # Build engineered features using the same logic as training
    full = make_features(raw)

    # Reorder/select exactly the training feature columns
    full = full[FEATURE_COLUMNS]
    return full


@app.post("/predict/")
def predict(data: dict):
    try:
        feats = preprocess_payload(data)
        feats_scaled = scaler.transform(feats)
        prob = float(model.predict_proba(feats_scaled)[0][1])
        pred = int(model.predict(feats_scaled)[0])
        return {
            "model": "RandomForest (deployed)",
            "reason": "RF selected due to best ROC-AUC among RF/LGBM/Stack",
            "prediction": "Won" if pred == 1 else "Lost",
            "probability": round(prob, 4),
        }
    except Exception as e:
        return {"error": str(e)}
