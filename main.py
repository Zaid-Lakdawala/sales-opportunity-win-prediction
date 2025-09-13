# 1) Generate synthetic sales data with a realistic, non-linear win probability
import numpy as np
import pandas as pd


def generate_sales_data(n=1000, random_state=42):
    """Create a synthetic dataset with base CRM-like features and a hidden win process.

    Returns a DataFrame with columns:
      opportunity_size, meetings_held, actions_completed, days_open, stage, win
    """
    np.random.seed(random_state)

    # Base features (simple, interpretable distributions)
    df = pd.DataFrame({
        "opportunity_size": np.random.randint(5000, 100000, n),
        "meetings_held": np.random.poisson(3, n),
        "actions_completed": np.random.poisson(5, n),
        "days_open": np.random.randint(5, 200, n),
        "stage": np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.2, 0.25, 0.25, 0.2, 0.1]),
    })

    # Hidden non-linear formula for win propensity
    logits = (
        0.00007 * df["opportunity_size"]
        + 0.35 * np.log1p(df["meetings_held"])  # diminishing returns on meetings
        + 0.25 * np.sqrt(df["actions_completed"])  # diminishing returns on actions
        + 0.4 * (df["stage"] >= 4).astype(int)  # big lift late in funnel
        + 0.15 * (df["stage"] == 5).astype(int)  # extra push at closed-won
        - 0.03 * np.log1p(df["days_open"])  # old deals decay
        + 0.2 * ((df["meetings_held"] > 3) & (df["stage"] >= 3)).astype(int)  # interaction
    )

    # Sigmoid-transform logits to probabilities, centered around mean for spread
    probs = 1 / (1 + np.exp(-0.5 * (logits - logits.mean())))

    # Sample binary outcomes from probabilities
    df["win"] = np.random.binomial(1, probs)
    return df

# Example run
df = generate_sales_data(1000)
print(df.head())
print("\nWin/Loss distribution:")
print(df["win"].value_counts(normalize=True))
print("\nCorrelations with target:")
print(df.corr(numeric_only=True)["win"].sort_values(ascending=False))# 2) Feature engineering helper used in both training and API
import numpy as np
import pandas as pd


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

# Apply features to training data
df = make_features(df)
df.head()# 3) Quick EDA: class balance and basic stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="win", data=df)
plt.title("Win vs Lost Distribution")
plt.show()

# Sanity checks
df.isnull().sum()
df.describe()# 4) Train/test split and scaling (keep consistent across models)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(["win"], axis=1)
y = df["win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the exact feature order for API inference
FEATURE_COLUMNS = X.columns.tolist()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)# 5) Baseline model: RandomForest (great for tabular)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

rf_baseline = RandomForestClassifier(n_estimators=300, random_state=42)
rf_baseline.fit(X_train_scaled, y_train)

y_pred = rf_baseline.predict(X_test_scaled)
y_prob = rf_baseline.predict_proba(X_test_scaled)[:, 1]

print("RF Baseline Accuracy:", accuracy_score(y_test, y_pred))
print("RF Baseline ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nRF Baseline Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Lost", "Won"], yticklabels=["Lost", "Won"])
plt.title("RF Baseline Confusion Matrix")
plt.show()# 6) Hyperparameter tuning: RandomForest with GridSearchCV (optimize accuracy)
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gs_rf = GridSearchCV(rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs_rf.fit(X_train_scaled, y_train)

print("Best RF Params:", gs_rf.best_params_)
print("Best RF CV Accuracy:", gs_rf.best_score_)

best_rf = gs_rf.best_estimator_

# Evaluate tuned RF on test set
y_pred = best_rf.predict(X_test_scaled)
y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]
print("Tuned RF Test Accuracy:", accuracy_score(y_test, y_pred))
print("Tuned RF Test ROC-AUC:", roc_auc_score(y_test, y_prob))# 7) LightGBM: baseline + tuning (for comparison)

import lightgbm as lgb

lgb_base = lgb.LGBMClassifier(random_state=42)
lgb_base.fit(X_train_scaled, y_train)

y_pred = lgb_base.predict(X_test_scaled)
y_prob = lgb_base.predict_proba(X_test_scaled)[:, 1]
print("LGBM Baseline Accuracy:", accuracy_score(y_test, y_pred))
print("LGBM Baseline ROC-AUC:", roc_auc_score(y_test, y_prob))

# Tuning
param_grid = {
    'num_leaves': [31, 50, 70],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500]
}

gs_lgb = GridSearchCV(lgb.LGBMClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs_lgb.fit(X_train_scaled, y_train)
print("Best LGBM Params:", gs_lgb.best_params_)
print("Best LGBM CV Accuracy:", gs_lgb.best_score_)

best_lgb = gs_lgb.best_estimator_
y_pred = best_lgb.predict(X_test_scaled)
y_prob = best_lgb.predict_proba(X_test_scaled)[:, 1]
print("Tuned LGBM Test Accuracy:", accuracy_score(y_test, y_pred))
print("Tuned LGBM Test ROC-AUC:", roc_auc_score(y_test, y_prob))# 8) Stacking ensemble: RF + LGBM with LogisticRegression meta-learner
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stack = StackingClassifier(
    estimators=[('rf', best_rf), ('lgbm', best_lgb)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    stack_method='predict_proba',
)

stack.fit(X_train_scaled, y_train)
y_pred = stack.predict(X_test_scaled)
y_prob = stack.predict_proba(X_test_scaled)[:, 1]
print("Stack Accuracy:", accuracy_score(y_test, y_pred))
print("Stack ROC-AUC:", roc_auc_score(y_test, y_prob))# 9) Cross-validation, calibration, and ROC curves
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

cv_scores = cross_val_score(stack, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print("Stack CV ROC-AUC:", cv_scores, "Mean:", np.mean(cv_scores))

rf_cal = CalibratedClassifierCV(best_rf, cv=5)
lgb_cal = CalibratedClassifierCV(best_lgb, cv=5)
stack_cal = CalibratedClassifierCV(stack, cv=5)

rf_cal.fit(X_train_scaled, y_train)
lgb_cal.fit(X_train_scaled, y_train)
stack_cal.fit(X_train_scaled, y_train)

models = {"RandomForest": rf_cal, "LightGBM": lgb_cal, "Stacked": stack_cal}
plt.figure(figsize=(8,6))
for name, mdl in models.items():
    y_pred = mdl.predict(X_test_scaled)
    y_prob = mdl.predict_proba(X_test_scaled)[:, 1]
    print(f"\n{name} Acc: {accuracy_score(y_test, y_pred):.3f}")
    print(f"{name} ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")

plt.plot([0,1],[0,1], 'k--')
plt.title("ROC Curve Comparison (Calibrated)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()# 10) Persist deployment artifacts (RF chosen for best ROC-AUC)
import joblib

joblib.dump(best_rf, "sales_win_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")
print("Saved best RF, scaler, and feature columns for API.")

# 11) FastAPI for inference (uses RandomForest only)
# We deploy RandomForest because it achieved the best ROC-AUC among RF/LGBM/Stack.
from fastapi import FastAPI
import joblib
import pandas as pd

# Load artifacts
model = joblib.load("sales_win_model.pkl")
scaler = joblib.load("scaler.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")

app = FastAPI(title="Sales Opportunity Win Predictor (RF)")

# Helper to transform raw input into full feature set
def preprocess_payload(payload: dict) -> pd.DataFrame:
    raw = pd.DataFrame([payload])
    # Ensure required base fields exist
    required = ["opportunity_size", "meetings_held", "actions_completed", "days_open", "stage"]
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
        return {"error": str(e)}# Example payload for /predict
# {
#   "opportunity_size": 55000,
#   "meetings_held": 3,
#   "actions_completed": 12,
#   "days_open": 40,
#   "stage": 2
# }
# Expected response keys: model, reason, prediction, probability