"""
train_model.py
================
Trains an XGBoost classifier to predict Hospital Patient Risk
(High Risk vs. Low Risk) from a simulated clinical dataset.

Produces the following artifacts (saved with joblib) for use by app.py:
    - model.pkl        : trained XGBClassifier
    - scaler.pkl        : fitted StandardScaler (for numeric features)
    - explainer.pkl     : fitted SHAP TreeExplainer
    - feature_names.pkl : ordered list of feature names used by the model
    - metrics.pkl       : dict of evaluation metrics + ROC curve + confusion matrix
    - X_test.pkl / y_test.pkl : held-out test set (used by the dashboard's
                                  "Model Performance" tab)

Run:
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)
from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 1. SYNTHETIC PATIENT DATASET
# ---------------------------------------------------------------------------
def generate_patient_data(n_samples: int = 4000) -> pd.DataFrame:
    """
    Simulates a realistic hospital patient dataset with vitals and labs.
    The binary target `high_risk` is generated from a clinically-inspired
    logistic function of the underlying features (plus noise), so the
    model has genuine, learnable signal rather than pure randomness.
    """
    age = np.random.normal(55, 17, n_samples).clip(18, 95)
    bmi = np.random.normal(27, 5.5, n_samples).clip(15, 55)
    glucose = np.random.normal(110, 32, n_samples).clip(60, 300)
    systolic_bp = np.random.normal(128, 18, n_samples).clip(80, 220)
    diastolic_bp = np.random.normal(82, 12, n_samples).clip(50, 140)
    heart_rate = np.random.normal(78, 13, n_samples).clip(40, 160)
    resp_rate = np.random.normal(17, 3.5, n_samples).clip(8, 40)
    spo2 = np.random.normal(96.5, 2.7, n_samples).clip(70, 100)
    cholesterol = np.random.normal(195, 40, n_samples).clip(100, 400)
    creatinine = np.random.normal(1.0, 0.45, n_samples).clip(0.3, 6.0)
    smoker = np.random.binomial(1, 0.22, n_samples)
    diabetes = np.random.binomial(1, 0.18, n_samples)
    prior_admission = np.random.binomial(1, 0.15, n_samples)

    # Clinically-motivated linear combination -> logistic probability
    logit = (
        -15.6
        + 0.045 * age
        + 0.055 * bmi
        + 0.020 * glucose
        + 0.028 * systolic_bp
        + 0.018 * diastolic_bp
        + 0.022 * heart_rate
        - 0.9 * (spo2 - 90) / 5.0
        + 0.010 * cholesterol
        + 1.15 * creatinine
        + 0.9 * smoker
        + 1.1 * diabetes
        + 1.3 * prior_admission
        - 0.05 * resp_rate
    )
    prob = 1 / (1 + np.exp(-logit))
    high_risk = np.random.binomial(1, prob)

    df = pd.DataFrame(
        {
            "age": age.round(1),
            "bmi": bmi.round(1),
            "glucose": glucose.round(1),
            "systolic_bp": systolic_bp.round(1),
            "diastolic_bp": diastolic_bp.round(1),
            "heart_rate": heart_rate.round(1),
            "resp_rate": resp_rate.round(1),
            "spo2": spo2.round(1),
            "cholesterol": cholesterol.round(1),
            "creatinine": creatinine.round(2),
            "smoker": smoker,
            "diabetes": diabetes,
            "prior_admission": prior_admission,
            "high_risk": high_risk,
        }
    )
    return df


FEATURE_NAMES = [
    "age",
    "bmi",
    "glucose",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "resp_rate",
    "spo2",
    "cholesterol",
    "creatinine",
    "smoker",
    "diabetes",
    "prior_admission",
]

# Features that get standardized (continuous vitals/labs).
# Binary flags (smoker, diabetes, prior_admission) are left as-is.
CONTINUOUS_FEATURES = [
    "age",
    "bmi",
    "glucose",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "resp_rate",
    "spo2",
    "cholesterol",
    "creatinine",
]


def main():
    print("=" * 60)
    print("Generating synthetic hospital patient dataset...")
    df = generate_patient_data(n_samples=4000)
    print(f"Dataset shape: {df.shape}")
    print(f"Class balance:\n{df['high_risk'].value_counts(normalize=True)}\n")

    X = df[FEATURE_NAMES].copy()
    y = df["high_risk"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # ------------------------------------------------------------------
    # 2. SCALE CONTINUOUS FEATURES
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[CONTINUOUS_FEATURES] = scaler.fit_transform(X_train[CONTINUOUS_FEATURES])
    X_test_scaled[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES])

    # ------------------------------------------------------------------
    # 3. TRAIN XGBOOST CLASSIFIER
    # ------------------------------------------------------------------
    print("Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 4. EVALUATE
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 5. SHAP EXPLAINER
    # ------------------------------------------------------------------
    print("\nBuilding SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    # ------------------------------------------------------------------
    # 6. SAVE ARTIFACTS
    # ------------------------------------------------------------------
    print("Saving artifacts to disk...")
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(explainer, "explainer.pkl")
    joblib.dump(FEATURE_NAMES, "feature_names.pkl")
    joblib.dump(CONTINUOUS_FEATURES, "continuous_features.pkl")

    metrics = {
        "roc_auc": roc_auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
    }
    joblib.dump(metrics, "metrics.pkl")
    joblib.dump(X_test_scaled, "X_test.pkl")
    joblib.dump(y_test, "y_test.pkl")

    print("\nAll artifacts saved successfully:")
    for f in [
        "model.pkl",
        "scaler.pkl",
        "explainer.pkl",
        "feature_names.pkl",
        "continuous_features.pkl",
        "metrics.pkl",
        "X_test.pkl",
        "y_test.pkl",
    ]:
        print(f"  - {f}")
    print("\nDone! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
