# 🏥 Hospital Patient Risk Prediction System

A production-ready, modular Streamlit application that predicts patient risk
(High Risk vs. Low Risk) using an **XGBoost** classifier trained on simulated
clinical vitals and labs, with **SHAP**-powered explainability and interactive
**Plotly** visualizations.

## Project Structure

```
hospital_risk_app/
├── train_model.py       # Generates data, trains XGBoost, saves model + SHAP explainer
├── app.py                # Streamlit multi-tab clinical dashboard
├── requirements.txt      # Python dependencies
└── README.md
```

After running `train_model.py`, the following artifacts are created in the
same directory (used by `app.py`):

```
model.pkl                # Trained XGBClassifier
scaler.pkl                # StandardScaler for continuous features
explainer.pkl              # SHAP TreeExplainer
feature_names.pkl          # Ordered feature list
continuous_features.pkl    # Which features were scaled
metrics.pkl                 # ROC-AUC, recall, precision, confusion matrix, ROC curve
X_test.pkl / y_test.pkl     # Held-out test set
```

## How to Run Locally

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This prints evaluation metrics to the console and saves all model artifacts
(`.pkl` files) into the project directory.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in
your browser.

## Using the App

1. **Sidebar (Patient Vitals Input)** — Enter demographics, vitals, labs, and
   risk factors, then click **Calculate Risk Score**.
2. **Tab 1 – Risk Assessment** — View the risk banner, probability gauge, and
   a clinical recommendation tier.
3. **Tab 2 – Explainable AI (SHAP)** — See a SHAP waterfall plot explaining
   exactly which features pushed this patient's score up or down, plus a
   ranked table of top contributing factors.
4. **Tab 3 – Patient History Log** — Every assessment made during the session
   is logged in a table (downloadable as CSV), with summary stats.
5. **Tab 4 – Model Performance** — Global ROC-AUC, recall, precision, F1,
   accuracy, confusion matrix, ROC curve, and global feature importance.

## Notes

- The dataset is **synthetically generated** (`train_model.py`) using a
  clinically-inspired logistic function over age, BMI, glucose, blood
  pressure, heart rate, respiratory rate, SpO2, cholesterol, creatinine,
  smoking status, diabetes, and prior admission — so the model has genuine,
  learnable signal rather than pure noise.
- This project is a **demonstration / educational tool** only and is **not**
  validated for real clinical decision-making.
- To retrain with different data or hyperparameters, edit
  `generate_patient_data()` or the `XGBClassifier(...)` parameters in
  `train_model.py`, then re-run it — the app will automatically pick up the
  new artifacts on next launch.
