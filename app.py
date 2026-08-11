"""
app.py
=======
Hospital Patient Risk Prediction System
A clinical-themed, multi-tab Streamlit dashboard powered by an XGBoost
classifier and SHAP explainability.

Run:
    streamlit run app.py

Requires artifacts produced by train_model.py (model.pkl, scaler.pkl,
explainer.pkl, feature_names.pkl, continuous_features.pkl, metrics.pkl,
X_test.pkl, y_test.pkl) to be present in the same directory.
"""

import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import auc as sk_auc

# ---------------------------------------------------------------------------
# PAGE CONFIG & CLINICAL THEME
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hospital Patient Risk Prediction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0B5D8C"      # clinical blue
DANGER = "#C0392B"       # high risk red
SAFE = "#1E8449"         # low risk green
WARN = "#E67E22"         # moderate amber
BG_CARD = "#F4F8FB"

CUSTOM_CSS = f"""
<style>
    .main {{
        background-color: #FAFCFE;
    }}
    h1, h2, h3 {{
        color: {PRIMARY};
        font-family: 'Segoe UI', sans-serif;
    }}
    .metric-card {{
        background-color: {BG_CARD};
        border-left: 5px solid {PRIMARY};
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }}
    .risk-banner-high {{
        background-color: #FDEDEC;
        border: 2px solid {DANGER};
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        color: {DANGER};
        font-size: 22px;
        font-weight: 700;
    }}
    .risk-banner-low {{
        background-color: #EAFAF1;
        border: 2px solid {SAFE};
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        color: {SAFE};
        font-size: 22px;
        font-weight: 700;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {BG_CARD};
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-weight: 600;
    }}
    footer {{visibility: hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LOAD ARTIFACTS (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        explainer = joblib.load("explainer.pkl")
        feature_names = joblib.load("feature_names.pkl")
        continuous_features = joblib.load("continuous_features.pkl")
        metrics = joblib.load("metrics.pkl")
        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")
        return (
            model,
            scaler,
            explainer,
            feature_names,
            continuous_features,
            metrics,
            X_test,
            y_test,
        )
    except FileNotFoundError:
        return None


artifacts = load_artifacts()

if artifacts is None:
    st.error(
        " Model artifacts not found. Please run `python train_model.py` "
        "first to train and save the model, then relaunch this app."
    )
    st.stop()

(
    model,
    scaler,
    explainer,
    FEATURE_NAMES,
    CONTINUOUS_FEATURES,
    metrics,
    X_test,
    y_test,
) = artifacts


# ---------------------------------------------------------------------------
# SESSION STATE (patient history log)
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=[
            "Timestamp",
            "Patient ID",
            "Age",
            "BMI",
            "Glucose",
            "Systolic BP",
            "Heart Rate",
            "SpO2",
            "Risk Probability (%)",
            "Risk Category",
        ]
    )

if "last_input" not in st.session_state:
    st.session_state.last_input = None
if "last_proba" not in st.session_state:
    st.session_state.last_proba = None


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <h1> Hospital Patient Risk Prediction System</h1>
    <p style="color:#555; font-size:16px;">
    Clinical decision-support dashboard powered by <b>XGBoost</b> and
    <b>SHAP</b> explainable AI. For research / educational demonstration
    purposes only — not a substitute for professional medical judgment.
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# SIDEBAR — PATIENT INPUT FORM
# ---------------------------------------------------------------------------
st.sidebar.header(" Patient Vitals Input")
st.sidebar.caption("Enter the patient's current clinical measurements.")

with st.sidebar.form("patient_form"):
    patient_id = st.text_input("Patient ID", value=f"PT-{np.random.randint(1000, 9999)}")

    st.markdown("**Demographics**")
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=55, step=1)
    bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=55.0, value=27.0, step=0.1)

    st.markdown("**Vitals**")
    systolic_bp = st.slider("Systolic BP (mmHg)", 80, 220, 128)
    diastolic_bp = st.slider("Diastolic BP (mmHg)", 50, 140, 82)
    heart_rate = st.slider("Heart Rate (bpm)", 40, 160, 78)
    resp_rate = st.slider("Respiratory Rate (breaths/min)", 8, 40, 17)
    spo2 = st.slider("SpO2 (%)", 70, 100, 97)

    st.markdown("**Labs**")
    glucose = st.number_input("Glucose (mg/dL)", min_value=60.0, max_value=300.0, value=110.0, step=1.0)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=195.0, step=1.0)
    creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.3, max_value=6.0, value=1.0, step=0.05)

    st.markdown("**Risk Factors**")
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    prior_admission = st.selectbox("Prior Hospital Admission (12mo)", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Calculate Risk Score", use_container_width=True)


def build_input_frame():
    raw = {
        "age": age,
        "bmi": bmi,
        "glucose": glucose,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "resp_rate": resp_rate,
        "spo2": spo2,
        "cholesterol": cholesterol,
        "creatinine": creatinine,
        "smoker": 1 if smoker == "Yes" else 0,
        "diabetes": 1 if diabetes == "Yes" else 0,
        "prior_admission": 1 if prior_admission == "Yes" else 0,
    }
    df_raw = pd.DataFrame([raw])[FEATURE_NAMES]
    df_scaled = df_raw.copy()
    df_scaled[CONTINUOUS_FEATURES] = scaler.transform(df_raw[CONTINUOUS_FEATURES])
    return df_raw, df_scaled


if submitted:
    df_raw, df_scaled = build_input_frame()
    proba = float(model.predict_proba(df_scaled)[0, 1])

    st.session_state.last_input = df_scaled
    st.session_state.last_input_raw = df_raw
    st.session_state.last_proba = proba
    st.session_state.last_patient_id = patient_id

    risk_label = "High Risk" if proba >= 0.5 else "Low Risk"
    new_row = {
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Patient ID": patient_id,
        "Age": age,
        "BMI": bmi,
        "Glucose": glucose,
        "Systolic BP": systolic_bp,
        "Heart Rate": heart_rate,
        "SpO2": spo2,
        "Risk Probability (%)": round(proba * 100, 2),
        "Risk Category": risk_label,
    }
    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([new_row])], ignore_index=True
    )


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [" Risk Assessment", " Explainable AI (SHAP)", " Patient History Log", " Model Performance"]
)


# ===========================================================================
# TAB 1: RISK ASSESSMENT
# ===========================================================================
with tab1:
    st.subheader("Patient Risk Assessment")

    if st.session_state.last_proba is None:
        st.info(
            " Fill out the patient vitals in the sidebar and click "
            "**Calculate Risk Score** to generate a risk assessment."
        )
    else:
        proba = st.session_state.last_proba
        pct = proba * 100
        risk_label = "HIGH RISK" if proba >= 0.5 else "LOW RISK"

        col1, col2 = st.columns([1, 1.3])

        with col1:
            banner_class = "risk-banner-high" if proba >= 0.5 else "risk-banner-low"
            st.markdown(
                f"""
                <div class="{banner_class}">
                    Patient {st.session_state.last_patient_id}: {risk_label}<br>
                    <span style="font-size:34px;">{pct:.1f}%</span> probability of high risk
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.markdown("**Input Summary**")
            st.dataframe(
                st.session_state.last_input_raw.T.rename(columns={0: "Value"}),
                use_container_width=True,
            )

        with col2:
            gauge_color = DANGER if proba >= 0.5 else SAFE
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=pct,
                    number={"suffix": "%", "font": {"size": 40}},
                    title={"text": "Risk Probability Gauge", "font": {"size": 20}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 30], "color": "#EAFAF1"},
                            {"range": [30, 50], "color": "#FEF9E7"},
                            {"range": [50, 75], "color": "#FDEBD0"},
                            {"range": [75, 100], "color": "#FDEDEC"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.8,
                            "value": 50,
                        },
                    },
                )
            )
            fig.update_layout(height=380, margin=dict(t=60, b=20, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Clinical Recommendation**")
        if proba >= 0.75:
            st.error(
                "🔴 **Critical Priority** — Immediate clinical review recommended. "
                "Consider close monitoring, specialist consultation, and expedited diagnostics."
            )
        elif proba >= 0.5:
            st.warning(
                "🟠 **Elevated Risk** — Schedule a follow-up assessment and review "
                "modifiable risk factors (glucose control, blood pressure management)."
            )
        elif proba >= 0.3:
            st.info(
                "🟡 **Moderate/Watch** — Continue routine monitoring; reassess if "
                "symptoms or vitals change."
            )
        else:
            st.success("🟢 **Low Risk** — No immediate action indicated; maintain routine care.")


# ===========================================================================
# TAB 2: EXPLAINABLE AI (SHAP)
# ===========================================================================
with tab2:
    st.subheader("Explainable AI — Why This Prediction?")

    if st.session_state.last_input is None:
        st.info(
            " Run a risk assessment first (sidebar) to view the SHAP "
            "explanation for that specific patient."
        )
    else:
        df_scaled = st.session_state.last_input
        df_raw = st.session_state.last_input_raw

        shap_values = explainer(df_scaled)

        st.markdown(
            "The **SHAP waterfall plot** below shows how each feature pushed "
            "this specific patient's prediction away from the model's baseline "
            "(average) risk. Red bars increase predicted risk; blue bars decrease it."
        )

        # Build a readable waterfall using Plotly (label with raw clinical values)
        sv = shap_values.values[0]
        base_value = shap_values.base_values[0]
        feat_labels = [
            f"{name} = {df_raw.iloc[0][name]}" for name in FEATURE_NAMES
        ]

        order = np.argsort(np.abs(sv))[::-1]
        sv_sorted = sv[order]
        labels_sorted = [feat_labels[i] for i in order]

        cumulative = base_value
        y_labels = ["Base value"] + labels_sorted + ["Final prediction"]
        measures = ["absolute"] + ["relative"] * len(sv_sorted) + ["total"]
        values = [base_value] + list(sv_sorted) + [0]

        waterfall_fig = go.Figure(
            go.Waterfall(
                orientation="h",
                measure=measures,
                y=y_labels,
                x=values,
                connector={"line": {"color": "rgba(120,120,120,0.4)"}},
                decreasing={"marker": {"color": SAFE}},
                increasing={"marker": {"color": DANGER}},
                totals={"marker": {"color": PRIMARY}},
            )
        )
        waterfall_fig.update_layout(
            title="SHAP Waterfall — Feature Contribution to Risk Score (log-odds)",
            height=500,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(waterfall_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Top Contributing Factors**")
        top_n = 5
        contrib_df = pd.DataFrame(
            {
                "Feature": [feat_labels[i] for i in order[:top_n]],
                "SHAP Impact (log-odds)": [round(float(sv[i]), 4) for i in order[:top_n]],
                "Direction": [
                    "⬆ Increases Risk" if sv[i] > 0 else "⬇ Decreases Risk"
                    for i in order[:top_n]
                ],
            }
        )
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

        with st.expander("ℹ How to read this"):
            st.write(
                "SHAP (SHapley Additive exPlanations) values quantify each "
                "feature's contribution to the difference between the model's "
                "average prediction (base value) and the prediction for this "
                "specific patient, in log-odds units. Larger absolute values "
                "indicate stronger influence on the outcome."
            )


# ===========================================================================
# TAB 3: PATIENT HISTORY LOG
# ===========================================================================
with tab3:
    st.subheader("Session Patient History Log")

    if st.session_state.history.empty:
        st.info("No patients assessed yet in this session. Submit the form to begin logging.")
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Assessed", len(st.session_state.history))
        col_b.metric(
            "High Risk Count",
            int((st.session_state.history["Risk Category"] == "High Risk").sum()),
        )
        col_c.metric(
            "Avg. Risk Probability",
            f"{st.session_state.history['Risk Probability (%)'].mean():.1f}%",
        )

        def highlight_risk(row):
            color = "#FDEDEC" if row["Risk Category"] == "High Risk" else "#EAFAF1"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            st.session_state.history.style.apply(highlight_risk, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        csv = st.session_state.history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download History as CSV",
            data=csv,
            file_name="patient_history_log.csv",
            mime="text/csv",
        )

        if st.button(" Clear History Log"):
            st.session_state.history = st.session_state.history.iloc[0:0]
            st.rerun()


# ===========================================================================
# TAB 4: MODEL PERFORMANCE
# ===========================================================================
with tab4:
    st.subheader("Global Model Performance")
    st.caption("Evaluated on a held-out test set (20% split, stratified).")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(
        f'<div class="metric-card"><b>ROC-AUC</b><br><span style="font-size:22px;">{metrics["roc_auc"]:.3f}</span></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-card"><b>Recall</b><br><span style="font-size:22px;">{metrics["recall"]:.3f}</span></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-card"><b>Precision</b><br><span style="font-size:22px;">{metrics["precision"]:.3f}</span></div>',
        unsafe_allow_html=True,
    )
    m4.markdown(
        f'<div class="metric-card"><b>F1-Score</b><br><span style="font-size:22px;">{metrics["f1"]:.3f}</span></div>',
        unsafe_allow_html=True,
    )
    m5.markdown(
        f'<div class="metric-card"><b>Accuracy</b><br><span style="font-size:22px;">{metrics["accuracy"]:.3f}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        cm = metrics["confusion_matrix"]
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Low Risk", "High Risk"],
            y=["Low Risk", "High Risk"],
        )
        cm_fig.update_layout(height=420, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(cm_fig, use_container_width=True)

    with col2:
        st.markdown("**ROC Curve**")
        fpr, tpr = metrics["fpr"], metrics["tpr"]
        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"XGBoost (AUC = {metrics['roc_auc']:.3f})",
                line=dict(color=PRIMARY, width=3),
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="gray", width=2, dash="dash"),
            )
        )
        roc_fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=420,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(x=0.4, y=0.1),
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Global Feature Importance (XGBoost Gain)**")
    importance = model.feature_importances_
    imp_df = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": importance}).sort_values(
        "Importance", ascending=True
    )
    imp_fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    imp_fig.update_layout(height=450, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(imp_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    " Hospital Patient Risk Prediction System — Demo build using synthetic data. "
    "Not intended for real clinical use. Built with Streamlit, XGBoost & SHAP."
)
