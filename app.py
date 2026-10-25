
import streamlit as st
import sys 
import pandas as pd
from pathlib import Path

# =========================
# Project paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import predict_heart_disease
import inspect
REPORTS_PATH = PROJECT_ROOT / "reports"

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Heart Disease | ML Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Custom styling
# =========================
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #142C3A, #173B40);
        border: 1px solid #28545A;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #F2F7F8;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #C4D9DC;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-card {
        padding: 1rem 1.2rem;
        border: 1px solid #303944;
        border-radius: 12px;
        background-color: #151B24;
        margin-bottom: 1rem;
    }

    div[data-testid="stForm"] {
        border: 1px solid #303944;
        border-radius: 14px;
        padding: 1.4rem;
        background-color: #111720;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        background-color: #36B7A5;
        color: #071A1A;
        border: none;
        border-radius: 9px;
        font-weight: 700;
        min-height: 2.8rem;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #62D3C0;
        color: #071A1A;
    }

    div[data-testid="stMetric"] {
        background-color: #151B24;
        border: 1px solid #303944;
        padding: 1rem;
        border-radius: 12px;
    }

    h2, h3 {
        color: #E7F0F2;
    }

    .footer {
        text-align: center;
        color: #8C9AA8;
        font-size: 0.85rem;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Classification</h1>
    <p>
        Machine Learning dashboard powered by Random Forest.
        Explore patient features, generate predictions, and
        understand the model's analysis.
    </p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational project only. Predictions are not medical diagnoses "
    "and must not be used for clinical decisions."
)

prediction_tab, analysis_tab = st.tabs([
    "🔍  Prediction",
    "📊  Model Analysis"
])

# =================================================
# TAB 1: PREDICTION
# =================================================
with prediction_tab:

    st.header("Patient Information")
    st.caption(
        "Enter the feature values used by the trained model. "
        "Hover over the ⓘ icons for explanations."
    )

    with st.form("prediction_form"):

        # Section 1
        st.markdown(
            '<div class="section-card"><h3>👤 Basic Information</h3>'
            '<p>General patient characteristics.</p></div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=50,
                help="Patient age in years."
            )

        with col2:
            sex = st.selectbox(
                "Sex",
                options=[1, 0],
                format_func=lambda x:
                    "Male (1)" if x == 1 else "Female (0)",
                help="Dataset encoding: 1 = male, 0 = female."
            )

        st.divider()

        # Section 2
        st.markdown(
            '<div class="section-card"><h3>🩺 Clinical Measurements</h3>'
            '<p>Blood pressure, cholesterol, and heart rate.</p></div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=50,
                max_value=250,
                value=120,
                help="Resting blood pressure, measured in mm Hg."
            )

            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=50,
                max_value=700,
                value=200,
                help="Serum cholesterol level, measured in mg/dl."
            )

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                options=[0, 1],
                format_func=lambda x:
                    "No (0)" if x == 0 else "Yes (1)",
                help="Dataset indicator for fasting blood sugar above 120 mg/dl."
            )

        with col2:
            thalach = st.number_input(
                "Maximum Heart Rate Achieved",
                min_value=50,
                max_value=250,
                value=150,
                help="Maximum heart rate recorded during the exercise test, in beats per minute."
            )

            exang = st.selectbox(
                "Exercise-Induced Angina",
                options=[0, 1],
                format_func=lambda x:
                    "No (0)" if x == 0 else "Yes (1)",
                help="Whether exercise-induced angina was recorded: 1 = yes, 0 = no."
            )

        st.divider()

        # Section 3
        st.markdown(
            '<div class="section-card"><h3>💓 Heart Test Features</h3>'
            '<p>Clinical test results encoded as numeric values.</p></div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            cp = st.selectbox(
                "Chest Pain Type (cp)",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1 — Typical angina",
                    2: "2 — Atypical angina",
                    3: "3 — Non-anginal pain",
                    4: "4 — Asymptomatic"
                }[x],
                help="Chest pain category. These are the original Cleveland dataset codes."
            )

            restecg = st.selectbox(
                "Resting ECG (restecg)",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0 — Normal",
                    1: "1 — ST-T abnormality",
                    2: "2 — Left ventricular hypertrophy"
                }[x],
                help="Resting electrocardiogram category using dataset codes."
            )

            oldpeak = st.number_input(
                "ST Depression (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="ST depression induced by exercise relative to rest."
            )

        with col2:
            slope = st.selectbox(
                "Peak Exercise ST Slope (slope)",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "1 — Upsloping",
                    2: "2 — Flat",
                    3: "3 — Downsloping"
                }[x],
                help="Slope category of the peak-exercise ST segment."
            )

            ca = st.selectbox(
                "Major Vessels (ca)",
                options=[0, 1, 2, 3],
                help="Number of major vessels colored by fluoroscopy, encoded from 0 to 3."
            )

            thal = st.selectbox(
                "Thalassemia (thal)",
                options=[3, 6, 7],
                format_func=lambda x: {
                    3: "3 — Normal",
                    6: "6 — Fixed defect",
                    7: "7 — Reversible defect"
                }[x],
                help="Original Cleveland dataset category codes for thal."
            )

        submitted = st.form_submit_button(
            "🔍  Generate Prediction",
            use_container_width=True
        )

    
    if submitted:
        patient_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        try:
            result = predict_heart_disease(patient_data)
           
            st.subheader("Prediction Results")

            if result["prediction"] == 1:
                st.error("⚠️ " + result["label"])
            else:
                st.success("✓ " + result["label"])

            probabilities = result.get("probabilities")

            if probabilities:
                st.markdown("### Model Output Probabilities")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "No Disease (Class 0)",
                        f'{probabilities.get(0, 0) * 100:.1f}%'
                    )

                with col2:
                    st.metric(
                        "Disease (Class 1)",
                        f'{probabilities.get(1, 0) * 100:.1f}%'
                    )

                chart_data = pd.DataFrame({
                    "Class": ["No Disease", "Disease"],
                    "Model Output (%)": [
                        probabilities.get(0, 0) * 100,
                        probabilities.get(1, 0) * 100
                    ]
                }).set_index("Class")

                st.bar_chart(chart_data)

            else:
                st.warning(
                    "The model did not return probabilities."
                )

            summary = pd.DataFrame({
                "Feature": list(patient_data.keys()),
                "Entered Value": list(patient_data.values())
            })

            st.markdown("### Patient Input Summary")
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "This is an educational demonstration, not a medical "
                "diagnosis. Model output probabilities are not "
                "confirmed medical risks."
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

# =================================================
# TAB 2: MODEL ANALYSIS
# =================================================

with analysis_tab:

    st.header("Model Evaluation & Analysis")

    st.write(
        "Explore the Random Forest model's performance "
        "and the visualizations generated during development."
    )

    # Model performance metrics from the held-out test set
    st.subheader("📊 Random Forest Performance")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Accuracy", "91.8%")
    metric2.metric("Precision", "87.1%")
    metric3.metric("Recall", "96.4%")
    metric4.metric("F1-score", "91.5%")

    st.caption(
        "These metrics are from the previously reported test-set evaluation. "
        "They describe overall model performance, not an individual patient's result."
    )

    st.divider()

    # Visual comparison
    st.subheader("📈 Model Comparison")

    comparison_path = REPORTS_PATH / "model_comparison.png"

    if comparison_path.exists():
        st.image(
            str(comparison_path),
            caption="Comparison of model evaluation metrics",
            use_container_width=True
        )
    else:
        st.warning("Model comparison image was not found.")

    st.divider()

    # Feature importance
    st.subheader("🎯 Feature Importance")

    importance_path = REPORTS_PATH / "feature_importance.png"

    if importance_path.exists():
        st.image(
            str(importance_path),
            caption="Feature importance from the Random Forest model",
            use_container_width=True
        )
    else:
        st.warning("Feature importance image was not found.")

    st.info(
        "Feature importance indicates how useful features were to the model. "
        "It does not establish that a feature causes heart disease."
    )

    st.divider()

    # Cross-validation results
    st.subheader("🔄 Cross-Validation Results")

    cv_data = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Mean": [0.810, 0.815, 0.765, 0.785],
        "Standard Deviation": [0.017, 0.043, 0.074, 0.028]
    })

    st.dataframe(
        cv_data.style.format({
            "Mean": "{:.3f}",
            "Standard Deviation": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    cv_chart = cv_data.set_index("Metric")[["Mean"]]

    st.bar_chart(cv_chart)

    st.warning(
        "The cross-validation scores differ noticeably from the reported "
        "test-set scores. Before presenting this project as a validated model, "
        "check the evaluation pipeline, data splits, and whether the same "
        "preprocessing and metrics were used in both evaluations."
    )

# =========================
# Footer
# =========================
st.divider()

st.markdown("""
<div class="footer">
    Heart Disease Classification | Machine Learning Portfolio Project
    <br>
    Built with Python, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)
