"""
Streamlit dashboard for the predictive maintenance model.

Updated for the calibrated-baseline model and new feature set:
- 12 features (5 raw + 3 one-hot + 3 engineered continuous + 1 boolean)
- Threshold 0.01 from cost analysis (notebook 06)
- Loads production model from artifact (baseline or calibrated as selected by training)
"""

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Suppress routine warnings that don't affect the dashboard
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")


# ---------------------------------------------------------------------------
# Configuration and constants
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide",
)

# Cost-aware decision threshold. Selected by empirical sweep in
# notebooks/06_cost_analysis.ipynb under the 16:1 FN:FP cost asymmetry.
# Theoretical Bayes-optimal is 0.059; empirical landed at 0.01 due to
# under-prediction in the model's low-probability tail.
DECISION_THRESHOLD = 0.01

# Risk band boundaries for operator dashboard
PROB_ADVISORY = 0.01
PROB_WARNING = 0.10
PROB_CRITICAL = 0.50

# Physics conversion constant for Power_W
RPM_TO_RAD_PER_SEC = 2 * np.pi / 60  # ≈ 0.10472

# Manufacturer-supplied tool change-out threshold (placeholder)
TOOL_WEAR_RISK_THRESHOLD = 180  # minutes

# Resolve the model path relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "data" / "processed" / "predictive_maintenance_model.pkl"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: Path):
    """Load the trained model artifact. Returns None if the file is missing."""
    if not model_path.exists():
        return None
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def build_input_dataframe(
    product_type: str,
    air_temp_c: float,
    process_temp_c: float,
    speed: int,
    torque: float,
    tool_wear: int,
    feature_order: list[str],
) -> pd.DataFrame:
    """Construct a single-row DataFrame matching the trained model's feature order.

    Uses the new feature set:
    - Raw: Air_Temp, Process_Temp, Rotational_Speed, Torque, Tool_Wear
    - Categorical (one-hot): Type_L, Type_M, Type_H
    - Engineered: Temp_Delta, Power_W, Energy_Per_Wear, Tool_Wear_Risk_Zone
    """
    # Convert Celsius to Kelvin (training data was in Kelvin)
    air_temp_k = air_temp_c + 273.15
    process_temp_k = process_temp_c + 273.15

    # One-hot encoding for product type
    type_l = 1 if product_type == "L (Low)" else 0
    type_m = 1 if product_type == "M (Medium)" else 0
    type_h = 1 if product_type == "H (High)" else 0

    # Engineered features (must match recipe in 02_feature_engineering.ipynb)
    temp_delta = process_temp_k - air_temp_k
    power_w = torque * speed * RPM_TO_RAD_PER_SEC
    energy_per_wear = power_w / (tool_wear + 1)
    tool_wear_risk_zone = 1 if tool_wear > TOOL_WEAR_RISK_THRESHOLD else 0

    row = {
        "Air_Temp": air_temp_k,
        "Process_Temp": process_temp_k,
        "Rotational_Speed": speed,
        "Torque": torque,
        "Tool_Wear": tool_wear,
        "Type_L": type_l,
        "Type_M": type_m,
        "Type_H": type_h,
        "Temp_Delta": temp_delta,
        "Power_W": power_w,
        "Energy_Per_Wear": energy_per_wear,
        "Tool_Wear_Risk_Zone": tool_wear_risk_zone,
    }
    return pd.DataFrame([row])[feature_order]


# ---------------------------------------------------------------------------
# SHAP plotting
# ---------------------------------------------------------------------------
def plot_shap_waterfall(model, df_input: pd.DataFrame, feature_names: list[str]):
    """Generate a SHAP waterfall plot for the input row.

    Handles both raw LightGBM and CalibratedClassifierCV wrappers. SHAP needs
    the underlying tree model, so we unwrap calibrated classifiers if present.
    """
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV wraps a FrozenEstimator wrapping the actual model
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "estimator"):
            base = base.estimator

    explainer = shap.TreeExplainer(base)
    shap_values = explainer.shap_values(df_input)

    if isinstance(shap_values, list):
        sv_row = shap_values[1][0]
        expected_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, list)
            else explainer.expected_value
        )
    else:
        sv_row = shap_values[0]
        expected_value = explainer.expected_value

    fig, _ = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=sv_row,
            base_values=expected_value,
            data=df_input.iloc[0],
            feature_names=feature_names,
        ),
        show=False,
    )
    return fig


def risk_category(prob: float) -> tuple[str, str]:
    """Map probability to (category_label, icon)."""
    if prob >= PROB_CRITICAL:
        return "Critical", "🚨"
    if prob >= PROB_WARNING:
        return "Warning", "⚠️"
    if prob >= PROB_ADVISORY:
        return "Advisory", "👁️"
    return "Healthy", "✅"


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.title("🏭 Intelligent Factory Guard")
st.caption(
    f"Predictive maintenance with cost-aware decision threshold. "
    f"Current threshold: {DECISION_THRESHOLD:.2f} "
    "(see notebooks/06_cost_analysis.ipynb for methodology)."
)

artifact = load_model(MODEL_PATH)
if artifact is None:
    st.error(
        f"Model file not found at expected location:\n\n`{MODEL_PATH}`\n\n"
        "Run notebooks/04_model_training.ipynb to generate the model file."
    )
    st.stop()

model = artifact["model"]
feature_order = artifact["features"]
is_calibrated = artifact.get("is_calibrated", False)

# Sidebar
st.sidebar.header("🔧 Machine Settings")
product_type = st.sidebar.selectbox(
    "Product Type (Quality)", ["L (Low)", "M (Medium)", "H (High)"]
)
air_temp_c = st.sidebar.slider("Air Temperature (°C)", 15.0, 35.0, 25.0, step=0.1)
process_temp_c = st.sidebar.slider("Process Temperature (°C)", 25.0, 45.0, 35.0, step=0.1)
speed = st.sidebar.slider("Rotational Speed (RPM)", 1100, 2900, 1500, step=10)
torque = st.sidebar.slider("Torque (Nm)", 3.0, 80.0, 40.0, step=0.5)
tool_wear = st.sidebar.slider("Tool Wear (minutes)", 0, 260, 100, step=1)

with st.sidebar.expander("Model info"):
    st.write(f"Calibrated: {is_calibrated}")
    cal_metrics = artifact.get("calibration_metrics", {})
    if cal_metrics:
        baseline_brier = cal_metrics.get("brier_baseline_test")
        cv_auc = cal_metrics.get("cv_auc_calibrated_mean")
        if baseline_brier is not None:
            st.write(f"Test Brier (baseline): {baseline_brier:.4f}")
        if cv_auc is not None:
            st.write(f"CV AUC (mean): {cv_auc:.3f}")

# Main panel
if st.button("🔍 Run Diagnostics"):
    df_input = build_input_dataframe(
        product_type, air_temp_c, process_temp_c,
        speed, torque, tool_wear, feature_order,
    )

    prob = model.predict_proba(df_input)[0][1]
    category, icon = risk_category(prob)

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Diagnosis")
        if category == "Critical":
            st.error(f"{icon} {category.upper()} — Imminent failure")
            st.write("Stop operation, investigate immediately")
        elif category == "Warning":
            st.warning(f"{icon} {category.upper()} — Meaningful elevation")
            st.write("Address proactively; do not defer past current shift")
        elif category == "Advisory":
            st.info(f"{icon} {category.upper()} — Above cost-optimal cut-point")
            st.write("Schedule inspection at next opportunity")
        else:
            st.success(f"{icon} {category.upper()}")
            st.write("Operating normally; routine monitoring")

    with c2:
        st.markdown("### Risk Probability")
        st.metric(
            label="Failure Risk",
            value=f"{prob:.2%}",
            delta=f"Threshold: {DECISION_THRESHOLD:.0%}",
            delta_color="off",
        )

    with st.expander("See internal sensor readings (processed)"):
        st.dataframe(df_input.style.format("{:.2f}"))

    st.divider()
    head_col1, head_col2 = st.columns([0.8, 0.2])
    with head_col1:
        st.markdown("### 🧠 Model Reasoning")
    with head_col2:
        st.info("ℹ️ **How to read this**")

    with st.expander("What do these numbers mean?"):
        st.markdown("""
        * **E[f(x)] (base value):** Average risk across all machines.
        * **Red bars:** Features pushing risk **up**.
        * **Blue bars:** Features pushing risk **down**.
        * **f(x):** Final raw risk score for this machine.
        """)

    try:
        fig = plot_shap_waterfall(model, df_input, feature_order)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP visualisation unavailable: {e}")
        st.write("Probability output is correct; only the SHAP plot is affected.")
