"""
Streamlit dashboard for the predictive maintenance model.

This is the cleanup version: removes the os.walk fallback for model loading,
uses script-relative paths, and tightens a few small issues. The fleet view
rebuild is a separate piece of work (Day 4) and will replace this single-
machine view as the dashboard's primary mode.
"""

import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Suppress the routine warnings that don't affect the dashboard
warnings.filterwarnings(
    "ignore",
    message=".*Trying to unpickle estimator LabelEncoder.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*LightGBM binary classifier.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*FigureCanvasAgg is non-interactive.*",
)


# ---------------------------------------------------------------------------
# 1. Configuration and constants
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide",
)

# Decision threshold. Selected via cost-aware threshold analysis in
# notebooks/05_cost_analysis.ipynb. Update this value if the cost-optimal
# threshold changes after rerunning the analysis with new cost parameters.
DECISION_THRESHOLD = 0.33

# Physics conversion constant for Power_W = Torque * RPM * RPM_TO_RAD_PER_SEC
RPM_TO_RAD_PER_SEC = 0.1047

# Resolve the model path relative to this script's location, so the dashboard
# works whether launched from repo root or from inside app/
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "data" / "processed" / "predictive_maintenance_model.pkl"


# ---------------------------------------------------------------------------
# 2. Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: Path):
    """Load the trained model artifact. Returns None if the file is missing."""
    if not model_path.exists():
        return None
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# 3. Input handling
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
    """Construct a single-row DataFrame matching the trained model's feature order."""
    # Convert celsius to kelvin (the model was trained on kelvin values)
    air_temp_k = air_temp_c + 273.15
    process_temp_k = process_temp_c + 273.15

    # Engineered features (must match the recipe in 02_feature_engineering.ipynb)
    temp_delta = process_temp_k - air_temp_k
    power_w = torque * speed * RPM_TO_RAD_PER_SEC
    risk_heuristic = power_w * temp_delta

    type_map = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 2}
    type_encoded = type_map[product_type]

    row = {
        "Air_Temp": air_temp_k,
        "Process_Temp": process_temp_k,
        "Rotational_Speed": speed,
        "Torque": torque,
        "Tool_Wear": tool_wear,
        "Type_Encoded": type_encoded,
        "Temp_Delta": temp_delta,
        "Power_W": power_w,
        "Risk_Heuristic": risk_heuristic,
    }
    # Return columns in the order the model expects
    return pd.DataFrame([row])[feature_order]


# ---------------------------------------------------------------------------
# 4. SHAP plotting
# ---------------------------------------------------------------------------
def plot_shap_waterfall(model, df_input: pd.DataFrame, feature_names: list[str]):
    """Generate a SHAP waterfall plot for the input row. Returns matplotlib figure."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)

    # SHAP returns either a list (older versions) or a single array (newer versions)
    # for binary classification. Handle both.
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


# ---------------------------------------------------------------------------
# 5. App layout
# ---------------------------------------------------------------------------
st.title("🏭 Intelligent Factory Guard")
st.caption(
    "Predictive maintenance with cost-aware decision threshold. "
    f"Current threshold: {DECISION_THRESHOLD:.2f} "
    "(see notebooks/05_cost_analysis.ipynb for methodology)."
)

artifact = load_model(MODEL_PATH)

if artifact is None:
    st.error(
        f"Model file not found at expected location:\n\n`{MODEL_PATH}`\n\n"
        "Run notebooks/03_model_training.ipynb to generate the model file."
    )
    st.stop()

model = artifact["model"]
feature_order = artifact["features"]


# ---------- Sidebar: machine input ----------
st.sidebar.header("🔧 Machine Settings")

product_type = st.sidebar.selectbox(
    "Product Type (Quality)",
    ["L (Low)", "M (Medium)", "H (High)"],
)

air_temp_c = st.sidebar.slider(
    "Air Temperature (°C)",
    min_value=15.0, max_value=35.0, value=25.0, step=0.1,
)
process_temp_c = st.sidebar.slider(
    "Process Temperature (°C)",
    min_value=25.0, max_value=45.0, value=35.0, step=0.1,
)
speed = st.sidebar.slider(
    "Rotational Speed (RPM)",
    min_value=1100, max_value=2900, value=1500, step=10,
)
torque = st.sidebar.slider(
    "Torque (Nm)",
    min_value=3.0, max_value=80.0, value=40.0, step=0.5,
)
tool_wear = st.sidebar.slider(
    "Tool Wear (minutes)",
    min_value=0, max_value=260, value=100, step=1,
)


# ---------- Main: prediction ----------
if st.button("🔍 Run Diagnostics"):
    df_input = build_input_dataframe(
        product_type=product_type,
        air_temp_c=air_temp_c,
        process_temp_c=process_temp_c,
        speed=speed,
        torque=torque,
        tool_wear=tool_wear,
        feature_order=feature_order,
    )

    prob = model.predict_proba(df_input)[0][1]
    prediction = int(prob > DECISION_THRESHOLD)

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Diagnosis")
        if prediction == 1:
            st.error("🚨 FAILURE PREDICTED")
            st.write("Immediate maintenance required")
        else:
            st.success("✅ SYSTEM HEALTHY")
            st.write("Operations normal")

    with c2:
        st.markdown("### Risk Probability")
        delta_color = "inverse" if prediction == 1 else "normal"
        st.metric(
            label="Failure Risk",
            value=f"{prob:.1%}",
            delta=f"{prob * 100:.1f}% risk",
            delta_color=delta_color,
        )

    with st.expander("See internal sensor readings (processed)"):
        st.dataframe(df_input.style.format("{:.2f}"))

    # ---------- SHAP explanation ----------
    st.divider()

    head_col1, head_col2 = st.columns([0.8, 0.2])
    with head_col1:
        st.markdown("### 🧠 Model Reasoning")
    with head_col2:
        st.info("ℹ️ **How to read this**")

    with st.expander("What do these numbers mean?"):
        st.markdown(
            """
            * **E[f(x)] (base value):** The average risk score across all machines.
              This is the model's starting point before considering this specific machine.
            * **Red bars:** Features pushing the risk **up** (bad news).
            * **Blue bars:** Features pushing the risk **down** (good news).
            * **f(x) (final score):** The final raw risk score for this machine.
            """
        )

    fig = plot_shap_waterfall(model, df_input, feature_order)
    st.pyplot(fig)
