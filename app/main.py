import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator LabelEncoder.*")
warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")

# ---------------------------------------------------------
# 1. CONFIGURATION & SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide"
)

# ---------------------------------------------------------
# 2. LOAD THE MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model_filename = "predictive_maintenance_model.pkl"
    
    # 1. Try the hardcoded path first (Fastest)
    # Adjust this if your structure is different
    current_dir = os.path.dirname(os.path.abspath(__file__))
    potential_paths = [
        os.path.join(current_dir, '..', 'data', 'processed', model_filename),
        os.path.join(current_dir, 'data', 'processed', model_filename),
        os.path.join(current_dir, model_filename)
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            return joblib.load(path)

    # 2. If that fails, SEARCH the entire directory tree (The "Search & Rescue")
    # Start looking from the current directory upwards
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    for root, dirs, files in os.walk(root_dir):
        if model_filename in files:
            found_path = os.path.join(root, model_filename)
            return joblib.load(found_path)
            
    # 3. If still not found, return None
    return None

artifact = load_model()

# ---------------------------------------------------------
# 3. SIDEBAR CONTROLS (User Inputs)
# ---------------------------------------------------------
st.title("🏭 Intelligent Factory Guard")

if artifact is None:
    st.error("❌ Error: Model file not found. Please check data/processed/predictive_maintenance_model.pkl")
    st.stop()

st.sidebar.header("🔧 Machine Settings")

# A. Product Type
type_input = st.sidebar.selectbox("Product Type (Quality)", ["L (Low)", "M (Medium)", "H (High)"])

# B. Temperature (User sees Celsius, we will convert later)
st.sidebar.markdown("### 🌡️ Temperature Conditions")
air_temp_c = st.sidebar.number_input("Air Temperature [°C]", value=25.0, step=0.5)
process_temp_c = st.sidebar.number_input("Process Temperature [°C]", value=35.0, step=0.5)

# C. Operating Parameters (Switched to Number Input for Precision)
st.sidebar.markdown("### ⚙️ Operating Parameters")
# step=1 ensures we send integers where appropriate
speed = st.sidebar.number_input("Rotational Speed [rpm]", min_value=1100, max_value=2900, value=1500, step=10)
torque = st.sidebar.number_input("Torque [Nm]", min_value=0.0, max_value=80.0, value=40.0, step=1.0)
tool_wear = st.sidebar.number_input("Tool Wear [min]", min_value=0, max_value=250, value=0, step=5)

# ---------------------------------------------------------
# 4. PHYSICS ENGINE (Hidden Logic)
# ---------------------------------------------------------
# A. Unit Conversion (Celsius -> Kelvin)
# The model was trained on Kelvin, so we MUST convert before prediction.
air_temp_k = air_temp_c + 273.15
process_temp_k = process_temp_c + 273.15

# B. Map Type
type_map = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 2}
type_encoded = type_map[type_input]

# C. Calculate Physics Features
# 1. Delta Temp (Note: Delta is the same in C or K, but good to be consistent)
temp_delta = process_temp_k - air_temp_k

# 2. Power Output (Watts)
power_w = torque * speed * 0.1047

# 3. Risk Heuristic
risk_heuristic = int((power_w < 200) | (power_w > 2500))

# D. Create the Feature Vector
# We map the calculated variables to the feature names expected by the model
input_data = {
    'Air_Temp': air_temp_k,
    'Process_Temp': process_temp_k,
    'Rotational_Speed': speed,
    'Torque': torque,
    'Tool_Wear': tool_wear,
    'Type_Encoded': type_encoded,
    'Temp_Delta': temp_delta,
    'Power_W': power_w,
    'Risk_Heuristic': risk_heuristic
}

# Create DataFrame and Reorder Columns to match Training
df_input = pd.DataFrame([input_data])
df_input = df_input[artifact['features']]

# ---------------------------------------------------------
# 5. PREDICTION & EXPLANATION
# ---------------------------------------------------------
if st.button("🔍 Run Diagnostics"):
    
    # A. Get Probability
    prob = artifact['model'].predict_proba(df_input)[0][1]
    prediction = int(prob > 0.33) # Threshold 0.33
    
    st.divider()
    c1, c2 = st.columns(2)
    
    # B. Display Output
    with c1:
        st.markdown("### Diagnosis")
        if prediction == 1:
            st.error(f"🚨 FAILURE PREDICTED")
            st.write("Immediate Maintenance Required")
        else:
            st.success(f"✅ SYSTEM HEALTHY")
            st.write("Operations Normal")
            
    with c2:
        st.markdown("### Risk Probability")
        # Color logic for the metric
        delta_color = "inverse" if prediction == 1 else "normal"
        st.metric(label="Failure Risk", value=f"{prob:.1%}", delta=f"{prob*100:.1f}% Risk", delta_color=delta_color)
        
    with st.expander("See Internal Sensor Readings (Processed)"):
        # Show the user the converted Kelvin values so they trust the math
        st.dataframe(df_input.style.format("{:.2f}"))

    # C. SHAP Explanation
    st.divider()
    
    # Create two columns: Title on left, Help button on right
    head_col1, head_col2 = st.columns([0.8, 0.2])
    with head_col1:
        st.markdown("### 🧠 Model Reasoning")
    with head_col2:
        # A help tooltip for the user
        st.info("ℹ️ **How to read this?**")
        
    # Add an Expander to explain the math if they need it
    with st.expander("What do these numbers mean?"):
        st.markdown("""
        * **E[f(x)] (Base Value):** The average risk score across all machines in history. This is the starting point.
        * **Red Bars:** Features pushing the risk **UP** (Bad News).
        * **Blue Bars:** Features pushing the risk **DOWN** (Good News).
        * **f(x) (Final Score):** The final raw risk score for *this* machine.
        """)

    model = artifact['model']
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)
    
    # Handle SHAP version differences
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
        
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(
        shap.Explanation(values=sv[0], 
                         base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                         data=df_input.iloc[0],
                         feature_names=artifact['features']),
        show=False
    )
    st.pyplot(fig)

