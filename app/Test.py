import joblib
import pandas as pd

# 1. Load the Brain directly
artifact = joblib.load('data/processed/predictive_maintenance_model.pkl')
model = artifact['model']
features = artifact['features']

print(f"Model Expects Features: {features}")

# 2. Create the "Doom" Row (Max Torque, Max Wear, Stall Speed)
# We manually calculate the physics to be sure
doom_data = {
    'Air_Temp': 300.0,
    'Process_Temp': 310.0,
    'Rotational_Speed': 1300,  # STALL SPEED
    'Torque': 70.0,            # MAX TORQUE
    'Tool_Wear': 220,          # BROKEN TOOL
    'Type_Encoded': 0,         # Low Quality
    'Temp_Delta': 10.0,        # (310 - 300)
    'Power_W': 9527.0,         # (70 * 1300 * 0.1047) -> HIGH POWER
    'Risk_Heuristic': 1        # Power > 2500 -> High Risk
}

# 3. Convert to DataFrame and Enforce Order
df_doom = pd.DataFrame([doom_data])
df_doom = df_doom[features]  # Sort columns exactly like training

# 4. Ask the Model
prob = model.predict_proba(df_doom)[0][1]
print(f"\n------------------------------------------------")
print(f"doom_row Risk Probability: {prob:.4f} ({prob*100:.2f}%)")
print(f"------------------------------------------------")