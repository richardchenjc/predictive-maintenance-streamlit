"""
Smoke test for the trained predictive maintenance model.

Standalone sanity check. Loads the trained classifier and runs a prediction
on a manually-constructed worst-case input ('doom row') to confirm the model
produces a high failure probability when shown extreme operating conditions.

After any change to feature engineering, training, or model serialisation,
run this to confirm the model still behaves sensibly on a known-bad scenario
before pushing the changes.

Usage (from repo root):
    python app/smoke_test.py

Expected output: failure probability in the 70-99% range, depending on
specific calibration of the trained model.
"""

import numpy as np
from pathlib import Path

import joblib
import pandas as pd


# Resolve the model path relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "data" / "processed" / "predictive_maintenance_model.pkl"

# Physics conversion constant (must match feature engineering notebook)
RPM_TO_RAD_PER_SEC = 2 * np.pi / 60

# Manufacturer-supplied tool change-out threshold (must match feature engineering)
TOOL_WEAR_RISK_THRESHOLD = 180


def main() -> None:
    """Load the model and run the doom-row prediction."""
    # 1. Load the trained model artifact
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Run notebooks/04_model_training.ipynb to generate the model."
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features = artifact["features"]
    is_calibrated = artifact.get("is_calibrated", False)

    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Model is calibrated: {is_calibrated}")
    print(f"Expected feature order: {features}")
    print()

    # 2. Construct the 'doom row' — worst-case scenario across all inputs
    # Combines several known failure risk factors simultaneously:
    #   - Low-quality product (Type_L = 1)
    #   - Stall-zone operating point (low speed + high torque)
    #   - End-of-life tool (Tool_Wear at 220 minutes, near maximum)
    #   - High thermal load (Temp_Delta = 10 K)
    air_temp = 300.0
    process_temp = 310.0
    rotational_speed = 1300       # Stall zone: low rpm at high load
    torque = 70.0                 # Near-maximum torque
    tool_wear = 220               # Near end-of-life tool

    # Engineered features — formulas must match 02_feature_engineering.ipynb
    temp_delta = process_temp - air_temp                                # 10.0
    power_w = torque * rotational_speed * RPM_TO_RAD_PER_SEC             # 9527.7
    energy_per_wear = power_w / (tool_wear + 1)                          # 43.1
    tool_wear_risk_zone = 1 if tool_wear > TOOL_WEAR_RISK_THRESHOLD else 0  # 1

    doom_data = {
        "Air_Temp": air_temp,
        "Process_Temp": process_temp,
        "Rotational_Speed": rotational_speed,
        "Torque": torque,
        "Tool_Wear": tool_wear,
        "Type_L": 1,         # Low quality product
        "Type_M": 0,
        "Type_H": 0,
        "Temp_Delta": temp_delta,
        "Power_W": power_w,
        "Energy_Per_Wear": energy_per_wear,
        "Tool_Wear_Risk_Zone": tool_wear_risk_zone,
    }

    # 3. Convert to DataFrame with columns in the order the model expects
    df_doom = pd.DataFrame([doom_data])[features]

    # 4. Run prediction and report
    prob = model.predict_proba(df_doom)[0][1]

    print("-" * 60)
    print("Doom row inputs:")
    for col in features:
        print(f"  {col:25s} = {df_doom.iloc[0][col]:>10.2f}")
    print()
    print(f"Predicted failure probability: {prob:.4f} ({prob * 100:.2f}%)")
    print("-" * 60)

    # 5. Sanity assertion
    # With the new cost-aware threshold of 0.01, the assertion is mainly that
    # the doom row should be well above the threshold. We use 0.50 as a strong
    # sanity check — the model should be very confident on a row this bad.
    SANITY_THRESHOLD = 0.50
    if prob < SANITY_THRESHOLD:
        print()
        print(
            f"WARNING: Doom-row prediction ({prob:.2%}) is below the sanity "
            f"threshold ({SANITY_THRESHOLD:.0%}). The model may not be "
            f"correctly identifying extreme operating conditions as high-risk. "
            f"Investigate before proceeding."
        )


if __name__ == "__main__":
    main()
