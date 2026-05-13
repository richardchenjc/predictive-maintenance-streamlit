"""
Smoke test for the trained predictive maintenance model.

This script is a standalone sanity check. It loads the trained LightGBM
classifier and runs a prediction on a manually-constructed worst-case input
('doom row') to confirm the model still produces a high failure probability
when shown extreme operating conditions.

The intended use is during development: after any change to feature engineering,
training, or model serialisation, run this script to confirm the model still
behaves sensibly on a known-bad scenario before pushing the changes. It is
NOT a replacement for the test-set evaluation in 04_model_explanation.ipynb,
which evaluates performance on unseen real data.

Usage (from repo root):
    python app/smoke_test.py

Expected output: failure probability in the 80-99% range, depending on the
specific calibration of the trained model.
"""

import os
from pathlib import Path

import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# Resolve the model path relative to this script's location, so the test runs
# whether invoked from the repo root or from inside the app/ directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "data" / "processed" / "predictive_maintenance_model.pkl"


def main() -> None:
    """Load the model and run the doom-row prediction."""
    # -----------------------------------------------------------------------
    # 1. Load the trained model artifact
    # -----------------------------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Run notebooks/04_model_training.ipynb to generate the model."
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features = artifact["features"]

    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Expected feature order: {features}")

    # -----------------------------------------------------------------------
    # 2. Construct the 'doom row' - a worst-case scenario across all inputs
    # -----------------------------------------------------------------------
    # The doom row combines several known failure risk factors simultaneously:
    #   - Low-quality product (Type_Encoded = 0)
    #   - Stall-zone operating point (low speed + high torque)
    #   - End-of-life tool (Tool_Wear at 220 minutes, near maximum)
    #   - High thermal load (Temp_Delta = 10K)
    #
    # All engineered features are calculated using the SAME formulas as the
    # feature engineering pipeline in 02_feature_engineering.ipynb. Mismatches
    # here would feed the model inconsistent inputs and invalidate the test.
    air_temp = 300.0
    process_temp = 310.0
    rotational_speed = 1300       # Stall zone: low rpm at high load
    torque = 70.0                 # Near-maximum torque
    tool_wear = 220               # Near end-of-life tool

    # Engineered features (must match formulas in feature engineering notebook)
    temp_delta = process_temp - air_temp                    # 10.0
    power_w = torque * rotational_speed * 0.1047            # 9527.7
    risk_heuristic = power_w * temp_delta                   # 95277 (Power_W * Temp_Delta)

    doom_data = {
        "Air_Temp": air_temp,
        "Process_Temp": process_temp,
        "Rotational_Speed": rotational_speed,
        "Torque": torque,
        "Tool_Wear": tool_wear,
        "Type_Encoded": 0,         # Low quality product
        "Temp_Delta": temp_delta,
        "Power_W": power_w,
        "Risk_Heuristic": risk_heuristic,
    }

    # -----------------------------------------------------------------------
    # 3. Convert to DataFrame with columns in the order the model expects
    # -----------------------------------------------------------------------
    df_doom = pd.DataFrame([doom_data])[features]

    # -----------------------------------------------------------------------
    # 4. Run prediction and report
    # -----------------------------------------------------------------------
    prob = model.predict_proba(df_doom)[0][1]

    print()
    print("-" * 60)
    print(f"Doom row inputs:")
    for col in features:
        print(f"  {col:20s} = {df_doom.iloc[0][col]:>10.2f}")
    print()
    print(f"Predicted failure probability: {prob:.4f} ({prob * 100:.2f}%)")
    print("-" * 60)

    # -----------------------------------------------------------------------
    # 5. Sanity assertion - flag if prediction is unexpectedly low
    # -----------------------------------------------------------------------
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
