"""
Autoencoder training script for unsupervised anomaly detection.

This is a standalone script for training the autoencoder on healthy data only.
It mirrors the methodology used in notebooks/07_anomaly_detection_demo.ipynb,
but without the HDF hold-out experiment - this trains a 'production-style'
autoencoder using all healthy data, intended to be deployed alongside the
LightGBM classifier in app/main.py.

Updates from the previous version:
- New feature set (9 AE features matching notebook 07): adds Energy_Per_Wear
  and Tool_Wear_Risk_Zone, drops the typo'd 'Speed' and 'Power_Factor' names
- StandardScaler instead of MinMaxScaler (more appropriate for tree/AE pairing)
- 9-5-9 architecture instead of 7-4-7 to match the new feature count
- Threshold set on a held-out validation set (not training set) to avoid
  the autoencoder having artificially low MSE on memorised rows
- Linear output activation instead of sigmoid (input is StandardScaler-normalised,
  not bounded to [0,1])

Usage (from repo root):
    python app/anomaly_detection.py
"""

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "data" / "processed" / "ai4i2020_featured.csv"
MODEL_OUT_PATH = SCRIPT_DIR.parent / "data" / "processed" / "autoencoder_model.pkl"

# Feature set for the autoencoder. We exclude one-hot Type columns because
# reconstruction of categorical indicators isn't meaningful in MSE terms;
# tree-side production model handles those.
AE_FEATURES = [
    "Air_Temp", "Process_Temp", "Rotational_Speed", "Torque", "Tool_Wear",
    "Temp_Delta", "Power_W", "Energy_Per_Wear", "Tool_Wear_Risk_Zone",
]


def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("Loading data...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Featured dataset not found at {DATA_PATH}. "
            f"Run notebooks/02_feature_engineering.ipynb first."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows.")

    # -----------------------------------------------------------------------
    # 2. Split: healthy data for AE training, all data for evaluation
    # -----------------------------------------------------------------------
    # AE strategy: train ONLY on healthy data, so the model learns what
    # normal looks like and fails to reconstruct anomalous patterns.
    normal_data = df[df["Machine_Failure"] == 0]
    failure_data = df[df["Machine_Failure"] == 1]

    # Split healthy data: 70% training, 15% validation (threshold setting), 15% test
    X_normal = normal_data[AE_FEATURES].values.astype(float)
    X_train, X_temp = train_test_split(
        X_normal, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test_normal = train_test_split(
        X_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    X_test_failure = failure_data[AE_FEATURES].values.astype(float)

    print(f"  AE training (healthy only):    {len(X_train):,} rows")
    print(f"  AE validation (threshold set): {len(X_val):,} rows")
    print(f"  Test - healthy:                {len(X_test_normal):,} rows")
    print(f"  Test - failures:               {len(X_test_failure):,} rows")

    # -----------------------------------------------------------------------
    # 3. Scale features (StandardScaler - fit on training only)
    # -----------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_normal_scaled = scaler.transform(X_test_normal)
    X_test_failure_scaled = scaler.transform(X_test_failure)

    # -----------------------------------------------------------------------
    # 4. Build autoencoder: 9 -> 5 -> 9 with linear output
    # -----------------------------------------------------------------------
    input_dim = X_train_scaled.shape[1]
    latent_dim = 5

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(latent_dim, activation="relu")(input_layer)
    # Linear output: StandardScaler outputs aren't bounded to [0,1] so
    # sigmoid would be wrong.
    decoded = Dense(input_dim, activation="linear")(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    print(f"\nAutoencoder architecture: {input_dim} -> {latent_dim} -> {input_dim}")

    # -----------------------------------------------------------------------
    # 5. Train on healthy data only
    # -----------------------------------------------------------------------
    print("\nTraining autoencoder...")
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=50,
        batch_size=64,
        validation_data=(X_val_scaled, X_val_scaled),
        shuffle=True,
        verbose=0,
    )
    print(f"  Final training loss:   {history.history['loss'][-1]:.4f}")
    print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")

    # -----------------------------------------------------------------------
    # 6. Set threshold on VALIDATION set (not training set)
    # -----------------------------------------------------------------------
    # Setting threshold on training data underestimates reconstruction error
    # the AE will have on truly unseen healthy operation. Using validation set
    # gives a more honest threshold.
    val_recon = autoencoder.predict(X_val_scaled, verbose=0)
    val_mse = np.mean((X_val_scaled - val_recon) ** 2, axis=1)
    threshold = float(np.percentile(val_mse, 95))

    print(f"\nAnomaly threshold (95th percentile of validation MSE): {threshold:.4f}")

    # -----------------------------------------------------------------------
    # 7. Evaluate on test set (held-out healthy + all failures)
    # -----------------------------------------------------------------------
    test_normal_recon = autoencoder.predict(X_test_normal_scaled, verbose=0)
    test_normal_mse = np.mean((X_test_normal_scaled - test_normal_recon) ** 2, axis=1)

    test_failure_recon = autoencoder.predict(X_test_failure_scaled, verbose=0)
    test_failure_mse = np.mean((X_test_failure_scaled - test_failure_recon) ** 2, axis=1)

    n_failures_caught = (test_failure_mse > threshold).sum()
    n_normal_flagged = (test_normal_mse > threshold).sum()

    print("\n" + "-" * 60)
    print("Evaluation:")
    print(f"  Failures caught: {n_failures_caught}/{len(test_failure_mse)} ({n_failures_caught/len(test_failure_mse):.1%})")
    print(f"  False alarms on healthy test data: {n_normal_flagged}/{len(test_normal_mse)} ({n_normal_flagged/len(test_normal_mse):.1%})")
    print("-" * 60)

    # -----------------------------------------------------------------------
    # 8. Save the trained autoencoder + scaler + threshold
    # -----------------------------------------------------------------------
    artifact = {
        "autoencoder": autoencoder,
        "scaler": scaler,
        "threshold": threshold,
        "features": AE_FEATURES,
    }
    joblib.dump(artifact, MODEL_OUT_PATH)
    print(f"\nSaved autoencoder artifact to {MODEL_OUT_PATH}")

    # -----------------------------------------------------------------------
    # 9. Diagnostic plots
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: reconstruction error distributions
    ax = axes[0]
    ax.hist(test_normal_mse, bins=40, alpha=0.6, color="steelblue",
            label=f"Healthy test (n={len(test_normal_mse)})", edgecolor="black")
    ax.hist(test_failure_mse, bins=40, alpha=0.6, color="darkred",
            label=f"Failures (n={len(test_failure_mse)})", edgecolor="black")
    ax.axvline(threshold, color="black", linestyle="--",
               label=f"Threshold ({threshold:.3f})")
    ax.set_xlabel("Reconstruction error (MSE)")
    ax.set_ylabel("Count")
    ax.set_title("Reconstruction error: healthy vs failure")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: PCA view of test set, coloured by AE flag
    ax = axes[1]
    X_test_all = np.vstack([X_test_normal_scaled, X_test_failure_scaled])
    test_mse_all = np.concatenate([test_normal_mse, test_failure_mse])
    test_flag_all = test_mse_all > threshold

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test_all)

    ax.scatter(X_pca[~test_flag_all, 0], X_pca[~test_flag_all, 1],
               c="steelblue", alpha=0.3, s=10, label="AE: Normal")
    ax.scatter(X_pca[test_flag_all, 0], X_pca[test_flag_all, 1],
               c="darkred", alpha=0.7, s=20, label="AE: Anomaly")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Autoencoder detection (PCA view)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = SCRIPT_DIR.parent / "figures" / "12_autoencoder_standalone.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"Saved diagnostic plot to {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
