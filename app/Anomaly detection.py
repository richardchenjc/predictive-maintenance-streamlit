import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# 1. LOAD DATA
# ---------------------------------------------------------
print("Loading Data...")
try:
    df = pd.read_csv('processed_data.csv')
except FileNotFoundError:
    print("Error: processed_data.csv not found. Please run your data prep script first.")
    exit()

# 2. DATA SPLITTING & PREPROCESSING
# ---------------------------------------------------------
# Anomaly Detection Strategy: Train ONLY on "Normal" data.
# The model learns to reconstruct "Normal". It will fail to reconstruct "Anomalies".

normal_data = df[df['Machine_Failure'] == 0]
failure_data = df[df['Machine_Failure'] == 1]

# Select Physics-Based Features
# Note: Neural Nets are sensitive to scale, so we MUST scale data.
features = ['Air_Temp', 'Process_Temp', 'Speed', 'Torque', 'Tool_Wear', 'Temp_Delta', 'Power_Factor']

# Split Normal Data: 80% for Training, 20% for Testing
# We shuffle to ensure randomness
X_train_normal = normal_data[features].sample(frac=0.8, random_state=42)
X_test_normal = normal_data.drop(X_train_normal.index)[features]

# The Test set includes the unseen Normal data AND all the Failures
X_test_failure = failure_data[features]
X_test = pd.concat([X_test_normal, X_test_failure])

# SCALING (Critical for Autoencoders)
# We fit the scaler ONLY on training data to avoid data leakage
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)

print(f"Training on {len(X_train_scaled)} normal samples.")
print(f"Testing on {len(X_test_scaled)} samples (containing {len(X_test_failure)} failures).")

# 3. BUILD THE AUTOENCODER MODEL
# ---------------------------------------------------------
# Concept: Compress 7 features down to 4 (Encoder), then expand back to 7 (Decoder)
input_dim = X_train_scaled.shape[1] # 7 features
encoding_dim = 4 

input_layer = Input(shape=(input_dim,))

# Encoder: Compression
encoder = Dense(encoding_dim, activation="relu")(input_layer)

# Decoder: Reconstruction
# We use 'sigmoid' activation because our data is scaled 0-1
decoder = Dense(input_dim, activation="sigmoid")(encoder)

# Combine into Model
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile using Mean Squared Error (MSE) to measure reconstruction quality
autoencoder.compile(optimizer='adam', loss='mse')

# 4. TRAIN THE MODEL
# ---------------------------------------------------------
print("Training Autoencoder...")
history = autoencoder.fit(
    X_train_scaled, 
    X_train_scaled,  # Target is the Input itself (Reconstruction)
    epochs=50, 
    batch_size=16, 
    shuffle=True, 
    validation_split=0.1,
    verbose=0 # Set to 1 if you want to see training logs
)
print("Training Complete.")

# 5. DETECT ANOMALIES
# ---------------------------------------------------------
# We calculate the "Reconstruction Error" (MSE) for the Test set
reconstructions = autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)

# DEFINE THRESHOLD
# A common technique: Anything with error higher than 95% of the Training Data is an anomaly
# In a real app, this threshold is a tunable parameter.
threshold = np.percentile(mse, 95) 
print(f"Anomaly Threshold set at: {threshold:.4f}")

# Flag Anomalies
test_predictions = mse > threshold # True if Anomaly

# 6. EVALUATION
# ---------------------------------------------------------
# Create a DataFrame to analyze results
results = X_test.copy()
results['Reconstruction_Error'] = mse
results['Predicted_Anomaly'] = test_predictions
results['Actual_Failure'] = 0
results.loc[X_test_failure.index, 'Actual_Failure'] = 1

# Check if we caught the failures
caught_failures = results[(results['Actual_Failure'] == 1) & (results['Predicted_Anomaly'] == True)]

print("-" * 30)
print(f"Total Actual Failures: {len(X_test_failure)}")
print(f"Failures Detected by Autoencoder: {len(caught_failures)}")
print("-" * 30)

# 7. VISUALIZATION
# ---------------------------------------------------------
# Plot 1: Reconstruction Error Histogram
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(mse, bins=50, alpha=0.7, color='blue', label='Reconstruction Error')
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Mean Squared Error (MSE)')
plt.legend()

# Plot 2: PCA Visualization of Anomalies
# Squash 7D data to 2D to visualize the "Cluster"
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_scaled)

plt.subplot(1, 2, 2)
# Plot Normal Predictions
normal_indices = results['Predicted_Anomaly'] == False
plt.scatter(X_pca[normal_indices, 0], X_pca[normal_indices, 1], 
            c='blue', label='Predicted Normal', alpha=0.3, s=10)

# Plot Anomaly Predictions
anomaly_indices = results['Predicted_Anomaly'] == True
plt.scatter(X_pca[anomaly_indices, 0], X_pca[anomaly_indices, 1], 
            c='red', label='Predicted Anomaly', alpha=0.8, s=20)

plt.title('Autoencoder Detection (PCA View)')
plt.legend()

plt.tight_layout()
plt.show()