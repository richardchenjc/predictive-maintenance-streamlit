import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA # For visualization

# 1. Load your processed data
# Assuming you saved this earlier. If not, load raw and apply column names.
df = pd.read_csv('C:\\Users\\Richard\\Downloads\\Semester break\\Data\\Processed\\ai4i2020_featured.csv')

# 2. SEPARATE THE DATA
# In anomaly detection, we often train on "Good" data to learn what "Normal" looks like.
normal_data = df[df['Machine_Failure'] == 0]
failure_data = df[df['Machine_Failure'] == 1]

# Select features (Sensors) - Drop labels and identifiers
features = ['Air_Temp', 'Process_Temp', 'Speed', 'Torque', 'Tool_Wear', 'Temp_Delta', 'Power_Factor']

# We will train on a portion of the NORMAL data
X_train = normal_data[features].sample(frac=0.8, random_state=42)

# We will test on the remaining normal data + ALL the failures
X_test_normal = normal_data.drop(X_train.index)[features]
X_test_failure = failure_data[features]
X_test = pd.concat([X_test_normal, X_test_failure])

# 3. TRAIN ISOLATION FOREST
# contamination='auto' allows the model to estimate the % of outliers, 
# or set it small (e.g. 0.05) if you expect few anomalies.
iso_forest = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
iso_forest.fit(X_train)

# 4. PREDICT
# Returns -1 for Anomaly, 1 for Normal
test_predictions = iso_forest.predict(X_test)

# Add predictions back to the test set for analysis
X_test['Anomaly_Pred'] = test_predictions
X_test['Actual_Label'] = 0 # Default to 0
X_test.loc[X_test_failure.index, 'Actual_Label'] = 1 # Mark actual failures

# 5. EVALUATE
# Did we catch the failures?
caught_failures = X_test[(X_test['Actual_Label'] == 1) & (X_test['Anomaly_Pred'] == -1)]
print(f"Total Failures in Test Set: {len(X_test_failure)}")
print(f"Failures Detected as Anomalies: {len(caught_failures)}")

# 6. VISUALIZE (Using PCA to squash 7D data into 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test[features])

plt.figure(figsize=(10, 6))
# Plot Normal Predictions
plt.scatter(X_pca[X_test['Anomaly_Pred'] == 1, 0], X_pca[X_test['Anomaly_Pred'] == 1, 1], 
            c='blue', label='Predicted Normal', alpha=0.5, s=10)
# Plot Anomaly Predictions
plt.scatter(X_pca[X_test['Anomaly_Pred'] == -1, 0], X_pca[X_test['Anomaly_Pred'] == -1, 1], 
            c='red', label='Predicted Anomaly', alpha=0.8, s=30)

plt.title('Isolation Forest Anomaly Detection (PCA Visualization)')
plt.legend()
plt.show()