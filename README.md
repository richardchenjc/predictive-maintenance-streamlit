# Predictive Maintenance with Cost-Aware Operating Point

A predictive maintenance system for industrial equipment built on the AI4I 2020 dataset. The system combines a supervised LightGBM classifier for known failure patterns with an unsupervised autoencoder for novel anomaly detection, and is calibrated around the cost asymmetry between missed failures and false alarms rather than around F1 score.

## What this project is and isn't

This is a portfolio project demonstrating how an engineer with operational experience thinks about deploying ML in industrial settings. It uses a public benchmark dataset (UCI AI4I 2020), not real plant data, and the cost figures used to calibrate the operating point are taken from published literature, not from any specific deployment. The methodology — cost-aware threshold selection, dual-model architecture, operator-readable explanations — is designed to demonstrate the *shape of thinking* that makes ML usable in regulated and operational environments, rather than to claim production-ready performance.

## The core design decision: cost asymmetry, not F1

Predictive maintenance literature typically reports model performance using F1 score or AUC, which weight precision and recall equally. In practice this is rarely the right calibration. In manufacturing, the costs are highly asymmetric: a missed failure leads to unplanned downtime, secondary damage, and lost production, while a false alarm leads to an unnecessary inspection that takes minutes.

Published industrial data on unplanned downtime costs:
- ARC Advisory Group estimates unplanned downtime costs Fortune Global 500 manufacturers approximately US\$1.4 trillion per year, or roughly 11% of annual revenue
- Senseye's True Cost of Downtime report (2023) puts the average cost at \$129,000-\$260,000 per hour for heavy industry
- The cost of a false alarm investigation at the operator level is typically 15-30 minutes of inspection time, with no production impact if the inspection happens during scheduled downtime

The cost ratio between a missed failure and a false alarm is therefore conservatively 10-20× and often higher. Under this asymmetry, optimising for F1 is incorrect. The operating point should be tuned to favour recall, accepting more false alarms as the price of catching more real failures.

This project implements that calibration explicitly. The LightGBM classifier uses `scale_pos_weight=28` (matching the negative-to-positive class ratio in the data) during training, and the decision threshold is tuned to 0.33 rather than the default 0.5. The cost analysis section of the notebook walks through how this threshold is selected, and how it would shift under different cost ratios.

## Architecture: why two models

The supervised LightGBM classifier handles the known failure modes well — the AI4I dataset includes Tool Wear, Heat Dissipation, Power, and Overstrain failures, all of which have learnable patterns in the sensor data. The model achieves 84% recall and 88% precision on these patterns at the tuned threshold.

The unsupervised autoencoder exists to catch the failure modes that aren't in the training data. In a real deployment, equipment can fail in ways the historical data hasn't seen — new failure modes, edge cases, the long tail of unusual operating conditions. A supervised classifier is structurally unable to detect what it hasn't been trained on. The autoencoder, trained only on healthy operation data, learns the latent representation of a normal machine and flags anything that doesn't reconstruct well.

The demonstration of the autoencoder's value is in `Notebooks/anomaly_detection_demo.ipynb`. The supervised model is trained with one failure type held out from the training data, and we show that the autoencoder still catches the held-out type in the test set, demonstrating that it provides genuine coverage beyond what the supervised model can offer.

## Physics-based feature engineering

Raw sensor readings (air temperature, process temperature, speed, torque, tool wear) are augmented with three engineered features that encode physical relationships:

**Temp_Delta** = Process_Temp - Air_Temp. This is the temperature differential across the machine, which directly indicates heat transfer efficiency. A rising Temp_Delta with constant speed and torque means heat is not being dissipated effectively, which precedes thermal failure.

**Power_W** = Torque × Rotational_Speed × 0.1047. The factor 0.1047 converts RPM to radians per second, so Power_W is the actual mechanical power in watts. The dataset provides torque and speed separately, but mechanical stress and energy consumption are functions of their product, not their individual values.

**Risk_Heuristic** = Power_W × Temp_Delta. A composite score capturing the interaction effect. When both power output and thermal load are high simultaneously, failure risk multiplies rather than adds. The product captures this multiplicative effect that a linear combination cannot.

These three features together encode the physics of the operating envelope. The model gives them substantial weight in its predictions, as the SHAP analysis in `Notebooks/model_explanation.ipynb` shows.

## Operator-facing explainability

SHAP values tell an engineer which features pushed the prediction toward "failure" or "healthy." That's useful for ML practitioners, but it doesn't help the operator on the floor decide what to do. The translation layer in `app/diagnostic_translator.py` maps SHAP value patterns to operator-readable diagnoses with recommended actions. For example, a high Temp_Delta contribution combined with a low Power_W contribution produces "Heat dissipation pattern detected; cooling system check recommended" rather than "feature_5 high, feature_6 low."

The translation templates are calibrated against the physical meaning of each feature combination, not against real maintenance records. In a production deployment they would need to be tuned against historical maintenance interventions and operator feedback. The current implementation demonstrates the design pattern and shows what the deployment-grade version would look like.

## Fleet dashboard

The Streamlit dashboard at `app/main.py` shows the current risk status of a simulated fleet of machines, with each machine's individual diagnosis available as a drill-down. The fleet view is the operator's home page — at-a-glance, which machines need attention. Clicking through to a single machine shows the diagnostic translation as the primary output, with the underlying SHAP analysis available as an engineering view for ML practitioners and reliability engineers.

## Performance

Performance figures at the tuned threshold on the held-out test set:

| Metric | Value |
|---|---|
| Recall (sensitivity to failures) | 0.84 |
| Precision (correctness of failure alarms) | 0.88 |
| F1 | 0.86 |
| Decision threshold | 0.33 |

These figures should not be compared directly against published benchmarks on the AI4I dataset that optimise for F1, because the optimisation target is different. A model that achieves higher F1 by reducing recall would be worse under this cost calibration, not better.

## Project structure

```
predictive-maintenance-streamlit/
├── app/
│   ├── main.py                       # Streamlit fleet dashboard
│   ├── diagnostic_translator.py      # SHAP-to-operator-language layer
│   ├── anomaly_detection.py          # Autoencoder training script
│   └── smoke_test.py                 # Standalone model sanity check
├── notebooks/
│   ├── 01_data_ingestion_and_cleaning.ipynb   # Load and clean AI4I 2020
│   ├── 02_feature_engineering.ipynb           # Temp_Delta, Power_W, Risk_Heuristic
│   ├── 03_data_visualisation.ipynb            # EDA on engineered features, stall-zone analysis
│   ├── 04_model_training.ipynb                # LightGBM with cost-aware threshold
│   ├── 05_model_explanation.ipynb             # SHAP analysis
│   ├── 06_cost_analysis.ipynb                 # Threshold selection under cost asymmetry
│   └── 07_anomaly_detection_demo.ipynb        # Autoencoder OOD demonstration
├── figures/                          # Generated plots (regenerated when notebooks run)
│   ├── 01_correlation_drivers.png
│   ├── 02_distribution_shift.png
│   └── 03_risk_profile.png
├── data/                             # Local only, gitignored
│   ├── raw/                          # AI4I 2020 source CSV
│   └── processed/                    # Cleaned and featured CSVs, model pkl
├── requirements.txt
└── README.md
```

## Running locally

```bash
git clone https://github.com/richardchenjc/predictive-maintenance.git
cd predictive-maintenance
pip install -r requirements.txt
streamlit run app/main.py
```

To reproduce the model from scratch, run the notebooks in numerical order. The pipeline expects the raw AI4I 2020 dataset at `data/raw/ai4i2020.csv`, available from the UCI Machine Learning Repository.

## Dataset

UCI AI4I 2020 Predictive Maintenance Dataset. 10,000 synthetic data points modelling a milling machine, with 3.4% failure rate and five failure type labels: Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, and Random Failure. The Machine_Failure target is set to 1 if any of the five failure types occurred.

Reference: Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. Third International Conference on Artificial Intelligence for Industries (AI4I).

## Tech stack

Python 3.13, LightGBM 4.6, scikit-learn, TensorFlow/Keras (autoencoder), SHAP, Streamlit.

## Author

Chen Jui Chia (Richard) — MSc Data Science for Sustainability, NUS (May 2026). Previously Operations Engineer at Pfizer and Process Engineer at A*STAR ICES.
