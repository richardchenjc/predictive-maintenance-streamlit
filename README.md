# Predictive Maintenance with Cost-Aware Operating Point

A predictive maintenance system for industrial equipment built on the AI4I 2020 dataset. The system combines a supervised LightGBM classifier for known failure patterns with an unsupervised autoencoder for novel anomaly detection, and is calibrated around the cost asymmetry between missed failures and false alarms rather than around F1 score.

## What this project is and isn't

This is a portfolio project demonstrating how an engineer with operational experience thinks about deploying ML in industrial settings. It uses a public benchmark dataset (UCI AI4I 2020), not real plant data. The cost figures used to calibrate the operating point are synthesised from published industry sources (Siemens, ABB, Deloitte, BLS), with explicit acknowledgement of which numbers are cited versus constructed.

The methodology — diagnostic-first calibration, cost-aware threshold selection, dual-model architecture, operator-readable explanations — is designed to demonstrate the *shape of thinking* that makes ML usable in regulated and operational environments. It is not a deployment-ready system; specific cost figures and operational thresholds would need recalibration against any real deployment context.

## The core design decision: cost asymmetry, not F1

Predictive maintenance literature typically reports model performance using F1 score or AUC, which weight precision and recall equally. In practice this is rarely the right calibration. In manufacturing, the costs are highly asymmetric.

Published industry data on unplanned downtime costs:

- **Siemens (2024)** *True Cost of Downtime*: Fortune Global 500 manufacturers lose approximately US\$1.4 trillion annually to unplanned downtime, equivalent to 11% of total revenues. Automotive manufacturing reaches \$2.3 million per hour; heavy industry has risen to \$59 million per hour.
- **ABB (2023)** *Value of Reliability Survey*: median unplanned downtime cost across industrial sectors is approximately US\$125,000 per hour, with two-thirds of surveyed facilities experiencing downtime at least monthly.
- **Deloitte (2022)** *Predictive Maintenance and the Smart Factory*: unplanned downtime costs industrial manufacturers an estimated US\$50 billion annually. Reactive emergency repairs typically cost 3-5× the equivalent planned maintenance.

For false-alarm costs, no published per-event figure exists — false alarm cost is inherently deployment-specific. Our analysis synthesises a representative figure from BLS wage data (May 2024 median wage for industrial machinery mechanics) and the Siemens production-interruption rates. Full citations and the limitations of this synthesis are documented in `notebooks/06_cost_analysis.ipynb`.

The cost ratio of missed failures to false alarms is typically 10-20× in industrial contexts. Under this asymmetry, optimising for F1 is incorrect. The operating point should be tuned to favour recall.

## Methodology: diagnostic-first calibration

The previous iteration of this project used `scale_pos_weight=28` to handle the 28:1 class imbalance and selected the decision threshold (0.33) by visual inspection of the precision-recall curve. Both choices were directionally right but quantitatively incomplete.

The current iteration takes a different approach:

1. **Train an unweighted baseline.** No `scale_pos_weight`. Standard log-loss minimisation.
2. **Diagnose calibration before applying it.** Plot the reliability diagram on a held-out calibration set. Examine the shape of any miscalibration before choosing a calibration technique.
3. **Apply calibration only if needed.** In this case, the unweighted baseline turned out to be well-calibrated already (Brier score ≈ 0.008, AUC = 0.978). Platt scaling slightly worsened the Brier score, so we ship the baseline. This is the diagnostic-first principle: choose tools based on evidence, not convention.
4. **Derive the threshold from cost analysis.** Bayes-optimal threshold = c_FP / (c_FP + c_FN) ≈ 0.059 for our central case. Empirical sweep landed at 0.01, indicating the model slightly under-predicts in the low-probability tail. We use the empirical optimum.

The full analysis is in `notebooks/04_model_training.ipynb` (calibration diagnosis) and `notebooks/06_cost_analysis.ipynb` (threshold derivation).

## Architecture: why two models

The supervised LightGBM classifier handles the failure modes present in training data. The AI4I dataset includes Tool Wear, Heat Dissipation, Power, and Overstrain failures — all learnable from sensor data.

The unsupervised autoencoder catches failure modes the classifier hasn't been trained on. A supervised classifier is structurally unable to detect what isn't in its training data; in a real deployment, new failure modes are common. The autoencoder, trained only on healthy operation, learns the latent representation of a normal machine and flags anything that doesn't reconstruct well.

`notebooks/07_anomaly_detection_demo.ipynb` validates this architectural argument empirically: we hold out Heat Dissipation Failures (HDF) from LightGBM training, then show the autoencoder catches a meaningful fraction of HDF cases the classifier misses. The demonstration uses the AI4I-preserved failure type labels (TWF/HDF/PWF/OSF/RNF) to set up a controlled hold-out experiment.

## Physics-based feature engineering

Raw sensor readings (Air_Temp, Process_Temp, Rotational_Speed, Torque, Tool_Wear) are augmented with engineered features grounded in mechanical engineering:

**Temp_Delta** = Process_Temp − Air_Temp. The temperature gradient driving heat transfer (Fourier's law). A narrowing gradient under constant load indicates cooling system degradation.

**Power_W** = Torque × Rotational_Speed × (2π/60). Actual mechanical power in watts. The physical quantity that determines stress and heat generation; two operating points with the same Power_W produce the same load even if their individual torque and RPM differ.

**Energy_Per_Wear** = Power_W / (Tool_Wear + 1). A specific-cutting-energy proxy. As tools dull, they need more power per unit of accumulated work — this ratio rises, providing an indirect measure of tool condition. Trees cannot easily approximate ratios at split nodes, making this a non-redundant engineered feature.

**Tool_Wear_Risk_Zone** = (Tool_Wear > 180) as a boolean. Encodes the manufacturer-specified end-of-life threshold. Tool life follows Taylor's equation with a sharp cliff near end-of-life; a boolean captures this cliff structure better than the continuous variable alone. The 180-minute threshold is a placeholder; real deployment would receive this from manufacturer specification.

**Product type one-hot** (Type_L, Type_M, Type_H). The L/M/H quality grade encoded as three binary columns rather than as an ordinal scale, avoiding the unverified assumption that distance from L to M equals distance from M to H.

Risk_Heuristic from the previous iteration (Power_W × Temp_Delta) was removed. Pre-computed products of two features create multicollinearity for tree models without adding new information — trees learn interactions natively through split structure. The change improves SHAP attribution clarity in `notebooks/05_model_explanation.ipynb`.

## Operator-facing explainability

SHAP values tell an engineer which features pushed a prediction toward "failure" or "healthy." That's useful for ML practitioners but doesn't help an operator decide what to do. The translation layer in `app/diagnostic_translator.py` maps SHAP value patterns to operator-readable diagnoses with recommended actions.

Five diagnostic patterns are recognised, each matching a physical failure mechanism:

- **Heat dissipation** — high Temp_Delta SHAP + elevated raw Temp_Delta. Action: inspect cooling system.
- **Tool wear** — high Tool_Wear SHAP + raw Tool_Wear above concern threshold. Action: schedule tool replacement.
- **Stall zone** — low rotational speed + high torque. Action: review cutting parameters first (aggressive feed rate can produce this signature on healthy hardware), then investigate spindle drive if parameters are within spec.
- **Mechanical overstrain** — high Power_W + Energy_Per_Wear SHAP contributions. Action: reduce load, inspect bearings and lubricant.
- **Multi-factor** — no single dominant pattern. Action: engineering review.

Probability outputs are categorised into four risk bands aligned to the cost-optimal threshold:

| Probability | Risk category | Operator response |
|---|---|---|
| 0 to 0.01 | **Healthy** | Routine monitoring |
| 0.01 to 0.10 | **Advisory** | Log for engineer review; schedule inspection at next convenient window |
| 0.10 to 0.50 | **Warning** | Address proactively; do not defer past current shift |
| 0.50+ | **Critical** | Stop operation, investigate immediately |

The naming follows process-control conventions (DCS/SCADA systems use advisory/warning/critical hierarchies) so operators trained on industrial control systems recognise the framework.

## Dashboard

The Streamlit dashboard at `app/main.py` provides single-machine predictions with operator-readable diagnoses and SHAP drill-down. The current implementation supports interactive what-if exploration (adjust sensor readings via sliders and see how the prediction changes). A fleet view extension is planned but not yet implemented.

## Performance

Performance figures from the calibrated baseline model on the held-out test set (see `notebooks/04_model_training.ipynb`):

| Metric | Value |
|---|---|
| Test set Brier score (lower is better) | ≈ 0.008 |
| AUC-ROC (5-fold CV mean) | 0.978 ± 0.005 |
| Average precision (5-fold CV mean) | 0.87 ± 0.01 |
| Decision threshold | 0.01 |

The model output is approximately well-calibrated — when the model says probability ≈ 0.06, roughly 6% of those cases actually fail. This was diagnosed via reliability diagram on the calibration set; no post-hoc calibration was applied because the unweighted baseline was already well-calibrated.

These figures should not be compared directly against published F1-optimised benchmarks on AI4I 2020. The optimisation target is different (cost-minimisation vs F1), and the calibration goal is different (calibrated probabilities vs ranking).

## Project structure

```
predictive-maintenance-streamlit/
├── app/
│   ├── main.py                          # Streamlit dashboard
│   ├── diagnostic_translator.py         # SHAP-to-operator-language layer
│   ├── anomaly_detection.py             # Autoencoder training script
│   └── smoke_test.py                    # Standalone model sanity check
├── notebooks/
│   ├── 01_data_ingestion_and_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb     # Temp_Delta, Power_W, Energy_Per_Wear, Tool_Wear_Risk_Zone
│   ├── 03_data_visualisation.ipynb      # EDA on featured data, stall-zone investigation
│   ├── 04_model_training.ipynb          # Unweighted baseline + calibration diagnosis
│   ├── 05_model_explanation.ipynb       # SHAP analysis
│   ├── 06_cost_analysis.ipynb           # Cost-aware threshold derivation
│   └── 07_anomaly_detection_demo.ipynb  # Autoencoder OOD demonstration
├── figures/                             # Generated plots (regenerated when notebooks run)
├── data/                                # Local only, gitignored
│   ├── raw/                             # Place AI4I 2020 source CSV here
│   └── processed/                       # Cleaned and featured CSVs, model pkl
├── requirements.txt
└── README.md
```

The `data/` folder is gitignored to keep the repo small. Place the raw AI4I 2020 dataset at `data/raw/ai4i2020.csv` before running the notebooks; the pipeline will populate `data/processed/` from there.

## Running locally

```bash
git clone https://github.com/richardchenjc/predictive-maintenance-streamlit.git
cd predictive-maintenance-streamlit
pip install -r requirements.txt
streamlit run app/main.py
```

To reproduce the model from scratch, run the notebooks in numerical order. Each notebook consumes outputs from the previous and writes intermediate files to `data/processed/`.

## Dataset

UCI AI4I 2020 Predictive Maintenance Dataset. 10,000 synthetic data points modelling a milling machine, with 3.4% failure rate and five failure type labels: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), and Random Failure (RNF). The Machine_Failure target is set to 1 if any of the five failure types occurred.

Reference: Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. Third International Conference on Artificial Intelligence for Industries (AI4I).

## Tech stack

Python 3.13, LightGBM, scikit-learn (CalibratedClassifierCV with FrozenEstimator pattern), TensorFlow/Keras (autoencoder), SHAP, Streamlit, pandas, matplotlib, seaborn.

## Author

Chen Jui Chia (Richard) — MSc Data Science for Sustainability, NUS (May 2026). Previously Operations Engineer at Pfizer and Process Engineer at A*STAR ICES.
