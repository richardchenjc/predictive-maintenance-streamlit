# 🏭 Factory Guard: Industrial Predictive Maintenance

**A Physics-Informed Machine Learning Solution for Reducing Unplanned Downtime.**

## 🎯 The Business Problem
Unplanned equipment failure costs manufacturers billions annually. Traditional "Black Box" AI models often fail because they ignore the laws of physics, leading to false alarms and operator distrust.

## 💡 The Solution
This project bridges **Chemical Engineering** and **Data Science**:
1.  **Physics-Based Feature Engineering:** Calculated `Heat Transfer Efficiency` and `Power Output` from raw sensors.
2.  **Failure Mode Detection:** Identified a "Bimodal Risk Profile" where machines fail due to both **Overstrain** (High Power) and **Stalling** (Low Power).
3.  **Explainable AI:** Uses SHAP values to attribute risk to specific physical root causes.

## 📊 Key Insights (Phase 1)
* **The Stall Zone:** Contrary to intuition, "Very Low Power" operations showed a **2.3% failure rate**, significantly higher than medium-load operations.
* **The Cause:** Data forensics revealed this correlates with low rotational speeds, leading to **inefficient cooling** (Heat Failure).

## 🛠️ Tech Stack
* **Core:** Python, Pandas, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Workflow:** Git, VS Code

## 🚀 How to Run
1.  Clone the repository.
2.  Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3.  Run `notebooks/01_ingestion.ipynb` to clean the raw data.