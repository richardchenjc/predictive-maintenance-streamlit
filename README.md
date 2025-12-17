# 🏭 Intelligent Predictive Maintenance System

## 📌 Executive Summary
A Machine Learning system that predicts industrial equipment failure **before it happens**. 
Unlike standard black-box models, this project incorporates **Physics-Based Feature Engineering** (Thermodynamics & Torque Strain) to identify failure modes that raw sensor data misses.

**Impact:**
* **Recall:** 84% (Captures the vast majority of failures).
* **Precision:** 88% (Minimizes false alarms and downtime costs).
* **Explainability:** Uses SHAP values to tell operators *exactly* why a machine is at risk (e.g., "Heat Transfer Efficiency is too low").

---

## 🛠️ Tech Stack
* **Model:** LightGBM (Gradient Boosting)
* **Explainability:** SHAP (Shapley Additive Explanations)
* **Interface:** Streamlit (Real-time Interactive Dashboard)
* **Language:** Python (Pandas, NumPy, Scikit-Learn)

---

## 📊 Key Engineering Insights
We discovered that raw sensors (`Air Temp`, `Process Temp`) were insufficient. By engineering domain-specific features, we significantly improved model performance:
1.  **Temp_Delta:** MODELED heat transfer efficiency (Process - Air).
2.  **Power_W:** Calculated actual mechanical output.
3.  **The "Stall Zone":** Identified a high-risk cluster at *Low Speed / High Torque* that simple linear models missed.

---

## 🚀 How to Run locally
1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the Dashboard**
    ```bash
    streamlit run app/main.py
    ```

---

## 📂 Project Structure
```text
├── app/
│   └── main.py          # The Streamlit Dashboard
├── data/
│   └── processed/       # The serialized model (.pkl)
├── notebooks/           # Jupyter Notebooks for training & analysis
├── requirements.txt     # Dependency list
└── README.md            # Project documentation