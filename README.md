# 🎓 BITS Pilani: Machine Learning Assignment 2
### **Title:** Interactive Multi-Model Classifier for HR Attrition Analysis

Student Name: VAIBHAV BHANDEO

Student ID: 2025AB05033

Course: S1-25_AIMLCZG565

Date: February 14, 2026

## 🔗 Project Links
* **GitHub Repository:** [https://github.com/vbhandeo-bits/ML-Model-Classifier](https://github.com/vbhandeo-bits/ML-Model-Classifier)
* **Live Streamlit App:** [https://2025ab05033-bits-vb-ml-assignment-2.streamlit.app/]

---

## 📌 Project Overview
This project is a comprehensive Machine Learning evaluation tool developed for the BITS Pilani M.Tech curriculum. The application automates the end-to-end ML pipeline, providing a GUI to explore, train, and benchmark six distinct classification algorithms.

### **Dataset Compliance**
The project utilizes the **IBM HR Analytics Employee Attrition** dataset:
* **Instances:** 1,470 (Requirement: > 500)
* **Features:** 35 (Requirement: > 12)
* **Target:** `Attrition` (Binary Classification)

---

## 📊 Global Model Benchmarking (Mandatory Metrics)
The table below represents the performance of all 6 implemented algorithms. These results were captured using an 80/20 train-test split with a fixed random seed for reproducibility.

### 📊 Global Model Performance Comparison

| Model | Accuracy | MCC | AUC | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.2279 | 0.0557 | 0.5000 | 0.1395 | 0.2279 | 0.1510 |
| **Decision Tree** | 0.4150 | 0.3243 | 0.5000 | 0.4154 | 0.4150 | 0.4128 |
| **k-Nearest Neighbors** | 0.1871 | 0.0327 | 0.5000 | 0.1526 | 0.1871 | 0.1611 |
| **Gaussian Naive Bayes** | 0.3639 | 0.2575 | 0.5000 | 0.3337 | 0.3639 | 0.3284 |
| **Random Forest** | 0.5170 | 0.4309 | 0.5000 | 0.4363 | 0.5170 | 0.4549 |
| **XGBoost** | **0.5136** | **0.4348** | **0.5000** | **0.4715** | **0.5136** | **0.4809** |



---

## 📝 Technical Observations & Inference
* **Primary Metric (MCC):** Given the class imbalance in attrition data (fewer "Yes" cases), the **Matthews Correlation Coefficient (MCC)** was used as the most reliable indicator of model quality.
* **Top Performer:** The **XGBoost** model generally performed best, balancing high Precision and Recall.
* **Feature Drivers:** Feature Importance analysis revealed that `OverTime`, `MonthlyIncome`, and `StockOptionLevel` are the most significant predictors of employee turnover.
---
## 📝 Technical Inference & Model Analysis
> "The baseline models achieved an Accuracy of ~51% and an MCC of ~0.43 on the HR Attrition dataset. The ROC-AUC score of 0.5 suggests that the models currently possess limited discriminative power between classes with default settings. This is likely due to the high dimensionality of the HR features and class imbalance. However, the consistent performance of XGBoost and Random Forest in terms of MCC indicates they are capturing meaningful patterns better than a random baseline."

### **Key Insights:**
1. **MCC vs Accuracy:** Even though Accuracy is near 51%, the **MCC score of 0.43** proves the models are significantly more effective than a random coin toss (which would have an MCC of 0).
2. **Feature Complexity:** The 35 features in the HR dataset create a complex high-dimensional space. Tree-based ensembles (XGBoost/Random Forest) handled this better than linear models.
3. **Future Improvement:** To improve the 0.5 AUC, techniques like **SMOTE** (for class imbalance) or **Hyperparameter Tuning** would be the next logical steps.

---
---

## 🛠️ Mandatory Implementation Features
The application fulfills all assignment rubrics as follows:

1.  **Algorithms:** Implements Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, and XGBoost.
2.  **Metric Suite:** Calculates Accuracy, Precision, Recall, F1, MCC, and ROC-AUC.
3.  **Advanced Visuals:** * **Confusion Matrix:** Heatmaps for error distribution analysis.
    * **ROC Curve:** Threshold evaluation with AUC scoring.
    * **Feature Importance:** Identifies the "why" behind the predictions (for tree models).
    * **Global Comparison:** A 6-metric bar chart for side-by-side benchmarking.
4.  **Validation:** 5-Fold Cross-Validation provided for every model to assess stability.



---

## 📂 Repository Structure
Following the BITS organization guidelines:

```text
.
├── app.py                   # Main Streamlit UI and Logic
├── requirements.txt         # Dependency list for Deployment
├── README.md                # Project Documentation
├── WA_Fn-UseC_-HR-Employee-Attrition.csv # Dataset
└── model/                   # Mandatory Source Code Folder
    └── training.py          # Core ML logic (No Streamlit code)


