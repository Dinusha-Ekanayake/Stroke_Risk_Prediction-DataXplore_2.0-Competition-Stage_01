# 🧠 Stroke Risk Prediction - DataXplore 2.0 Stage 01

<p align="center">
  <img src="assets/banner.png" alt="Stroke Risk Prediction Banner" width="100%">
</p>

### *End-to-End Healthcare ML Pipeline with Explainability*

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

> **Bayesian Minds** | University of Moratuwa  
> DataXplore 2.0 Data Science Competition — Stage 01

---

## 📋 Project Overview

This project presents a comprehensive machine learning analysis for predicting stroke risk using patient health and lifestyle data. Developed as part of the **DataXplore 2.0** competition, the analysis covers the full data science pipeline — from raw data exploration to model explainability using SHAP values.

**Dataset:** 9,722 patient records | 17 features | Binary classification (stroke vs no stroke)

---

## 🏆 Key Results

| Model | AUC-ROC | Sensitivity | Specificity | CV AUC |
|---|---|---|---|---|
| **XGBoost Tuned** ⭐ | **0.8512** | **0.82** | 0.7222 | 0.8468 |
| Random Forest | 0.8472 | 0.50 | 0.9300 | 0.8338 |
| XGBoost | 0.8371 | 0.72 | 0.8374 | 0.8349 |
| Logistic Regression | 0.8364 | 0.82 | 0.7531 | 0.8458 |

**Best Model:** XGBoost Tuned — achieves both the highest AUC (0.8512) and highest sensitivity (0.82), catching **82% of actual stroke patients**.

---

## 📁 Repository Structure

```
stroke-risk-prediction/
│
├── BayesianMinds_UniversityOfMoratuwa.ipynb   # Main analysis notebook
├── BayesianMinds_UniversityOfMoratuwa.pdf     # Full competition report
├── healthcare_data.csv                         # Dataset (9,722 records)
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

---

## 🔬 Analysis Pipeline

### 1. Data Understanding & Exploration
- Profiled 9,722 records across 17 features
- Identified 4,612 duplicate records (47.4%) — removed to restore true class distribution
- Detected non-random BMI missingness (201 records, obese category only)

### 2. Data Preparation
- **Deduplication:** 9,722 → 5,110 records
- **Feature Audit:** Dropped `employment_type` (age proxy, r=0.191) and identified derived redundant features
- **Leakage-Safe Pipeline:** Train/test split performed before all imputation and scaling
- **Encoding:** One-hot for nominal; ordinal mapping for ordered categories
- **Class Imbalance:** `class_weight='balanced'` and `scale_pos_weight=19.54` — no SMOTE (leakage risk)

### 3. Exploratory Data Analysis (19 Figures)
- Stroke distribution, risk factor category analysis
- Age × Glucose risk zone heatmap (clinically actionable triage matrix)
- BMI × Hypertension interaction analysis
- Pairwise scatter plots, KDE distributions, correlation heatmap with significance markers

### 4. Statistical Analysis
- **T-tests & Mann-Whitney U:** Age gap 67.7 vs 42.0 years (t=18.08, p<0.001)
- **Chi-Square & Cramér's V:** age_group strongest association (V=0.240)
- **Odds Ratios:** Heart disease OR=4.71 (95% CI: 3.34–6.64), Hypertension OR=3.70

### 5. Model Development
- 4 models: Logistic Regression, Random Forest, XGBoost, XGBoost Tuned
- **GridSearchCV:** 162 configurations × 5 folds = 810 total fits
- **Threshold optimisation:** F1 maximised at t=0.65; screening optimised at t=0.10–0.25
- **5-fold stratified CV:** All models achieve 0.83–0.85 CV AUC

### 6. Model Explainability (SHAP)
- SHAP TreeExplainer on XGBoost Tuned
- Beeswarm plot confirms: high age → consistently high positive SHAP contribution
- BMI shows heterogeneous SHAP distribution (non-linear interaction with age)

---

## 🔑 Key Findings

1. **Age is the dominant predictor** — confirmed by every method (t-test, Cramér's V, Gini, SHAP). Mean age gap of 25.8 years between stroke and non-stroke patients.

2. **Top 3 modifiable risk factors:** Glucose level, BMI, Hypertension — all clinically measurable and addressable.

3. **Heart Disease Paradox:** Highest odds ratio (OR=4.71) but low Gini importance — because Gini favours continuous variables split across many patients. Clinical significance must not be inferred from feature importance alone.

4. **Feature Selection Insight:** Removing derived redundant features improved base XGBoost AUC from 0.767 to 0.839 (+0.080). GridSearchCV on the full feature set achieved 0.8512 — demonstrating that systematic tuning can outperform manual feature reduction.

5. **Age × Glucose Risk Zone:** Patients aged 76+ with glucose >126 mg/dL face stroke rates exceeding 20% — an immediately actionable high-priority cohort.

---

## 🛠️ Setup & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
1. Clone the repository
2. Place `healthcare_data.csv` in the root directory
3. Open `BayesianMinds_UniversityOfMoratuwa.ipynb` in Jupyter or Google Colab
4. Run all cells top to bottom

### Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/stroke-risk-prediction/blob/main/BayesianMinds_UniversityOfMoratuwa.ipynb)

---

## 📊 Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Statistics | scipy.stats |
| Machine Learning | scikit-learn, xgboost |
| Explainability | SHAP |
| Environment | Google Colab |

---

## 👥 Team — Bayesian Minds

| Name | Contribution |
|---|---|
| Dinusha Ekanayake | Model Development & Hyperparameter Tuning |
| Kavinda Mihiran | Statistical Analysis |
| Amandi Arangala | Data Cleaning, EDA & Visualizations |
| Tharusha Udana | Report Writing & Discussion |

**University of Moratuwa**

---

## 📄 License

This project was developed for the DataXplore 2.0 competition. The dataset is provided by the competition organisers.

---

## 🔗 References

- WHO Global Health Estimates (2023)
- Framingham Stroke Risk Score
- Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System
- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP)
