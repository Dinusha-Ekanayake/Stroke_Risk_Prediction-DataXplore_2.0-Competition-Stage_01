# 🧠 Stroke Risk Prediction using Machine Learning  
### *End-to-End Healthcare ML Pipeline with Explainability*

---

## 🚀 Overview

This project presents a **production-style machine learning pipeline** designed to predict stroke risk using patient health and lifestyle data.

Unlike basic ML projects, this work emphasizes:

- ✅ Clinical relevance  
- ✅ Strong statistical validation  
- ✅ Model interpretability (SHAP)  
- ✅ Robust, leakage-free machine learning practices  

📊 **Dataset:** 9,722 patient records | 17 features | Binary classification  

---

## 🎯 Objective

Stroke is a leading cause of death and long-term disability worldwide. Early identification of high-risk individuals enables:

- Preventive healthcare interventions  
- Lifestyle modifications  
- Reduced mortality rates  

👉 **Goal:**  
Build a model that **maximizes sensitivity (recall)** while maintaining balanced performance and interpretability for healthcare use.

---

## 🏆 Key Results

| Model | AUC-ROC | Sensitivity | Specificity |
|------|--------|------------|------------|
| ⭐ **XGBoost Tuned** | **0.8512** | **0.82** | 0.7222 |
| Random Forest | 0.8472 | 0.50 | 0.9300 |
| XGBoost | 0.8371 | 0.72 | 0.8374 |
| Logistic Regression | 0.8364 | 0.82 | 0.7531 |

📌 **Key Insight:**  
The final model correctly identifies **82% of stroke patients**, making it suitable for screening and early detection scenarios.

---

## 🔬 End-to-End Pipeline

### 1️⃣ Data Cleaning & Preparation

- Removed **~47% duplicate records**
- Handled **missing BMI values**
- Identified and removed **redundant derived features**
- Ensured **leakage-safe preprocessing pipeline**

---

### 2️⃣ Exploratory Data Analysis (EDA)

- Risk factor distributions  
- Feature interaction analysis (Age × Glucose, BMI × Hypertension)  
- Correlation heatmaps with statistical significance  
- Distribution plots and comparative visualizations  

---

### 3️⃣ Statistical Analysis

- **T-tests & Mann-Whitney U tests**
- **Chi-Square tests & Cramér’s V**
- **Odds Ratios with 95% confidence intervals**

📌 Example finding:
> Patients with heart disease have approximately **5× higher stroke risk**

---

### 4️⃣ Model Development

Models implemented:

- Logistic Regression  
- Random Forest  
- XGBoost  
- **Tuned XGBoost (best model)**  

⚙️ **Hyperparameter tuning:**
- GridSearchCV with **810 total model fits**

---

### 5️⃣ Model Explainability

- SHAP (SHapley Additive exPlanations)
- Feature importance validation
- Individual prediction analysis

📌 Important principle:
> In healthcare, a model must be explainable to be trustworthy.

---

## 🔑 Key Insights

### 1️⃣ Age is the strongest predictor
- Significant distribution gap between stroke and non-stroke patients  
- Confirmed by both statistical tests and ML models  

---

### 2️⃣ Clinical Importance vs Model Importance

- Heart disease shows **high statistical significance**  
- But **low model importance** due to rarity  

👉 Insight:
> Statistical impact ≠ model importance  

---

### 3️⃣ Feature Quality > Feature Quantity

- Removing redundant features improved model performance  
- Demonstrates importance of thoughtful feature engineering  

---

### 4️⃣ High-Risk Patient Segment

- Elderly individuals with high glucose levels show significantly elevated stroke risk  
- Useful for **targeted healthcare interventions**

---

## ⚠️ Design Decision: Handling Class Imbalance

Instead of SMOTE, the model uses:

```python
class_weight = 'balanced'
scale_pos_weight = 19.54