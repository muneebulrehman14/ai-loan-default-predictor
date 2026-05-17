# Loan Default Prediction for Rural Pakistan Microfinance

## Research Project | Bahria University | Spring 2026

**Authors:** Muneeb Ul Rehman 

---

## 📌 Overview

This project develops a machine learning system to predict loan defaults for microfinance borrowers in rural Pakistan. Microfinance institutions face 20%+ default rates because borrowers lack formal credit histories. Our system helps loan officers make data-driven decisions.

## 📊 Dataset

- **255,347** borrower records
- **4 provinces:** Punjab, Sindh, KPK, Balochistan
- **21.75%** default rate (class imbalance addressed with SMOTE)
- **13 features:** age, income, loan amount, loan term, dependents, distance to bank, previous loans, credit score, education, collateral, occupation, region, loan purpose

## 🤖 Models Compared

| Model | Accuracy | AUC | Recall | F1 Score |
|-------|----------|-----|--------|----------|
| Random Forest | 70.21% | 0.602 | 0.278 | 0.288 |
| Logistic Regression | 58.99% | 0.573 | **0.475** | 0.333 |
| XGBoost | **77.47%** | **0.627** | 0.160 | 0.235 |

## 🔑 Key Findings

- **Loan term** is the most important predictor (0.2006 importance)
- **Credit scores** have minimal value in rural Pakistan (0.0632 importance)
- **Logistic Regression** catches 47.5% of defaulters (best for real-world use)
- **XGBoost** achieves highest accuracy (77.47%)

## 🖥️ GUI Application

Desktop application built with **Tkinter** that runs **offline** (no internet needed in villages):

- Enter 13 borrower details
- Get default probability from all 3 models
- Color-coded risk meter (green/yellow/red)
- Plain language recommendation
- Export results as PNG report



