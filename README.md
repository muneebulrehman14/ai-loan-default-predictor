
# 🏦 Loan Default Prediction System for Rural Pakistan Microfinance

## Research Project | Bahria University | Spring 2026

**Authors:** Muneeb Ul Rehman

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-RandomForest%20%7C%20XGBoost%20%7C%20LogisticRegression-green.svg)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/UI-Tkinter-orange.svg)](https://docs.python.org/3/library/tkinter.html)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Problem Statement

**8.5 million rural borrowers in Pakistan have no formal credit history.** 
- No credit scores, no income documents, no banking history
- Rural microfinance portfolios face **12–20% non-performing loan ratios**
- Loan officers rely on guesswork due to **78% vs 22% class imbalance**

## 💡 Our Solution

A **100% offline desktop application** that helps microfinance loan officers predict default risk before approving loans, using:

- ✅ **3 Machine Learning Models** (Random Forest, Logistic Regression, XGBoost)
- ✅ **SMOTE** for handling class imbalance
- ✅ **Color-coded risk meter** for instant decisions
- ✅ **Plain-language recommendations** for loan officers
- ✅ **PNG report export** for record keeping
- ✅ **100% offline** – works without internet in remote villages

## 📊 Dataset

- **255,347 real loan records** (Jan 2021 – Dec 2023)
- **4 provinces**: Punjab (42%), Sindh (28%), KPK (19%), Balochistan (11%)
- **Default rate**: 21.75% (90+ days past due)
- **13 input features**: age, income, loan term, collateral, education, occupation, region, etc.

## 🤖 Models Performance

| Model | Accuracy | Recall | F1 Score | Best For |
|-------|----------|--------|----------|----------|
| **XGBoost** | 77.47% | 16% | - | Highest accuracy |
| **Random Forest** | 70.2% | ~30% | 0.288 | Balanced performance |
| **Logistic Regression** | 59.0% | **47.5%** | ~37% | **Finding defaulters** ⭐ |

> **⭐ Recommendation for Microfinance:** Logistic Regression (best recall = catches most defaulters)

## 🔑 Key Insights

| Rank | Feature | Impact |
|------|---------|--------|
| #1 | **Loan Term** | Long-term (24-36 months) default **2.5× more** than short-term |
| #2 | **Education** | More educated = better money management, steadier income |
| #3 | **Collateral** | Asset pledging = more incentive to repay |
| #9 | **Credit Score** | ❌ Useless – rural borrowers have no formal credit history |

## 📱 Application Features

| Feature | Description |
|---------|-------------|
| **13-field input form** | All borrower details entered easily |
| **3 model predictions** | Risk probability from all models at once |
| **Color-coded risk meter** | 🟢 Green / 🟡 Yellow / 🔴 Red for instant decisions |
| **Plain-language advice** | e.g., "Approve with 12-month term" or "Require collateral" |
| **PNG report export** | Save & print results for loan officer's file |
| **100% offline** | Critical for rural Pakistan with unreliable internet |

## 🛠️ Tech Stack
