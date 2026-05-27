# DSE Market Stress Prediction using Machine Learning

This project presents a machine learning-based framework for predicting market stress conditions in the Dhaka Stock Exchange (DSE) using historical financial data collected from 2008 to 2022. The framework classifies trading days into three market conditions: **Normal**, **High Volatility**, and **Crash**. The proposed approach focuses on identifying market stress patterns and rare crash events to support financial risk analysis and market monitoring.

The study evaluates multiple classical machine learning models and incorporates explainable AI techniques to improve model interpretability and transparency.

---

# Project Objectives

- Predict market stress conditions in the Dhaka Stock Exchange
- Detect high-volatility and crash events
- Compare the performance of multiple machine learning models
- Analyze influential financial indicators
- Improve interpretability using explainable AI techniques

---

# Repository Structure

```bash
Experiment_v2/
│
├── Models/
│   ├── Decision_Tree.ipynb
│   ├── Random_Forest.ipynb
│   ├── Extra_Trees.ipynb
│   ├── logistic_Regression.ipynb
│   └── xgboost.ipynb
│
├── data_engineering.ipynb
├── error_analysis.ipynb
└── statistical_testing.ipynb
```

---

# Methodology

## 1. Data Engineering

Historical stock market data from 2008 to 2022 is processed through multiple preprocessing stages, including:

- Data cleaning
- Missing value handling
- Feature normalization
- Statistical feature generation
- Market stress labeling

Several financial indicators and engineered features are extracted, including:

- Daily Return
- Intraday Volatility
- High–Low Price Spread
- Rolling Volatility
- Volume Change Percentage
- Trade Change Percentage
- Momentum Indicators
- Price Gap Features

Notebook:
```bash
data_engineering.ipynb
```

---

## 2. Market Stress Labeling

A three-class target variable is constructed to classify trading days into:

- Normal Market
- High Volatility
- Crash

The labeling strategy is based on statistically derived stock-specific thresholds using daily return and intraday volatility measurements.

---

## 3. Machine Learning Models

The following machine learning models are trained and evaluated:

- Decision Tree
- Random Forest
- Extra Trees
- Logistic Regression
- XGBoost

The models are designed to identify different market stress conditions and rare crash events from historical market behavior.

Location:
```bash
Models/
```

---

## 4. Performance Evaluation

Model performance is evaluated using several classification metrics, including:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Experimental results show that XGBoost achieves the best overall performance with strong crash detection capability, while Random Forest and Logistic Regression also provide competitive results.

Notebook:
```bash
error_analysis.ipynb
```

---

## 5. Statistical Testing and Explainability

Statistical testing and explainable AI techniques are used to analyze model behavior and feature importance.

The framework helps:
- Identify influential financial indicators
- Interpret model predictions
- Analyze market crash patterns
- Improve transparency for financial decision-making

Notebook:
```bash
statistical_testing.ipynb
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

---

# Key Concepts

- Financial Market Stress Prediction
- Supervised Machine Learning
- Financial Feature Engineering
- Multi-Class Classification
- Explainable AI (XAI)
- Financial Risk Analysis
- Market Crash Detection
