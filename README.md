# DSE Market Stress Prediction using Ensemble Meta-Learning

This project predicts **market stress and crash risk in the Dhaka Stock Exchange (DSE)** using supervised machine learning and explainable AI techniques. Instead of predicting stock prices, the system focuses on identifying **high-risk, high-volatility, or crash days**, which is more useful for market risk monitoring and financial analysis.

The workflow includes **data preprocessing, feature engineering, multiple machine learning models, an ensemble meta-learning approach, and model explainability using LIME**.

---

## Project Objectives

- Detect **market stress and crash risk**
- Compare performance of multiple ML models
- Improve prediction using **ensemble meta-learning**
- Provide **explainability** using LIME feature analysis

---

## Methodology

### 1. Data Processing & Feature Engineering
The dataset is processed and enhanced with engineered features relevant to financial market behavior, such as:

- Daily return
- High–Low price spread
- Intraday volatility
- Volume change percentage
- Trade change percentage
- Price gap
- Rolling volatility
- Momentum-based indicators

Location:
```
Data Processing/
```

Notebooks:
- `data_processing_and_feature_engineering.ipynb`
- `final_dataset_analysis.ipynb`

---

### 2. Machine Learning Models

Four supervised machine learning models were trained and evaluated:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Extra Trees**

Location:
```
Models/
```

Notebooks:
- `logistic_Regression.ipynb`
- `Random_Forest.ipynb`
- `xgboost.ipynb`
- `Extra_Trees.ipynb`

---

### 3. Ensemble Meta-Learning

To improve prediction performance, the outputs of the base models are combined using a **meta-learning approach**.

Process:
1. Train base models.
2. Extract prediction probabilities from each model.
3. Use these probabilities as **input features for a Logistic Regression meta-model**.
4. The meta-model produces the **final prediction**.

Notebook:
```
Models/Meta_Learning.ipynb
```

---

### 4. Explainable AI (LIME)

To interpret model predictions, **LIME (Local Interpretable Model-Agnostic Explanations)** is used.

This allows:
- Understanding **which features influence predictions**
- Explaining **individual high-risk or crash predictions**
- Providing both **local and global feature insights**

---

## Project Structure

```
DSE_MARKET_STRESS
│
├── Data Processing
│   ├── data_processing_and_feature_engineering.ipynb
│   └── final_dataset_analysis.ipynb
│
└── Models
    ├── logistic_Regression.ipynb
    ├── Random_Forest.ipynb
    ├── xgboost.ipynb
    ├── Extra_Trees.ipynb
    └── Meta_Learning.ipynb
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- LIME

---

## Key Concepts Used

- Supervised Machine Learning
- Feature Engineering
- Ensemble Learning
- Meta Learning (Stacking)
- Explainable AI (XAI)
- Financial Risk Prediction

---
