# DSE Market Stress Prediction using Machine Learning

This project presents a rigorously validated machine learning framework for predicting market stress conditions in the Dhaka Stock Exchange (DSE) using historical financial data collected from 2008 to 2022. The framework classifies trading days into three market conditions: **Normal**, **High Volatility**, and **Crash**, with the target explicitly defined as a *next-day forecast* rather than a same-day description, to ensure the framework reflects a genuine early-warning task rather than a retrospective labeling exercise.

The study evaluates seven models spanning linear, tree-ensemble, recurrent deep-learning, and classical econometric paradigms, using leakage-safe preprocessing, expanding-window walk-forward cross-validation, and a battery of imbalance-aware evaluation metrics, statistical significance tests, and model-agnostic interpretability techniques.

---

# Project Objectives

- Predict next-day market-stress conditions in the Dhaka Stock Exchange without same-day information leakage
- Detect high-volatility and, in particular, rare crash events under severe class imbalance
- Compare classical machine learning, deep sequential learning, and classical time-series econometric approaches
- Establish statistically significant, properly validated model rankings
- Analyze influential financial indicators using out-of-sample, model-agnostic interpretability methods
- Support transparent, reproducible financial risk analysis

---

# Repository Structure

```bash
experiment_v3/
│
├── final data processing/
│   ├── data_comparison.ipynb      # data validation with lankabangla portal
│   ├── data_engineering.ipynb     # preprocessing, feature engineering, labeling
│   └── eda.ipynb                  # exploratory data analysis and validity checks
│
├── model training/
│   ├── logistic_regression.ipynb  # linear baseline
│   ├── random_forest.ipynb        # bagging tree ensemble
│   ├── xgboost.ipynb              # boosting tree ensemble
│   ├── rd_lstm.ipynb              # LSTM recurrent architecture
│   ├── rd_gru.ipynb               # GRU recurrent architecture
│   ├── HMM.ipynb                  # Gaussian Hidden Markov Model
│   └── GARCH_v1.ipynb             # asymmetric GJR-GARCH(1,1)
│
└── stat_testing.ipynb             # Friedman, Nemenyi, Wilcoxon significance tests
```

---

# Methodology

## 1. Data Preparation

Historical DSE trading data (2008–2022) is restricted to equity instruments only (1,791,069 → 932,078 rows, 1,007 → 321 tickers) and processed through a leakage-safe preprocessing pipeline:

- Infinite-value neutralization and forward-fill-only missing-value handling (no backward-fill, to prevent future information from leaking backward in time)
- Detection and isolation of suspected trading suspensions / corporate actions (single-day returns exceeding ±20%), retained in the dataset but excluded from threshold-fitting statistics
- Train-only Winsorization (1st/99th percentile) and log/signed-log transformation of skewed features
- Full exploratory validation, including correlation-based multicollinearity checks and cross-referencing of crash-day concentration against known DSE crisis periods (2010–2011, COVID-19)

Notebooks:
```bash
final data processing/data_engineering.ipynb
final data processing/eda.ipynb
final data processing/data_comparison.ipynb
```

## 2. Feature Engineering

An extensive set of technical indicators is engineered, covering trend, momentum, and volatility structure:

- **Trend**: EMA-12, MACD, MACD signal, MACD histogram
- **Momentum**: RSI-14, Stochastic %K/%D, Rate of Change (ROC-10)
- **Volatility**: Bollinger Bandwidth, Average True Range (ATR-14), rolling 5-day volatility
- **Volume & price structure**: On-Balance Volume (OBV), volume change %, high-low spread, price gap, daily return, intraday volatility, rolling 10-day momentum

Five redundant indicators (SMA-10, SMA-20, EMA-26, Bollinger upper/lower bands) exhibiting near-perfect correlation (r > 0.98) with retained features were dropped, yielding a final 18-feature set used across all models.

## 3. Market-Stress Labeling

A three-class target variable — Normal, High-Volatility, Crash — is constructed using **robust, train-only, stock-specific thresholds**:

- **Crash**: daily return falls below a median/MAD-based robust threshold (median − 2×robust-σ), replacing a naive mean/standard-deviation rule that was found to be distorted by extreme suspension-related outliers
- **High-Volatility**: intraday volatility exceeds the stock's 75th percentile
- A pooled fallback threshold is used for thin-history (recently listed) tickers
- The label is shifted forward to **t+1**, ensuring the framework predicts tomorrow's market state from today's information rather than describing today's state from today's values

## 4. Model Training

Seven models spanning four methodological paradigms are trained and evaluated:

| Paradigm | Models |
|---|---|
| Linear | Logistic Regression |
| Tree ensemble | Random Forest, XGBoost |
| Recurrent deep learning | LSTM, GRU |
| Classical time-series | Hidden Markov Model, GJR-GARCH(1,1) |

All models (excluding HMM/GARCH's per-ticker fitting) are validated using **5-fold expanding-window walk-forward cross-validation with an embargo period**, avoiding the leakage risks of a simple hold-out split or non-chronological k-fold. Hyperparameters are tuned via a staged, budget-bounded randomized search, and class imbalance is addressed through a direct comparison of class-weighting versus SMOTE, refit strictly within each cross-validation fold.

Location:
```bash
model training/
```

## 5. Performance Evaluation

Given the severity of class imbalance (crash events comprising approximately 5% of observations), model performance is evaluated primarily using **crash-class precision-recall area under the curve (PR-AUC)** rather than accuracy or ROC-AUC, which are known to be overly optimistic under imbalance. The evaluation framework additionally includes:

- Naive persistence and lagged-volatility baseline comparisons
- Operational metrics: recall at fixed false-positive-rate, precision among top-k alerts, event-level crash detection recall
- Calibration diagnostics: Brier Skill Score, Expected Calibration Error, reliability diagrams
- Regime-specific robustness across Pre-COVID, COVID crash, Post-COVID, and Russia-Ukraine periods

Random Forest achieves the strongest cross-validated crash-class discrimination, closely followed by XGBoost and LSTM, with all supervised models substantially outperforming naive baselines and the classical time-series models.

## 6. Statistical Testing and Interpretability

Model rankings are validated using the **Friedman test** across five cross-validation folds and seven models, followed by **Nemenyi post-hoc** pairwise comparison and **Wilcoxon signed-rank** tests. Interpretability is established using **out-of-sample permutation importance** and **SHAP**, with a dedicated companion-model analysis quantifying the marginal contribution of same-day return and volatility features to isolate their influence from the remaining feature set.

Notebook:
```bash
stat_testing.ipynb
```

---

# Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras (LSTM, GRU)
- hmmlearn (Hidden Markov Model)
- arch (GJR-GARCH)
- imbalanced-learn (SMOTE)
- SHAP
- SciPy / scikit-posthocs (Friedman, Nemenyi, Wilcoxon)
- Matplotlib, Seaborn

---

# Key Concepts

- Financial Market Stress Prediction
- Leakage-Free Time-Series Machine Learning
- Multi-Class Classification Under Severe Imbalance
- Deep Sequential Learning (LSTM, GRU)
- Classical Time-Series Econometrics (GARCH, HMM)
- Precision-Recall–Oriented Model Evaluation
- Statistical Significance Testing (Friedman, Nemenyi, Wilcoxon)
- Explainable AI (SHAP, Permutation Importance)
- Financial Risk Analysis and Early-Warning Systems
