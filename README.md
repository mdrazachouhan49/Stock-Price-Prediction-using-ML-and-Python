# Stock Price Prediction using Python and ML

This project builds a next day stock close price prediction model using historical market data and deploys it using Streamlit for quick interactive forecasting. The main focus is on building a clean, industry style machine learning notebook with proper preprocessing, feature engineering, evaluation, and model saving.

---

## Project Objective

Predict the next trading day Close price for Indian stocks using historical OHLC data from Yahoo Finance.  
The output is a short horizon forecast intended for learning and decision support, not a guaranteed price.

---

## Workflow Summary

| Stage | What is done |
| --- | --- |
| Data Collection | Download stock data using yfinance |
| Data Preparation | Clean missing values and prepare time series format |
| Feature Engineering | Create trend, volatility, lag based features |
| Model Training | Train XGBoost Regressor on engineered features |
| Evaluation | Test predictions and compute error metrics |
| Model Saving | Save trained model and feature columns as pkl |
| Deployment | Load pkl files in Streamlit and predict instantly |

---

## Notebook Explanation

### 1 Data Collection
The notebook pulls stock data from Yahoo Finance using yfinance.  
It typically uses a multi year window to give the model enough patterns and stability.

Main columns used
- Open
- High
- Low
- Close
- Volume

---

### 2 Data Cleaning and Preparation
Key steps include
- Dropping missing rows
- Sorting by date
- Ensuring numeric types
- Creating a supervised learning structure where features come from past days and target is the next day close

Target variable  
Next day Close

---

### 3 Feature Engineering
The notebook converts raw OHLC into predictive signals that capture price action.

Core engineered features

| Feature Type | Examples |
| --- | --- |
| Returns | Return 1d and lagged returns |
| Trend | SMA 5 SMA 10 SMA 20 |
| Volatility | Rolling standard deviation windows |
| Price ranges | High Low range Open Close range |
| Lag values | Close lag 1 2 3 5 10 |

These features help the model learn short term momentum, trend direction, and volatility behavior.

---

### 4 Model Training
The model used is XGBoost Regressor because it is
- Fast
- Strong on tabular data
- Handles non linear relationships well
- Works well with engineered lag features

The model is trained on past data and evaluated on a holdout test split.

---

### 5 Model Evaluation
Instead of using classification accuracy, regression models are evaluated using error metrics.

Common evaluation metrics

| Metric | Meaning |
| --- | --- |
| MAE | Average absolute error in price units |
| RMSE | Penalizes larger errors more |
| R2 Score | Explains variance captured by model |

The notebook also compares actual vs predicted prices to visually check model stability.

---

### 6 Model Saving for Deployment
To make deployment fast, the notebook saves

| File | Purpose |
| --- | --- |
| stock_xgb_model.pkl | Trained XGBoost model |
| stock_feature_cols.pkl | Feature list required during prediction |

This ensures the Streamlit app can run without retraining.

---

## Streamlit Deployment Overview

The Streamlit app is lightweight and focused on fast inference.

It allows users to
- Select popular NSE stocks from a dropdown
- Enter custom NSE or BSE tickers
- View clean charts for 1 day 1 week 1 month and 6 months
- See KPI metrics like range and volatility
- Get a next day close forecast instantly using the saved model

The chart range selector is only for visualization and does not change the prediction pipeline.

---

## How to Run

### Install Dependencies
```bash
py -m pip install streamlit yfinance pandas numpy scikit-learn xgboost matplotlib joblib
```
---

## Run the App
```bash
py -m streamlit run stock_app.py
```

## Note :
This model generates a short horizon estimate based on historical price patterns. Market movement is uncertain, so the forecast should be treated as a probabilistic signal and not a guaranteed outcome.

## Author
Md Raza Chouhan
