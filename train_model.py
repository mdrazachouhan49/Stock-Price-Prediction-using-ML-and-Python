import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from xgboost import XGBRegressor

ticker = "RELIANCE.NS"

df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False)
df.dropna(inplace=True)

data = df.copy()
data["Return_1d"] = data["Close"].pct_change()
data["HL_Range"] = data["High"] - data["Low"]
data["OC_Range"] = data["Open"] - data["Close"]

for w in [5, 10, 20]:
    data[f"SMA_{w}"] = data["Close"].rolling(window=w).mean()
    data[f"STD_{w}"] = data["Close"].rolling(window=w).std()

for lag in [1, 2, 3, 5, 10]:
    data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
    data[f"Return_Lag_{lag}"] = data["Return_1d"].shift(lag)

data["Target_Close_Next"] = data["Close"].shift(-1)
data.dropna(inplace=True)

feature_cols = [
    "Return_1d", "HL_Range", "OC_Range",
    "SMA_5", "SMA_10", "SMA_20",
    "STD_5", "STD_10", "STD_20",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5", "Close_Lag_10",
    "Return_Lag_1", "Return_Lag_2", "Return_Lag_3", "Return_Lag_5", "Return_Lag_10"
]

X = data[feature_cols]
y = data["Target_Close_Next"]

model = XGBRegressor(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X, y)

joblib.dump(model, "stock_xgb_model.pkl")
joblib.dump(feature_cols, "stock_feature_cols.pkl")

print("Saved: stock_xgb_model.pkl and stock_feature_cols.pkl")
