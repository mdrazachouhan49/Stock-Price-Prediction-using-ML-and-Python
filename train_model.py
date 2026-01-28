import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost (recommended)
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# -----------------------------
# Download Helpers
# -----------------------------
def download_data(ticker: str, period="10y", interval="1d", auto_adjust=True) -> pd.DataFrame:
    """
    Robust yfinance downloader:
    - Returns empty DataFrame if ticker fails
    - Fixes MultiIndex columns
    - Uses auto_adjust=True for split-adjusted prices
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna().copy()
        df.columns = [c.strip().title() for c in df.columns]
        return df

    except Exception:
        return pd.DataFrame()


def download_indexes(period="10y", interval="1d", auto_adjust=True) -> pd.DataFrame:
    nifty = download_data("^NSEI", period, interval, auto_adjust).add_prefix("NIFTY_")
    sensex = download_data("^BSESN", period, interval, auto_adjust).add_prefix("SENSEX_")

    if nifty.empty or sensex.empty:
        raise ValueError("Index download failed. Check internet or yfinance.")

    return nifty.join(sensex, how="inner")


# -----------------------------
# Feature Engineering
# -----------------------------
def build_features(stock_df: pd.DataFrame, index_df: pd.DataFrame, ticker_name: str) -> pd.DataFrame:
    """
    Creates only past-based features (no leakage).
    Target = next-day log return (scale-free, works across price ranges).
    """
    data = stock_df.copy()
    data = data.join(index_df, how="inner")

    # Stock identity
    data["Ticker"] = ticker_name

    # Returns
    data["Return_1d"] = data["Close"].pct_change()
    data["LogReturn_1d"] = np.log(data["Close"] / data["Close"].shift(1))

    # Ranges
    data["HL_Range"] = data["High"] - data["Low"]
    data["OC_Range"] = data["Open"] - data["Close"]

    # Rolling features
    for w in [5, 10, 20, 50]:
        data[f"SMA_{w}"] = data["Close"].rolling(w).mean()
        data[f"EMA_{w}"] = data["Close"].ewm(span=w, adjust=False).mean()
        data[f"STD_{w}"] = data["Close"].rolling(w).std()

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
        data[f"Return_Lag_{lag}"] = data["Return_1d"].shift(lag)

    # Volume signals
    if "Volume" in data.columns:
        data["Vol_SMA_20"] = data["Volume"].rolling(20).mean()
        data["Vol_Ratio"] = data["Volume"] / (data["Vol_SMA_20"] + 1e-9)

    # Market context
    data["NIFTY_Return_1d"] = data["NIFTY_Close"].pct_change()
    data["SENSEX_Return_1d"] = data["SENSEX_Close"].pct_change()

    for w in [5, 10, 20]:
        data[f"NIFTY_SMA_{w}"] = data["NIFTY_Close"].rolling(w).mean()
        data[f"SENSEX_SMA_{w}"] = data["SENSEX_Close"].rolling(w).mean()

    data["RelStrength_NIFTY"] = data["Return_1d"] - data["NIFTY_Return_1d"]
    data["RelStrength_SENSEX"] = data["Return_1d"] - data["SENSEX_Return_1d"]

    # Target
    data["Target_LogReturn_Next"] = data["LogReturn_1d"].shift(-1)

    data.dropna(inplace=True)
    return data


# -----------------------------
# Metrics
# -----------------------------
def metrics_logreturn(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# -----------------------------
# Training Pipeline
# -----------------------------
def main():
    print("Downloading index data...")
    index_df = download_indexes(period="10y", interval="1d", auto_adjust=True)
    print("Index data shape:", index_df.shape)

    # 30 stocks (includes MRF for high-price robustness)
    train_tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "AXISBANK.NS", "ITC.NS", "LT.NS", "BHARTIARTL.NS",
        "TITAN.NS", "SUNPHARMA.NS", "WIPRO.NS", "HCLTECH.NS", "M&M.NS",
        "TATAMOTORS.NS", "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
        "COALINDIA.NS", "JSWSTEEL.NS", "HINDALCO.NS", "CIPLA.NS", "BPCL.NS",
        "ULTRACEMCO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "EICHERMOT.NS", "MRF.NS"
    ]

    all_data = []
    failed = []

    print("\nDownloading and building features...")
    for tkr in train_tickers:
        stock_df = download_data(tkr, period="10y", interval="1d", auto_adjust=True)

        if stock_df.empty or len(stock_df) < 350:
            failed.append(tkr)
            continue

        feat_df = build_features(stock_df, index_df, ticker_name=tkr)
        all_data.append(feat_df)
        print("OK:", tkr, feat_df.shape)

    if len(all_data) == 0:
        raise ValueError("No valid stock data downloaded. Training cannot continue.")

    data = pd.concat(all_data, axis=0).sort_index()

    print("\nUniversal dataset shape:", data.shape)
    print("Failed tickers:", failed)

    target_col = "Target_LogReturn_Next"
    X = data.drop(columns=[target_col], errors="ignore").copy()
    y = data[target_col].copy()

    # Time split
    split_index = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = ["Ticker"]

    tscv = TimeSeriesSplit(n_splits=6)

    # Choose model
    if XGB_AVAILABLE:
        preprocess = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols)
            ]
        )

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", XGBRegressor(
                objective="reg:squarederror",
                random_state=42
            ))
        ])

        param_grid = {
            "model__n_estimators": [400, 700, 1000],
            "model__learning_rate": [0.01, 0.03, 0.05],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_alpha": [0, 0.1, 0.5, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0, 5.0]
        }

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=25,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            random_state=42,
            n_jobs=-1
        )

        print("\nTraining XGBoost...")
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print("Best params:", search.best_params_)

    else:
        preprocess = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), numeric_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols)
            ]
        )

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", Ridge())
        ])

        param_grid = {"model__alpha": np.logspace(-4, 4, 40)}

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=25,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            random_state=42,
            n_jobs=-1
        )

        print("\nTraining Ridge...")
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print("Best params:", search.best_params_)

    # Evaluate
    preds = best_model.predict(X_test)
    scores = metrics_logreturn(y_test, preds)

    print("\nTest Metrics (LogReturn):")
    for k, v in scores.items():
        print(f"{k}: {v:.6f}")

    # Save artifacts
    joblib.dump(best_model, "universal_stock_regression_model.pkl")
    joblib.dump(list(X.columns), "universal_feature_cols.pkl")

    print("\nSaved artifacts:")
    print("universal_stock_regression_model.pkl")
    print("universal_feature_cols.pkl")


if __name__ == "__main__":
    main()
