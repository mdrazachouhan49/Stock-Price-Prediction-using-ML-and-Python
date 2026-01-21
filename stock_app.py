# stock_app.py
# ==========================================================
# Fast + Subtle Stock Prediction Deployment (Streamlit)
# - Dark mode friendly UI
# - Background image: bg.jpg (same folder)
# - 1D / 1W / 1M / 6M chart selector (visual only)
# - Green line if trend up, Red if down
# - KPI dashboard row
# - Loads pre-trained XGBoost model from PKL (no training here)
# ==========================================================

import os
import base64
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt


# ----------------------------
# Page Config (must be first Streamlit command)
# ----------------------------
st.set_page_config(page_title="Stock Predictor", layout="centered")


# ----------------------------
# Background Image (bg.jpg)
# ----------------------------
def set_bg_image(image_file: str):
    if not os.path.exists(image_file):
        return

    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.72), rgba(0,0,0,0.72)),
                        url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image("bg.jpg")


# ----------------------------
# Global Styling (Dark mode friendly)
# ----------------------------
st.markdown("""
<style>
.stApp { color: white; }

section[data-testid="stSidebar"] {
    background-color: rgba(14, 17, 23, 0.88);
}

.block-container { padding-top: 1.6rem; }

.caption-text {
    font-style: italic;
    color: #B0B3B8;
    font-size: 14px;
    margin-top: 6px;
}

div[data-testid="stMetricValue"] { font-size: 22px; }
div[data-testid="stMetricDelta"] { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Title + Caption
# ----------------------------
st.markdown(
    """
    <h1 style="font-size: 34px; font-weight: 750; margin-bottom: 4px;">
        <span style="color: #00C853;">Stock Market</span>
        <span style="color: white;"> Next-Day Close Predictor</span>
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<div class='caption-text'>Market forecasts are probabilistic — use this as decision support, not certainty.</div>",
    unsafe_allow_html=True
)


# ----------------------------
# Load model once (FAST + safe path)
# ----------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "stock_xgb_model.pkl")
    feat_path = os.path.join(base_dir, "stock_feature_cols.pkl")

    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        return None, None

    model = joblib.load(model_path)
    feature_cols = joblib.load(feat_path)
    return model, feature_cols

model, feature_cols = load_model()

if model is None or feature_cols is None:
    st.error("Model files not found in the app folder.")
    st.write("Make sure these files exist in the same folder as stock_app.py:")
    st.code("stock_xgb_model.pkl\nstock_feature_cols.pkl\nbg.jpg")
    st.stop()


# ----------------------------
# Sidebar: Chart Range Selector (visual only)
# ----------------------------
st.sidebar.header("Chart Settings")

range_option = st.sidebar.radio(
    "Select Chart Range",
    ["1 Day", "1 Week", "1 Month", "6 Months"],
    index=3
)

st.sidebar.markdown(
    "<div class='caption-text'>Chart range does not affect prediction.</div>",
    unsafe_allow_html=True
)


# ----------------------------
# Stock List (20+ major stocks)
# ----------------------------
stock_map = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Larsen & Toubro": "LT.NS",
    "Life Insurance Corp": "LICI.NS",
    "Maruti Suzuki India": "MARUTI.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Titan Company": "TITAN.NS",
    "UltraTech Cement": "ULTRATECH.NS",
    "NTPC": "NTPC.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "ONGC": "ONGC.NS",
    "Bharat Electronics": "BEL.NS",
    "Hindustan Zinc": "HINDZINC.NS",
    "HAL": "HAL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Adani Power": "ADANIPOWER.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Vedanta": "VEDL.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Coal India": "COALINDIA.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Nestle India": "NESTLEIND.NS",
    "Powergrid Corporation": "POWERGRID.NS",
    "DMart": "DMART.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "BPCL": "BPCL.NS",
    "Punjab National Bank": "PNB.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Cholamandalam Investment": "CHOLAFIN.NS",
    "Canara Bank": "CANBK.NS",
    "Trent Limited": "TRENT.NS",
    "Ambuja Cements": "AMBUJACEM.NS",
    "Union Bank of India": "UNIONBANK.NS",
    "Power Finance Corporation": "PFC.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Samvardhana Motherson": "MOTHERSON.NS",
    "Solar Industries India": "SOLARINDS.NS",
    "Indian Bank": "INDIANB.NS",
    "Tata Power": "TATAPOWER.NS",
    "Cipla": "CIPLA.NS",
    "Vodafone Idea": "IDEA.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Cummins India": "CUMMINSIND.NS",
    "Indus Towers": "INDUSTOWER.NS",
    "GAIL (India)": "GAIL.NS",
    "Adani Energy Solutions": "ADANIENSOL.NS",
    "BSE": "BSE.NS",
    "HDFC AMC": "HDFCAMC.NS",
    "Jindal Steel & Power": "JINDALSTEL.NS",
    "Ashok Leyland": "ASHOKLEY.NS",
    "Polycab India": "POLYCAB.NS",
    "IDBI Bank": "IDBI.NS",
    "Siemens India": "SIEMENS.NS",
    "ABB India": "ABB.NS",
    "GMR Airports": "GMRAIRPORT.NS",
    "Lupin": "LUPIN.NS",
    "Shree Cement": "SHREECEM.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Zomato": "ETERNAL.NS",
    "Avenue Supermarts (DMart)": "DMART.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Jio Financial Services": "JIOFIN.NS",
    "Varun Beverages": "VBL.NS",
    "Indian Railway Finance Corporation": "IRFC.NS",
    "Tech Mahindra": "TECHM.NS",
    "Pidilite Industries": "PIDILITIND.NS",
    "Hero Electric": "HEROMOTOCO.NS", 
    "Bajaj Holdings & Investment": "BAJAJHLDNG.NS"
}


stock_name = st.selectbox("Select Stock", list(stock_map.keys()))
ticker = stock_map[stock_name]

custom_ticker = st.text_input("Or enter custom ticker (example: 500325.BO)", value="").strip()
if custom_ticker:
    ticker = custom_ticker

st.write(f"Using ticker: **{ticker}**")


# ----------------------------
# Helper: Ensure Close is 1D numeric Series
# ----------------------------
def get_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Ensures df['Close'] is returned as a 1D numeric pandas Series.
    Handles cases where yfinance returns multi-index columns.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # If MultiIndex columns exist, try to pick 'Close' correctly
    if isinstance(df.columns, pd.MultiIndex):
        # Common yfinance format: ('Close', '') or ('Close', 'RELIANCE.NS')
        close_cols = [c for c in df.columns if c[0] == "Close"]
        if close_cols:
            close = df[close_cols[0]]
        else:
            return pd.Series(dtype=float)
    else:
        if "Close" not in df.columns:
            return pd.Series(dtype=float)
        close = df["Close"]

    # If it's still DataFrame, take first column
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").dropna()
    return close


# ----------------------------
# Chart Data Loader (visual only)
# ----------------------------
@st.cache_data
def load_chart_data(ticker_symbol: str, range_choice: str):
    if range_choice == "1 Day":
        df = yf.download(ticker_symbol, period="1d", interval="5m", auto_adjust=False)
    elif range_choice == "1 Week":
        df = yf.download(ticker_symbol, period="7d", interval="30m", auto_adjust=False)
    elif range_choice == "1 Month":
        df = yf.download(ticker_symbol, period="1mo", interval="1h", auto_adjust=False)
    else:  # 6 Months
        df = yf.download(ticker_symbol, period="6mo", interval="1d", auto_adjust=False)

    df.dropna(inplace=True)
    return df

chart_df = load_chart_data(ticker, range_option)
close_s = get_close_series(chart_df)

if close_s.empty or len(close_s) < 2:
    st.error("No usable chart data available for this ticker/range.")
    st.stop()


# ----------------------------
# KPI Row (Dashboard Summary)
# ----------------------------
last_close = float(close_s.iloc[-1])
prev_close = float(close_s.iloc[-2])

daily_change = last_close - prev_close
daily_change_pct = (daily_change / prev_close) * 100 if prev_close != 0 else 0

range_low = float(close_s.min())
range_high = float(close_s.max())

volatility_pct = float(close_s.pct_change().std() * 100)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Close", f"{last_close:.2f}")
c2.metric("Change", f"{daily_change:.2f}", f"{daily_change_pct:.2f}%")
c3.metric("Range", f"{range_low:.2f} - {range_high:.2f}")
c4.metric("Volatility", f"{volatility_pct:.2f}%")


# ----------------------------
# Aesthetic Chart (green up / red down)
# ----------------------------
st.subheader(f"Close Price Trend of {ticker} ({range_option})")

start_price = float(close_s.iloc[0])
end_price = float(close_s.iloc[-1])

line_color = "#00C853" if end_price >= start_price else "#FF5252"

x_vals = close_s.index
y_vals = close_s.to_numpy().ravel()  # FIX: guaranteed 1D array for matplotlib

fig, ax = plt.subplots(figsize=(10, 3.6), dpi=130)

ax.plot(x_vals, y_vals, color=line_color, linewidth=2.2)

# Fill under curve (requires 1D)
ax.fill_between(
    x_vals,
    y_vals,
    y_vals.min(),
    alpha=0.08
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)

ax.set_title(f"{ticker} • Close Price", fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("")
ax.set_ylabel("Price", fontsize=10)
ax.tick_params(axis="both", labelsize=9)

ax.scatter(x_vals[-1], y_vals[-1], s=35)

ax.annotate(
    f"{end_price:.2f}",
    (x_vals[-1], y_vals[-1]),
    textcoords="offset points",
    xytext=(8, -8),
    fontsize=9
)

st.pyplot(fig, use_container_width=True)

direction = "Bullish" if end_price >= start_price else "Bearish"
st.markdown(
    f"<div class='caption-text'>Trend context: <b>{direction}</b> • Start: {start_price:.2f} → End: {end_price:.2f}</div>",
    unsafe_allow_html=True
)


# ----------------------------
# Prediction Data (fixed: 6 months daily for features)
# ----------------------------
@st.cache_data
def load_prediction_data(ticker_symbol: str):
    df = yf.download(ticker_symbol, period="6mo", interval="1d", auto_adjust=False)
    df.dropna(inplace=True)
    return df

df_pred = load_prediction_data(ticker)
close_pred = get_close_series(df_pred)

if df_pred.empty or close_pred.empty or len(df_pred) < 40:
    st.error("Not enough prediction data available for this ticker.")
    st.stop()


# ----------------------------
# Feature Engineering for Prediction (must match training)
# ----------------------------
data = df_pred.copy()

# If MultiIndex columns exist, flatten safely (rare, but safe)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [c[0] for c in data.columns]

data["Return_1d"] = data["Close"].pct_change()
data["HL_Range"] = data["High"] - data["Low"]
data["OC_Range"] = data["Open"] - data["Close"]

for w in [5, 10, 20]:
    data[f"SMA_{w}"] = data["Close"].rolling(window=w).mean()
    data[f"STD_{w}"] = data["Close"].rolling(window=w).std()

for lag in [1, 2, 3, 5, 10]:
    data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
    data[f"Return_Lag_{lag}"] = data["Return_1d"].shift(lag)

data.dropna(inplace=True)

missing_features = [c for c in feature_cols if c not in data.columns]
if missing_features:
    st.error("Feature mismatch between training and app data.")
    st.write("Missing features:")
    st.code("\n".join(missing_features))
    st.stop()

if len(data) < 1:
    st.error("Not enough processed rows for prediction.")
    st.stop()

X_latest = data[feature_cols].iloc[-1].values.reshape(1, -1)


# ----------------------------
# Prediction Output
# ----------------------------
pred_next_close = float(model.predict(X_latest)[0])
latest_close_pred = float(data["Close"].iloc[-1])

st.subheader("Next-Day Forecast :")
st.write(f"Latest Close Price in INR : **{latest_close_pred:.2f}**")
st.write(f"Predicted Next Close Price in INR : **{pred_next_close:.2f}**")

st.markdown(
    "<div class='caption-text'>This forecast is a short-horizon estimate based on historical price action.</div>",
    unsafe_allow_html=True
)
