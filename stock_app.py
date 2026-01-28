import base64
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Stock Close Predictor", layout="wide")


# -----------------------------
# Background + Dark Theme CSS
# -----------------------------
def set_bg(image_file: str):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
                background-size: cover;
            }}

            /* Wider container + dark overlay */
            .block-container {{
                max-width: 1250px;
                background-color: rgba(0, 0, 0, 0.68);
                padding: 2rem 2.2rem 2.2rem 2.2rem;
                border-radius: 18px;
            }}

            /* Text */
            h1, h2, h3, h4, h5, h6, p, span, label {{
                color: #ffffff !important;
            }}

            /* Buttons */
            .stButton > button {{
                background-color: rgba(255, 255, 255, 0.08);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.25);
                border-radius: 12px;
                padding: 0.50rem 1.1rem;
                font-weight: 600;
            }}
            .stButton > button:hover {{
                background-color: rgba(255, 255, 255, 0.15);
                border: 1px solid rgba(255, 255, 255, 0.35);
            }}

            /* Inputs */
            .stTextInput input, .stSelectbox div[data-baseweb="select"] {{
                background-color: rgba(255, 255, 255, 0.08) !important;
                color: white !important;
                border-radius: 12px !important;
            }}

            /* KPI Cards */
            div[data-testid="stMetric"] {{
                background-color: rgba(255, 255, 255, 0.06);
                padding: 14px;
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.15);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        pass


set_bg("bg.jpg")


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <h1 style="margin-bottom: 0.2rem;">
        <span style="color:#00ff88;">Stock Market</span> Close Predictor
    </h1>
    <p style="margin-top: 0.2rem; font-style: italic; opacity: 0.92;">
        Markets are probabilistic. Use signals responsibly, manage risk, and stay consistent.
    </p>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Load Model + Feature Columns
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("universal_stock_regression_model.pkl")
    feature_cols = joblib.load("universal_feature_cols.pkl")
    return model, feature_cols


model, feature_cols = load_artifacts()


# -----------------------------
# Download Helpers
# -----------------------------
@st.cache_data
def download_data(ticker: str, period="6mo", interval="1d", auto_adjust=True) -> pd.DataFrame:
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


@st.cache_data
def download_indexes(period="6mo", interval="1d", auto_adjust=True) -> pd.DataFrame:
    nifty = download_data("^NSEI", period=period, interval=interval, auto_adjust=auto_adjust).add_prefix("NIFTY_")
    sensex = download_data("^BSESN", period=period, interval=interval, auto_adjust=auto_adjust).add_prefix("SENSEX_")
    return nifty.join(sensex, how="inner")


# -----------------------------
# Feature Engineering (same as training)
# -----------------------------
def build_features(stock_df: pd.DataFrame, index_df: pd.DataFrame, ticker_name: str) -> pd.DataFrame:
    data = stock_df.copy()
    data = data.join(index_df, how="inner")

    data["Ticker"] = ticker_name

    data["Return_1d"] = data["Close"].pct_change()
    data["LogReturn_1d"] = np.log(data["Close"] / data["Close"].shift(1))

    data["HL_Range"] = data["High"] - data["Low"]
    data["OC_Range"] = data["Open"] - data["Close"]

    for w in [5, 10, 20, 50]:
        data[f"SMA_{w}"] = data["Close"].rolling(w).mean()
        data[f"EMA_{w}"] = data["Close"].ewm(span=w, adjust=False).mean()
        data[f"STD_{w}"] = data["Close"].rolling(w).std()

    for lag in [1, 2, 3, 5, 10]:
        data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
        data[f"Return_Lag_{lag}"] = data["Return_1d"].shift(lag)

    if "Volume" in data.columns:
        data["Vol_SMA_20"] = data["Volume"].rolling(20).mean()
        data["Vol_Ratio"] = data["Volume"] / (data["Vol_SMA_20"] + 1e-9)

    data["NIFTY_Return_1d"] = data["NIFTY_Close"].pct_change()
    data["SENSEX_Return_1d"] = data["SENSEX_Close"].pct_change()

    for w in [5, 10, 20]:
        data[f"NIFTY_SMA_{w}"] = data["NIFTY_Close"].rolling(w).mean()
        data[f"SENSEX_SMA_{w}"] = data["SENSEX_Close"].rolling(w).mean()

    data["RelStrength_NIFTY"] = data["Return_1d"] - data["NIFTY_Return_1d"]
    data["RelStrength_SENSEX"] = data["Return_1d"] - data["SENSEX_Return_1d"]

    data.dropna(inplace=True)
    return data


# -----------------------------
# Stock Selection (30 stocks)
# -----------------------------
stock_map = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "ITC": "ITC.NS",
    "L&T": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Titan": "TITAN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Wipro": "WIPRO.NS",
    "HCL Tech": "HCLTECH.NS",
    "Mahindra": "M&M.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "ONGC": "ONGC.NS",
    "Coal India": "COALINDIA.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Hindalco": "HINDALCO.NS",
    "Cipla": "CIPLA.NS",
    "BPCL": "BPCL.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "MRF": "MRF.NS"
}

left, right = st.columns([2, 1])

with left:
    stock_name = st.selectbox("Select Stock", list(stock_map.keys()))
    ticker = stock_map[stock_name]

with right:
    custom_ticker = st.text_input("Custom ticker", value="").strip()
    if custom_ticker:
        ticker = custom_ticker

st.write(f"Using ticker: **{ticker}**")


# -----------------------------
# Time Range Buttons (Chart only)
# -----------------------------
st.subheader("Chart View")

col1, col2, col3, col4 = st.columns(4)

if "chart_period" not in st.session_state:
    st.session_state.chart_period = "6mo"

with col1:
    if st.button("5D"):
        st.session_state.chart_period = "5d"
with col2:
    if st.button("1M"):
        st.session_state.chart_period = "1mo"
with col3:
    if st.button("3M"):
        st.session_state.chart_period = "3mo"
with col4:
    if st.button("6M"):
        st.session_state.chart_period = "6mo"


# -----------------------------
# Load Data (Chart)
# -----------------------------
chart_df = download_data(ticker, period=st.session_state.chart_period, interval="1d", auto_adjust=True)

if chart_df.empty or len(chart_df) < 5:
    st.error("Not enough data found for this ticker.")
    st.stop()


# -----------------------------
# KPIs
# -----------------------------
latest_close = float(chart_df["Close"].iloc[-1])
prev_close = float(chart_df["Close"].iloc[-2]) if len(chart_df) >= 2 else latest_close
change = latest_close - prev_close
change_pct = (change / (prev_close + 1e-9)) * 100

period_high = float(chart_df["Close"].max())
period_low = float(chart_df["Close"].min())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest Close", f"{latest_close:.2f}")
k2.metric("Day Change", f"{change:.2f}", f"{change_pct:.2f}%")
k3.metric("Period High", f"{period_high:.2f}")
k4.metric("Period Low", f"{period_low:.2f}")

st.caption("Formal note: These KPIs summarize the most recent market movement and price boundaries for the selected period.")
st.caption("Casual take: Quick snapshot. Clean trend. No noise.")
st.caption("Cool insight: Price action tells the story, discipline decides the outcome.")


# -----------------------------
# Chart Plot (Green if up, Red if down)
# -----------------------------
trend_up = chart_df["Close"].iloc[-1] >= chart_df["Close"].iloc[0]
line_color = "green" if trend_up else "red"

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(chart_df.index, chart_df["Close"], color=line_color, linewidth=2.2)
ax.set_title(f"{ticker} Close Price ({st.session_state.chart_period})")
ax.grid(True, alpha=0.25)
st.pyplot(fig)

trend_caption = "Trend: Bullish - Upward momentum detected." if trend_up else "Trend: Bearish - Downward pressure detected."
st.caption(f"Chart trend summary: {trend_caption}")


# -----------------------------
# Prediction Section (stable features using 2 years)
# -----------------------------
st.subheader("Next-Day Close Prediction")

df_pred = download_data(ticker, period="2y", interval="1d", auto_adjust=True)
index_pred = download_indexes(period="2y", interval="1d", auto_adjust=True)

if df_pred.empty or len(df_pred) < 80:
    st.error("Not enough data for prediction.")
    st.stop()

feat_df = build_features(df_pred, index_pred, ticker_name=ticker)

if feat_df.empty or len(feat_df) < 60:
    st.error("Not enough processed rows after feature engineering.")
    st.stop()

latest_close_pred = float(feat_df["Close"].iloc[-1])

X_live = feat_df.copy()
X_live["Ticker"] = ticker

X_live = X_live.reindex(columns=feature_cols, fill_value=np.nan)

pred_logret = float(model.predict(X_live.iloc[[-1]])[0])
pred_next_close = latest_close_pred * np.exp(pred_logret)

p1, p2 = st.columns(2)
p1.metric("Latest Close (Prediction Base)", f"{latest_close_pred:.2f}")
p2.metric("Predicted Next Close", f"{pred_next_close:.2f}")

st.caption("Formal note: The forecast is generated using a universal regression model trained on multi-stock market history.")
st.caption("Casual take: It is a model-based estimate, not a guarantee.")
st.caption("Cool insight: Let the probabilities guide you, not emotions.")


# -----------------------------
# Stock Fact Section (High/Low ever + Date)
# -----------------------------
st.subheader("Stock Facts")

all_history = download_data(ticker, period="max", interval="1d", auto_adjust=True)

if not all_history.empty and "Close" in all_history.columns:
    highest_close = float(all_history["Close"].max())
    highest_date = all_history["Close"].idxmax().date()

    lowest_close = float(all_history["Close"].min())
    lowest_date = all_history["Close"].idxmin().date()

    st.write(f"Highest Close Ever: **{highest_close:.2f}** on **{highest_date}**")
    st.write(f"Lowest Close Ever: **{lowest_close:.2f}** on **{lowest_date}**")

    random_fact_pool = [
        f"On {highest_date}, the stock marked its highest close on record.",
        f"The lowest recorded close was observed on {lowest_date}.",
        "Long-term charts often reward patience more than perfect timing.",
        "Volatility is the cost of entry for market participation."
    ]
    st.caption(f"Random market fact: {np.random.choice(random_fact_pool)}")
else:
    st.caption("Random market fact: Historical extremes could not be fetched for this ticker.")
