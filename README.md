# Stock Market Close Price Prediction

## Project Overview
This project focuses on building a robust and generalized machine learning model to predict the next trading day close price of stocks using historical market data. The notebook demonstrates a complete data science workflow starting from raw data acquisition to model evaluation, with an emphasis on correctness, stability, and real market constraints.

The project is designed to work across a wide range of stocks, from low-priced microcap equities to high-priced large-cap stocks and market indices.

---

## Problem Statement
Stock prices are influenced by complex market dynamics and exhibit non-stationary time series behavior. The objective of this project is not to forecast exact prices with certainty, but to learn statistically meaningful patterns from historical price action and quantify predictive performance using appropriate regression metrics.

---

## Data Collection
Historical stock data is sourced programmatically using the Yahoo Finance API. The notebook supports:
- Individual equities listed on NSE and BSE
- Market indices such as NIFTY and SENSEX
- Long historical windows up to 10 years when data availability permits

Adjusted price data is used to account for corporate actions such as splits and dividends, ensuring consistency across long time horizons.

---

## Exploratory Data Analysis
Exploratory analysis is performed to understand:
- Price trends and volatility over time
- Distribution of returns
- Presence of outliers and missing values
- Differences in scale between low-priced and high-priced stocks

Visualization and summary statistics are used to validate assumptions before feature engineering and modeling.

---

## Feature Engineering
The notebook constructs features using strictly past information to avoid data leakage. These features capture:
- Short-term momentum
- Price volatility
- Trend behavior
- Lagged price and return dynamics

A log-return based formulation is used internally to ensure scale invariance, allowing the model to generalize across stocks with very different price levels.

---

## Preprocessing and Scaling
To ensure numerical stability and comparability across features:
- Standardization is applied where required
- Targets are transformed to stabilize variance
- All preprocessing steps are fitted on training data only

This design allows the same model architecture to work effectively across different securities.

---

## Modeling Approach
The project employs a regularized regression baseline alongside a gradient boosting model to capture both linear structure and non-linear interactions in the data.

Key characteristics of the modeling approach:
- Time-aware train test split to respect temporal order
- Cross validation using expanding or time series splits
- Hyperparameter tuning to balance bias and variance
- Regularization to prevent overfitting

The final model selection prioritizes consistency and error stability rather than isolated metric peaks.

---

## Evaluation Metrics
Model performance is evaluated using regression-appropriate metrics:
- Mean Absolute Error
- Root Mean Squared Error
- R squared
- Error analysis on reconstructed price space

Metrics are interpreted in context, acknowledging the inherent noise and unpredictability of financial markets.

---

## Results and Observations
- Predictive performance varies across assets, as expected in real markets
- Log-return modeling improves generalization across price scales
- Regularization and proper validation significantly reduce overfitting
- The model captures short-term structure better than long-term price direction

The notebook clearly distinguishes between statistical performance and practical market usability.

---

## Design Philosophy
This project intentionally avoids unrealistic assumptions such as perfect foresight or future data leakage. Instead, it demonstrates how machine learning models should be built, evaluated, and interpreted when dealing with financial time series.

The focus is on reproducibility, clarity, and professional structure rather than over-engineering.

---

## Tools and Technologies
- Python
- Pandas and NumPy for data processing
- Matplotlib for visualization
- Scikit-learn for preprocessing and regression models
- Gradient boosting libraries for non-linear modeling
- Jupyter Notebook for experimentation and documentation

---

## Intended Use
This project is suitable as:
- A portfolio demonstration of financial data science skills
- An interview-ready case study for ML and data roles
- A foundation for further work in quantitative modeling or deployment

It is not intended to be used as financial advice or a trading system.

---

## Limitations
- Market behavior is inherently stochastic and non-stationary
- External macroeconomic and news factors are not modeled
- Predictions should be interpreted probabilistically, not deterministically

These limitations are acknowledged and reflected in the evaluation strategy.

---

## Conclusion
This notebook presents a disciplined and realistic approach to stock price regression modeling. It highlights how careful data handling, feature design, regularization, and evaluation are more important than model complexity when working with financial time series.

The project emphasizes methodological rigor over speculative claims, aligning with industry expectations for applied machine learning work.
