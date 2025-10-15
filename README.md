# Stock Price Forecasting: Prophet vs LSTM vs Hybrid Model (NVIDIA Case Study)

## Overview
This project builds and compares multiple time series forecasting models to predict **daily stock closing prices** for **NVIDIA (NVDA)** using 5 years of historical data.

We start from a simple **Prophet** model and progressively evolve to a **deep learning LSTM hybrid** with **technical indicators**, **market context (S&P500)**, and **hyperparameter tuning via Optuna**.
<img width="1051" height="449" alt="image" src="https://github.com/user-attachments/assets/d1846743-167e-482a-81e8-086c505b50d0" />

---

## Project Evolution

| Stage | Model Type | Description | Result Summary |
|-------|-------------|--------------|----------------|
| **1.Prophet (Baseline)** | Additive statistical model | Trend + seasonality + SPY regressor | Poor accuracy (MAE ≈ 24.5, R² ≈ -2.9) |
| **2.LSTM (Deep Learning)** | Sequential neural model | Captures nonlinear, temporal dependencies | MAE ≈ 9.1, R² ≈ 0.44 |
| **3.Hybrid LSTM (with Indicators)** | LSTM + BiLSTM + GRU variants | Learns complex dynamics + volatility patterns | MAE ≈ 4.0, R² ≈ 0.88 |
| **4. Hyperparameter Tuning (Optuna)** | Automated optimization | Tunes architecture, seq length, dropout, LR | Finalized robust, high-accuracy model |

---

## Data Pipeline

### **Sources**
- NVDA stock data — via `yfinance`
- S&P500 index (SPY) — market reference
- Period: **Last 5 years**
- 

### **Schema**
| Column | Description |
|--------|-------------|
| `Open` | Daily open price |
| `High` | Daily high |
| `Low` | Daily low |
| `Close` | Daily close (target variable) |
| `Volume` | Trading volume |
| `SPY_Close` | S&P500 closing price |

### **Data Cleaning**
- Replaced missing values (`NaN`, `null`, blanks, `'-'`)  
- Used **forward-backward fill** and **rolling average smoothing**  
- Flattened MultiIndex columns (`('Close', 'NVDA')` → `Close_NVDA`)

---

## ⚙️ Feature Engineering

To enhance predictive power, we added **technical indicators**:

| Indicator | Formula / Logic | Insight |
|------------|------------------|----------|
| `Return` | % change of close | Daily volatility |
| `SMA_10`, `SMA_30` | Simple moving averages | Trend smoothing |
| `EMA_10`, `EMA_30` | Exponential moving averages | Weighted momentum |
| `BB_upper`, `BB_lower` | Bollinger Bands | Volatility range |
| `RSI` | 14-day relative strength index | Overbought / oversold |
| `MACD` | EMA(12) - EMA(26) | Trend momentum |

All numeric features scaled using **MinMaxScaler (0–1 range)**.

---

## Models and Architecture

### ** Prophet (Baseline)**
Model Output ::
Tuned Prophet Model Performance
MAE : 23.73
RMSE: 675.88
MAPE: 13.51%
R²  : -2.755

### ** LSTM :
LSTM Model Performance
MAE : 9.15
RMSE: 100.12
MAPE: 5.24%
R²  : 0.444 (Much better than Prophet , as prophet does not handle sequential data or any addition patterns)
<img width="1185" height="590" alt="image" src="https://github.com/user-attachments/assets/f16ee323-c8c6-42fa-8c0c-1f9cb6599d79" />

### ** Hybrid (LSTM + GRU + BiLSTM) (best choice) with OPTUNA Optimization
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, n_features))),
    Dropout(0.3),
    GRU(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

Hybrid LSTM (tuned) Performance
MAE : 4.02
RMSE: 21.64
MAPE: 2.32%
R²  : 0.880
<img width="990" height="451" alt="image" src="https://github.com/user-attachments/assets/7718e285-8c4b-427f-a041-84f026841fda" />



