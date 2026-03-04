# Data Science Capstone — Forex Auto-Trading Bot

**Student:** Orgil BK | **Course:** AUM Data Science Capstone, Spring 2026 | **Instructor:** Robert Ritz

## Project Overview
Building an ML-powered trading bot for the EUR/USD forex pair using MetaTrader 5. The model classifies market conditions into **Buy**, **Sell**, or **Hold** signals based on technical indicators and automates trade execution.

## Checkpoints

| Checkpoint | Topic | Status |
|---|---|---|
| 1 — Proposal | Project scope, data source, approach | ✅ Done |
| 2 — EDA | Data collection, quality, visualizations, insights | ✅ Done |
| 3 — MVP | Baseline + ML model training | 🔜 Week 8 |
| 4 — Prototype | Full dashboard + bot integration | 🔜 Week 11 |

## Repository Structure

```
Capstone/
├── eda_checkpoint2.ipynb     # Checkpoint 2: EDA notebook
├── eurusd_h1_raw.csv         # Raw EUR/USD H1 data (2 years, via yfinance)
├── eurusd_h1_features.csv    # Feature-engineered dataset with labels
├── viz1_price_history.png
├── viz2_return_distribution.png
├── viz3_volatility_patterns.png
├── viz4_correlation_heatmap.png
├── viz5_bollinger_rsi.png
├── viz6_signal_distribution.png
└── viz7_rsi_atr_by_signal.png
```

## Setup

```bash
pip install yfinance pandas numpy matplotlib seaborn ta
jupyter notebook eda_checkpoint2.ipynb
```

> **Note:** Data is fetched from Yahoo Finance (`EURUSD=X`). No API key required.

## Tech Stack
- **Data:** yfinance (EUR/USD H1, 2 years)
- **Features:** RSI, MACD, Bollinger Bands, ATR, Moving Averages (`ta` library)
- **Model (planned):** XGBoost / Random Forest classifier
- **Dashboard (planned):** Streamlit + MetaTrader 5
