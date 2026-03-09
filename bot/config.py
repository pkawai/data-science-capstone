# config.py — All tunable settings in one place
# Edit values here; nothing is hardcoded elsewhere.

# ── Pairs ──────────────────────────────────────────────────────────────────────
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

PAIR_CONFIGS = {
    "EURUSD": {
        "yf_symbol":    "EURUSD=X",
        "mt5_symbol":   "EURUSD",
        "pip_size":     0.0001,
        "pip_value_usd": 10.0,     # USD per pip per standard lot
        "spread_pips":  1.5,
        "decimals":     5,
    },
    "GBPUSD": {
        "yf_symbol":    "GBPUSD=X",
        "mt5_symbol":   "GBPUSD",
        "pip_size":     0.0001,
        "pip_value_usd": 10.0,
        "spread_pips":  1.8,
        "decimals":     5,
    },
    "USDJPY": {
        "yf_symbol":    "USDJPY=X",
        "mt5_symbol":   "USDJPY",
        "pip_size":     0.01,      # JPY pairs: 1 pip = 0.01
        "pip_value_usd": 6.7,      # ~1000 JPY / 150 USD/JPY ≈ $6.70 per pip per lot
        "spread_pips":  1.5,
        "decimals":     3,
    },
    "AUDUSD": {
        "yf_symbol":    "AUDUSD=X",
        "mt5_symbol":   "AUDUSD",
        "pip_size":     0.0001,
        "pip_value_usd": 10.0,
        "spread_pips":  1.8,
        "decimals":     5,
    },
}


def model_path(symbol: str) -> str:
    """Return the model file path for a given symbol."""
    return f"model_{symbol}.pkl"


# ── Data ───────────────────────────────────────────────────────────────────────
TIMEFRAME     = "1h"                # yfinance interval
MT5_TIMEFRAME = 16388               # mt5.TIMEFRAME_H1 (avoid MT5 import on Mac)
YEARS_HISTORY = 2                   # how far back to pull data

# ── Trading sessions (UTC hours) ───────────────────────────────────────────────
ACTIVE_HOURS = set(range(7, 18))    # London + NY sessions (7am–5pm UTC)

# ── Feature engineering ───────────────────────────────────────────────────────
RSI_PERIOD    = 14
MACD_FAST     = 12
MACD_SLOW     = 26
MACD_SIGNAL   = 9
BB_PERIOD     = 20
BB_STD        = 2.0
ATR_PERIOD    = 14
MA_SHORT      = 20
MA_LONG       = 50
LAG_PERIODS   = [1, 2, 3]

# ── Labeling ──────────────────────────────────────────────────────────────────
LOOKAHEAD     = 5
ATR_THRESHOLD = 0.5

# Triple barrier labeling
TB_HORIZON    = 10
TB_TP_MULT    = 1.5
TB_SL_MULT    = 1.0

# ── Regime / trade filters ────────────────────────────────────────────────────
ADX_PERIOD    = 14
ADX_THRESHOLD = 25
VOL_RATIO_MAX = 2.0

# ── Model ─────────────────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":      "mlogloss",
    "random_state":     42,
    "n_jobs":           -1,
}
CONFIDENCE_THRESHOLD = 0.65

# ── Walk-forward backtest ─────────────────────────────────────────────────────
TRAIN_MONTHS  = 15
TEST_MONTHS   = 3
STEP_MONTHS   = 3

# ── Backtest costs (defaults — overridden per pair via PAIR_CONFIGS) ──────────
SLIPPAGE_PIPS = 0.5

# ── Risk management ───────────────────────────────────────────────────────────
ACCOUNT_BALANCE  = 10_000
RISK_PER_TRADE   = 0.015
SL_ATR_MULT      = 1.5
TP_ATR_MULT      = 3.0
MAX_OPEN_TRADES  = 1                # per pair
DAILY_LOSS_LIMIT = 0.03
LOT_SIZE         = 100_000

# ── MT5 connection ────────────────────────────────────────────────────────────
MT5_LOGIN    = 0
MT5_PASSWORD = ""
MT5_SERVER   = ""

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE  = "trades.log"
LOG_LEVEL = "INFO"
