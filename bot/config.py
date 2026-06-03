# config.py — All tunable settings in one place
# Edit values here; nothing is hardcoded elsewhere.

# ── Pairs ──────────────────────────────────────────────────────────────────────
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]

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
# MT5 encodes hourly frames as 16384 + hours. 16385 = H1, 16388 = H4.
# This was MISLABELED as H1 but set to 16388 (H4) — so the live bot fetched
# 4-hour candles while waking hourly (3 of 4 hours saw an unchanged "frozen"
# bar) AND fed H4 data into an H1-trained model. 16385 is the real H1.
MT5_TIMEFRAME = 16385               # mt5.TIMEFRAME_H1
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
# 20 beats 25 on a 2-yr walk-forward across all 3 pairs (more trades, equal-or-
# higher profit factor). See sweep_thresholds.py. 15 is too loose in choppy years.
ADX_THRESHOLD = 20
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
CONFIDENCE_THRESHOLD = 0.60
# Hard ceiling on each model's stored confidence threshold. Optuna can push the
# per-bundle value to 0.75-0.80 — unreachable for the model (USDJPY max ~0.76),
# so it never trades. model.load() clamps any stored threshold down to this.
# 0.65 chosen via sweep_confidence.py: PF ~2+ on all pairs, ~5-6 trades/wk/pair.
CONFIDENCE_CAP = 0.65

# ── Directional override ───────────────────────────────────────────────────────
# The 3-class model rarely puts >65% on a single class, so argmax+confidence
# collapses to HOLD and the bot never trades (observed: 4 days, 0 trades). With
# this ON, predict_signal ignores the HOLD class and takes the stronger of
# Buy/Sell when it clears DIRECTIONAL_FLOOR. Validated by analyze_hold_bias.py
# (2yr walk-forward): floor 0.50 -> ~1-2 trades/day/pair, profit factor 1.4-1.9.
DIRECTIONAL_OVERRIDE = True
DIRECTIONAL_FLOOR    = 0.50

# ── Walk-forward backtest ─────────────────────────────────────────────────────
TRAIN_MONTHS  = 15
TEST_MONTHS   = 3
STEP_MONTHS   = 3

# ── Backtest costs (defaults — overridden per pair via PAIR_CONFIGS) ──────────
SLIPPAGE_PIPS = 0.5

# ── Risk management ───────────────────────────────────────────────────────────
ACCOUNT_BALANCE  = 5_000
RISK_PER_TRADE   = 0.015
SL_ATR_MULT       = 1.0             # matches TB_SL_MULT (training labels)
TP_ATR_MULT       = 1.5             # matches TB_TP_MULT (training labels)
MAX_OPEN_TRADES   = 1               # per pair
DAILY_LOSS_LIMIT  = 0.03
LOT_SIZE          = 100_000
MAX_BARS_IN_TRADE = 20              # H1 bars before time-based exit (~20 hours)
TRAIL_ATR_MULT    = 1.5             # ATR multiplier for trailing stop distance

# Portfolio risk scaling — Option A: dynamic sizing based on open trade count
# [0 open trades, 1 open trade, 2+ open trades] → multiplier on RISK_PER_TRADE
RISK_SCALE = [1.0, 0.5, 1/3]       # 1.5% / 0.75% / 0.5%

# ── USD direction correlation map ────────────────────────────────────────────
# Used to block contradictory positions across correlated pairs.
# BUY EURUSD/GBPUSD = USD bearish; BUY USDJPY = USD bullish.
USD_DIRECTION_MAP = {
    "EURUSD": {"BUY": "USD_BEAR", "SELL": "USD_BULL"},
    "GBPUSD": {"BUY": "USD_BEAR", "SELL": "USD_BULL"},
    "USDJPY": {"BUY": "USD_BULL", "SELL": "USD_BEAR"},
}

# ── News blackout filter ──────────────────────────────────────────────────────
NEWS_BLACKOUT_MINUTES = 30          # skip trading ±30 min around high-impact news

# ── Meta-labeling ─────────────────────────────────────────────────────────────
META_CONFIDENCE_THRESHOLD = 0.50    # meta-model minimum win probability to trade

# ── Optuna hyperparameter search ──────────────────────────────────────────────
USE_OPTUNA    = True
OPTUNA_TRIALS = 50                  # number of search trials (higher = better but slower)

# ── MT5 connection ────────────────────────────────────────────────────────────
MT5_LOGIN    = 0
MT5_PASSWORD = ""
MT5_SERVER   = ""

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE  = "trades.log"
LOG_LEVEL = "INFO"

# ── Local overrides (auto-loaded; never committed) ─────────────────────────────
# calibrate_floor.py writes calibrated values (e.g. DIRECTIONAL_FLOOR) to
# local_overrides.json on the machine where the live models live. We load it
# LAST so it wins over the defaults above, without editing this file (no git
# conflicts). Safe if the file is absent.
import os as _os, json as _json
_ovr_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "local_overrides.json")
try:
    if _os.path.exists(_ovr_path):
        with open(_ovr_path) as _f:
            for _k, _v in _json.load(_f).items():
                globals()[_k] = _v
        print(f"[config] Applied local_overrides.json: "
              f"{list(_json.load(open(_ovr_path)).keys())}")
except Exception as _e:
    print(f"[config] Could not read local_overrides.json: {_e}")
