# config.py — All tunable settings in one place
# Edit values here; nothing is hardcoded elsewhere.

# ── Data ──────────────────────────────────────────────────────────────────────
SYMBOL        = "EURUSD=X"          # yfinance ticker  (Mac training)
MT5_SYMBOL    = "EURUSD"            # MetaTrader 5 symbol (Windows live)
TIMEFRAME     = "1h"                # yfinance interval
MT5_TIMEFRAME = 16388               # mt5.TIMEFRAME_H1 (avoid MT5 import on Mac)
YEARS_HISTORY = 2                   # how far back to pull data

# ── Trading sessions (UTC hours, inclusive) ───────────────────────────────────
SESSIONS = {
    0: (0,  6),    # Asian
    1: (7,  11),   # London
    2: (12, 16),   # New York
    3: (12, 17),   # London/NY overlap  (actual hours 12-16 overlap; 17 = NY close)
}
ACTIVE_SESSIONS = [1, 2, 3]         # sessions allowed to trade
ACTIVE_HOURS   = set(range(7, 18))  # UTC hours to allow new positions

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
LAG_PERIODS   = [1, 2, 3]          # lags for momentum features

# ── Labeling ──────────────────────────────────────────────────────────────────
LOOKAHEAD      = 5                   # candles forward (used in fixed-horizon labels)
ATR_THRESHOLD  = 0.5                 # label = Buy/Sell if |return| > 0.5 * ATR

# Triple barrier labeling (replaces fixed-horizon)
TB_HORIZON     = 10                  # max candles to wait for SL/TP hit
TB_TP_MULT     = 1.5                 # take-profit = 1.5 × ATR from entry
TB_SL_MULT     = 1.0                 # stop-loss   = 1.0 × ATR from entry

# ── Regime / trade filters ────────────────────────────────────────────────────
ADX_PERIOD     = 14
ADX_THRESHOLD  = 25                  # only trade when ADX > 25 (trending market)
VOL_RATIO_MAX  = 2.0                 # skip if ATR > 2× its 20-period average

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH     = "model.pkl"
XGBOOST_PARAMS = {
    "n_estimators":    500,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":     "mlogloss",
    "random_state":    42,
    "n_jobs":          -1,
}
CONFIDENCE_THRESHOLD = 0.70          # minimum class probability to act

# ── Walk-forward backtest ─────────────────────────────────────────────────────
TRAIN_MONTHS  = 15                   # training window per fold
TEST_MONTHS   = 3                    # out-of-sample test window per fold
STEP_MONTHS   = 3                    # roll-forward step

# ── Backtest costs ────────────────────────────────────────────────────────────
SPREAD_PIPS   = 1.5                  # one-way spread in pips
SLIPPAGE_PIPS = 0.5                  # slippage per fill in pips
PIP_SIZE      = 0.0001               # 1 pip for EURUSD

# ── Risk management ───────────────────────────────────────────────────────────
ACCOUNT_BALANCE   = 10_000           # USD demo account size
RISK_PER_TRADE    = 0.005            # 0.5% account risk per trade
SL_ATR_MULT       = 1.5              # stop loss = 1.5 × ATR
TP_ATR_MULT       = 3.0              # take profit = 3.0 × ATR (2:1 RR)
MAX_OPEN_TRADES   = 1                # maximum concurrent positions
DAILY_LOSS_LIMIT  = 0.03             # stop trading if daily P&L < -3%
LOT_SIZE          = 100_000          # standard lot in units

# ── MT5 connection ────────────────────────────────────────────────────────────
MT5_LOGIN    = 0                     # replace with your demo account number
MT5_PASSWORD = ""                    # replace with your demo password
MT5_SERVER   = ""                    # replace with broker server name

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE  = "trades.log"
LOG_LEVEL = "INFO"
