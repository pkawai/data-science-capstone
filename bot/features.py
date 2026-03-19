# features.py — Feature engineering + labeling

import numpy as np
import pandas as pd

import config


# ── Public API ─────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators and derived features to raw OHLCV."""
    df = df.copy()
    df = _add_core_indicators(df)
    df = _add_adx(df)
    df = _add_h4_trend(df)
    df = _add_d1_trend(df)
    df = _add_vol_ratio(df)
    df = _add_lagged_features(df)
    df = _add_session_feature(df)
    df = _add_time_features(df)
    df = _add_normalized_distance(df)
    df = _add_candle_features(df)
    df = _add_rsi_slope(df)
    df = df.dropna()
    return df


def create_labels(df: pd.DataFrame, method: str = "triple_barrier") -> pd.Series:
    """
    Create Buy (2) / Hold (1) / Sell (0) labels.

    method='triple_barrier'  — hits TP or SL first within TB_HORIZON candles (recommended)
    method='atr'             — fixed lookahead with ATR-scaled threshold
    """
    if method == "triple_barrier":
        return _triple_barrier_labels(df)
    return _atr_labels(df)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All feature column names — excludes raw OHLCV + Signal."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "Signal"}
    return [c for c in df.columns if c not in exclude]


# ── Labeling ───────────────────────────────────────────────────────────────────

def _triple_barrier_labels(df: pd.DataFrame) -> pd.Series:
    """
    For each candle, look forward up to TB_HORIZON candles.
    If price hits TB_TP_MULT*ATR above entry first → Buy (2)
    If price hits TB_SL_MULT*ATR below entry first → Sell (0)
    If neither hits within horizon               → Hold (1)
    """
    labels = np.ones(len(df), dtype=float)   # default Hold
    closes = df["Close"].values
    atrs   = df["ATR"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    horizon = config.TB_HORIZON
    n = len(df)

    for i in range(n - horizon):
        entry = closes[i]
        atr   = atrs[i]
        tp    = entry + config.TB_TP_MULT * atr
        sl    = entry - config.TB_SL_MULT * atr

        for j in range(1, horizon + 1):
            if highs[i + j] >= tp:
                labels[i] = 2   # Buy
                break
            if lows[i + j] <= sl:
                labels[i] = 0   # Sell
                break

    # Last `horizon` rows have no valid label
    labels[n - horizon:] = np.nan
    return pd.Series(labels, index=df.index, name="Signal")


def _atr_labels(df: pd.DataFrame) -> pd.Series:
    lookahead = config.LOOKAHEAD
    future_close = df["Close"].shift(-lookahead)
    ret = (future_close - df["Close"]) / df["Close"]
    threshold = config.ATR_THRESHOLD * df["ATR"] / df["Close"]

    labels = pd.Series(1, index=df.index, name="Signal")
    labels[ret >  threshold] = 2
    labels[ret < -threshold] = 0
    labels.iloc[-lookahead:] = np.nan
    return labels


# ── Core indicators ────────────────────────────────────────────────────────────

def _add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # RSI
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / config.RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / config.RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD
    ema_fast        = df["Close"].ewm(span=config.MACD_FAST, adjust=False).mean()
    ema_slow        = df["Close"].ewm(span=config.MACD_SLOW, adjust=False).mean()
    df["MACD"]      = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    ma             = df["Close"].rolling(config.BB_PERIOD).mean()
    std            = df["Close"].rolling(config.BB_PERIOD).std()
    df["BB_upper"] = ma + config.BB_STD * std
    df["BB_lower"] = ma - config.BB_STD * std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / ma
    df["BB_pct"]   = (df["Close"] - df["BB_lower"]) / (
        df["BB_upper"] - df["BB_lower"]
    ).replace(0, np.nan)

    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(config.ATR_PERIOD).mean()

    # Moving averages
    df["MA_20"]    = df["Close"].rolling(config.MA_SHORT).mean()
    df["MA_50"]    = df["Close"].rolling(config.MA_LONG).mean()
    df["MA_cross"] = (df["MA_20"] - df["MA_50"]) / df["ATR"]

    # Return
    df["Return_1h"] = df["Close"].pct_change()

    return df


# ── New features ───────────────────────────────────────────────────────────────

def _add_adx(df: pd.DataFrame) -> pd.DataFrame:
    """ADX (Average Directional Index) — measures trend strength."""
    period = config.ADX_PERIOD
    high, low, close = df["High"], df["Low"], df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move   = high.diff()
    down_move = (-low.diff())
    plus_dm   = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_s    = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    df["ADX"]      = dx.ewm(alpha=1 / period, adjust=False).mean()
    df["Plus_DI"]  = plus_di
    df["Minus_DI"] = minus_di
    return df


def _add_h4_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    H4 trend: resample H1 closes to 4H, compute EMA(20).
    1 = H4 bullish (close > ema), 0 = bearish.
    Forward-filled back to H1 index — no look-ahead bias.
    """
    h4_close = df["Close"].resample("4h").last()
    h4_ema   = h4_close.ewm(span=20, adjust=False).mean()
    h4_trend = (h4_close > h4_ema).astype(float)
    h4_rsi_delta = h4_close.diff(3) / h4_close.shift(3)   # 3-bar H4 momentum

    df["H4_trend"]     = h4_trend.reindex(df.index, method="ffill")
    df["H4_momentum"]  = h4_rsi_delta.reindex(df.index, method="ffill")
    return df


def _add_vol_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """ATR / 20-period ATR average — detects volatility spikes."""
    df["Vol_ratio"] = df["ATR"] / df["ATR"].rolling(20).mean().replace(0, np.nan)
    return df


def _add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in config.LAG_PERIODS:
        df[f"RSI_lag{lag}"]       = df["RSI"].shift(lag)
        df[f"Return_lag{lag}"]    = df["Return_1h"].shift(lag)
        df[f"MACD_hist_lag{lag}"] = df["MACD_hist"].shift(lag)
    return df


def _add_session_feature(df: pd.DataFrame) -> pd.DataFrame:
    hour    = df.index.hour
    session = np.zeros(len(df), dtype=int)
    session[(hour >= 7)  & (hour < 12)] = 1   # London
    session[(hour >= 12) & (hour < 17)] = 3   # London/NY overlap
    session[(hour >= 17) & (hour < 22)] = 2   # NY only
    df["Session"] = session
    return df


def _add_normalized_distance(df: pd.DataFrame) -> pd.DataFrame:
    df["Price_MA_dist"] = (df["Close"] - df["MA_20"]) / df["ATR"].replace(0, np.nan)
    return df


def _add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    atr = df["ATR"].replace(0, np.nan)
    df["Body_ratio"]     = (df["Close"] - df["Open"]).abs() / atr
    upper_wick           = df["High"] - df[["Close", "Open"]].max(axis=1)
    lower_wick           = df[["Close", "Open"]].min(axis=1) - df["Low"]
    df["Wick_imbalance"] = (upper_wick - lower_wick) / atr
    return df


def _add_rsi_slope(df: pd.DataFrame) -> pd.DataFrame:
    df["RSI_slope"] = df["RSI"] - df["RSI"].shift(3)
    return df


def _add_d1_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily (D1) 200-EMA trend — the most-watched institutional reference.
    D1_trend   : 1 if price > D1 EMA200 (bullish), 0 if bearish.
    D1_EMA200_dist : (close - D1 EMA200) / ATR — normalised distance.
    Forward-filled back to H1 to avoid look-ahead bias.
    """
    d1_close  = df["Close"].resample("1D").last()
    d1_ema200 = d1_close.ewm(span=200, adjust=False).mean()
    d1_trend  = (d1_close > d1_ema200).astype(float)
    d1_dist   = d1_close - d1_ema200

    df["D1_trend"]       = d1_trend.reindex(df.index, method="ffill")
    d1_dist_h1           = d1_dist.reindex(df.index, method="ffill")
    df["D1_EMA200_dist"] = d1_dist_h1 / df["ATR"].replace(0, np.nan)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical hour encoding + day-of-week flags.
    These let the model learn time-of-day and day-of-week patterns.
    """
    hour = df.index.hour
    dow  = df.index.dayofweek          # 0=Monday … 4=Friday
    df["Hour_sin"]    = np.sin(2 * np.pi * hour / 24)
    df["Hour_cos"]    = np.cos(2 * np.pi * hour / 24)
    df["Day_of_week"] = dow.astype(float)
    df["Is_monday"]   = (dow == 0).astype(float)
    df["Is_friday"]   = (dow == 4).astype(float)
    return df
