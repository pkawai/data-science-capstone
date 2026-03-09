# data_pipeline.py — Fetch OHLCV data from yfinance (Mac) or MT5 (Windows)

import platform
import pandas as pd

import config


def _is_windows() -> bool:
    return platform.system() == "Windows"


def fetch_historical(symbol: str,
                     source: str = "auto",
                     years: int = config.YEARS_HISTORY) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given symbol.

    Parameters
    ----------
    symbol : e.g. "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"
    source : "auto" | "yfinance" | "mt5"
    years  : years of history to pull
    """
    if source == "auto":
        source = "mt5" if _is_windows() else "yfinance"

    if source == "yfinance":
        return _fetch_yfinance(symbol, years)
    elif source == "mt5":
        return _fetch_mt5_historical(symbol, years)
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'yfinance' or 'mt5'.")


def fetch_latest_candles(symbol: str, n: int = 200) -> pd.DataFrame:
    """
    Fetch the most recent `n` closed H1 candles via MT5 (Windows only).
    """
    if not _is_windows():
        raise EnvironmentError("fetch_latest_candles() requires MT5 on Windows.")
    return _fetch_mt5_latest(symbol, n)


# ── Private helpers ────────────────────────────────────────────────────────────

def _fetch_yfinance(symbol: str, years: int) -> pd.DataFrame:
    import yfinance as yf

    pair_cfg   = config.PAIR_CONFIGS[symbol]
    yf_ticker  = pair_cfg["yf_symbol"]

    ticker = yf.Ticker(yf_ticker)
    df = ticker.history(period="730d", interval=config.TIMEFRAME, auto_adjust=True)

    if df.empty:
        # Fallback: try without =X suffix
        ticker = yf.Ticker(yf_ticker.replace("=X", ""))
        df = ticker.history(period="730d", interval=config.TIMEFRAME, auto_adjust=True)

    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {yf_ticker}. "
                           "Check your internet connection and try again.")

    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    print(f"[data_pipeline] {symbol} yfinance: {len(df):,} candles  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def _fetch_mt5_historical(symbol: str, years: int) -> pd.DataFrame:
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta, timezone

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialize() failed: {mt5.last_error()}")

    mt5_symbol = config.PAIR_CONFIGS[symbol]["mt5_symbol"]
    utc_now    = datetime.now(timezone.utc)
    utc_from   = utc_now - timedelta(days=years * 365)

    rates = mt5.copy_rates_range(mt5_symbol, config.MT5_TIMEFRAME, utc_from, utc_now)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no data for {mt5_symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"time": "Datetime", "open": "Open", "high": "High",
                             "low": "Low", "close": "Close", "tick_volume": "Volume"})
    df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()
    print(f"[data_pipeline] {symbol} MT5: {len(df):,} candles  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def _fetch_mt5_latest(symbol: str, n: int) -> pd.DataFrame:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialize() failed: {mt5.last_error()}")

    mt5_symbol = config.PAIR_CONFIGS[symbol]["mt5_symbol"]
    rates = mt5.copy_rates_from_pos(mt5_symbol, config.MT5_TIMEFRAME, 0, n + 1)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no data for {mt5_symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"time": "Datetime", "open": "Open", "high": "High",
                             "low": "Low", "close": "Close", "tick_volume": "Volume"})
    df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
    df = df.iloc[:-1]   # drop forming candle
    return df.sort_index()
