# data_pipeline.py — Fetch OHLCV data from yfinance (Mac) or MT5 (Windows)

import platform
import pandas as pd

import config


def _is_windows() -> bool:
    return platform.system() == "Windows"


def fetch_historical(source: str = "auto", years: int = config.YEARS_HISTORY) -> pd.DataFrame:
    """
    Fetch historical OHLCV data.

    Parameters
    ----------
    source : "auto" | "yfinance" | "mt5"
        "auto" picks yfinance on Mac/Linux, mt5 on Windows.
    years  : int
        How many years of history to pull.

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume]
    and a DatetimeIndex in UTC.
    """
    if source == "auto":
        source = "mt5" if _is_windows() else "yfinance"

    if source == "yfinance":
        return _fetch_yfinance(years)
    elif source == "mt5":
        return _fetch_mt5_historical(years)
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'yfinance' or 'mt5'.")


def fetch_latest_candles(n: int = 200) -> pd.DataFrame:
    """
    Fetch the most recent `n` closed H1 candles via MT5 (Windows only).
    Used by the live bot loop.
    """
    if not _is_windows():
        raise EnvironmentError("fetch_latest_candles() requires MT5 on Windows.")
    return _fetch_mt5_latest(n)


# ── Private helpers ────────────────────────────────────────────────────────────

def _fetch_yfinance(years: int) -> pd.DataFrame:
    import yfinance as yf

    # yfinance caps hourly forex data at ~730 days; use period= instead of
    # explicit dates to avoid the empty-response bug.
    ticker = yf.Ticker(config.SYMBOL)
    df = ticker.history(
        period="730d",
        interval=config.TIMEFRAME,
        auto_adjust=True,
    )

    # Fallback ticker if primary returns nothing
    if df.empty:
        ticker = yf.Ticker("EUR=X")
        df = ticker.history(period="730d", interval=config.TIMEFRAME, auto_adjust=True)

    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {config.SYMBOL}. "
                           "Check your internet connection and try again.")

    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    print(f"[data_pipeline] yfinance: {len(df):,} candles  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def _fetch_mt5_historical(years: int) -> pd.DataFrame:
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta, timezone

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialize() failed: {mt5.last_error()}")

    utc_now   = datetime.now(timezone.utc)
    utc_from  = utc_now - timedelta(days=years * 365)

    rates = mt5.copy_rates_range(
        config.MT5_SYMBOL,
        config.MT5_TIMEFRAME,
        utc_from,
        utc_now,
    )
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no data: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "time":  "Datetime",
        "open":  "Open",
        "high":  "High",
        "low":   "Low",
        "close": "Close",
        "tick_volume": "Volume",
    })
    df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()
    print(f"[data_pipeline] MT5: {len(df):,} candles  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def _fetch_mt5_latest(n: int) -> pd.DataFrame:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialize() failed: {mt5.last_error()}")

    rates = mt5.copy_rates_from_pos(
        config.MT5_SYMBOL,
        config.MT5_TIMEFRAME,
        0,      # 0 = most recent bar
        n,
    )
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no data: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "time":  "Datetime",
        "open":  "Open",
        "high":  "High",
        "low":   "Low",
        "close": "Close",
        "tick_volume": "Volume",
    })
    df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()
    # Drop the last bar (currently forming)
    df = df.iloc[:-1]
    return df
