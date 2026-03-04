# mt5_executor.py — MT5 connection and order execution (Windows only)
#
# Copy the entire bot/ folder to your Windows PC that has MetaTrader 5 installed.
# Run `python bot.py` there — this module handles all broker communication.
#
# Prerequisites (Windows):
#   pip install MetaTrader5
#   MetaTrader 5 desktop app must be running and logged in.

import time
import logging
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not installed — mt5_executor is disabled.")


# ── Connection ─────────────────────────────────────────────────────────────────

def connect() -> bool:
    """
    Initialize and log in to MT5.
    Returns True on success, raises ConnectionError on failure.
    """
    if not MT5_AVAILABLE:
        raise EnvironmentError("MetaTrader5 package is not installed.")

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialize() failed: {mt5.last_error()}")

    authorized = mt5.login(
        config.MT5_LOGIN,
        password=config.MT5_PASSWORD,
        server=config.MT5_SERVER,
    )
    if not authorized:
        mt5.shutdown()
        raise ConnectionError(f"MT5 login failed: {mt5.last_error()}")

    info = mt5.account_info()
    logger.info(f"Connected to MT5: {info.name} | "
                f"Balance: {info.balance} {info.currency}")
    return True


def disconnect() -> None:
    if MT5_AVAILABLE:
        mt5.shutdown()
        logger.info("MT5 connection closed.")


# ── Market data ────────────────────────────────────────────────────────────────

def get_latest_candles(n: int = 200) -> pd.DataFrame:
    """
    Fetch the last `n` closed H1 candles.
    The currently-forming candle (index 0) is excluded.
    """
    _require_mt5()
    rates = mt5.copy_rates_from_pos(
        config.MT5_SYMBOL,
        config.MT5_TIMEFRAME,
        0,
        n + 1,   # +1 to drop the forming candle
    )
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no candles: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "time": "Datetime", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "tick_volume": "Volume",
    })
    df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
    df = df.iloc[:-1]  # drop forming candle
    return df.sort_index()


def get_current_price() -> tuple[float, float]:
    """Returns (ask, bid) for the configured symbol."""
    _require_mt5()
    tick = mt5.symbol_info_tick(config.MT5_SYMBOL)
    if tick is None:
        raise RuntimeError(f"Cannot get tick for {config.MT5_SYMBOL}")
    return tick.ask, tick.bid


# ── Order management ───────────────────────────────────────────────────────────

def place_order(
    direction: int,   # 2=Buy, 0=Sell
    lots: float,
    sl: float,
    tp: float,
    comment: str = "bot",
) -> Optional[int]:
    """
    Place a market order.

    Returns
    -------
    ticket : int order ticket on success, raises RuntimeError on failure.
    """
    _require_mt5()

    ask, bid = get_current_price()
    order_type = mt5.ORDER_TYPE_BUY if direction == 2 else mt5.ORDER_TYPE_SELL
    price      = ask                if direction == 2 else bid

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    config.MT5_SYMBOL,
        "volume":    lots,
        "type":      order_type,
        "price":     price,
        "sl":        sl,
        "tp":        tp,
        "deviation": 10,   # max price slippage in points
        "magic":     20260304,
        "comment":   comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        raise RuntimeError(f"Order failed (retcode={code}): {mt5.last_error()}")

    logger.info(f"Order placed | ticket={result.order}  "
                f"{'BUY' if direction==2 else 'SELL'}  "
                f"lots={lots}  sl={sl}  tp={tp}")
    return result.order


def get_open_positions() -> list[dict]:
    """
    Returns a list of open position dicts for the configured symbol.
    Each dict has keys: ticket, type (2=Buy/0=Sell), volume, price_open, sl, tp.
    """
    _require_mt5()
    positions = mt5.positions_get(symbol=config.MT5_SYMBOL)
    if positions is None:
        return []
    return [
        {
            "ticket":     p.ticket,
            "type":       2 if p.type == 0 else 0,   # MT5: 0=BUY,1=SELL → our 2,0
            "volume":     p.volume,
            "price_open": p.price_open,
            "sl":         p.sl,
            "tp":         p.tp,
            "profit":     p.profit,
        }
        for p in positions
    ]


def close_position(ticket: int) -> bool:
    """
    Close an open position by ticket.
    Returns True on success.
    """
    _require_mt5()
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning(f"close_position: ticket {ticket} not found.")
        return False

    p = positions[0]
    ask, bid = get_current_price()
    close_type  = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
    close_price = bid if p.type == 0 else ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       config.MT5_SYMBOL,
        "volume":       p.volume,
        "type":         close_type,
        "position":     ticket,
        "price":        close_price,
        "deviation":    10,
        "magic":        20260304,
        "comment":      "bot_close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        logger.error(f"Close failed (retcode={code}): {mt5.last_error()}")
        return False

    logger.info(f"Position {ticket} closed.")
    return True


def get_account_balance() -> float:
    """Returns the current account balance in account currency."""
    _require_mt5()
    info = mt5.account_info()
    if info is None:
        raise RuntimeError("Cannot retrieve account info.")
    return info.balance


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_mt5():
    if not MT5_AVAILABLE:
        raise EnvironmentError("MetaTrader5 package is not installed.")
    if not mt5.terminal_info():
        raise ConnectionError("MT5 is not initialized. Call connect() first.")
