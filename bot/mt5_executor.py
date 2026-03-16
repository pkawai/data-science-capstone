# mt5_executor.py — MT5 connection and order execution (Windows only)
#
# Prerequisites (Windows):
#   pip install MetaTrader5
#   MetaTrader 5 desktop app must be running and logged in.

import logging

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
    if not MT5_AVAILABLE:
        raise EnvironmentError("MetaTrader5 package is not installed.")
    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialize() failed: {mt5.last_error()}")
    authorized = mt5.login(config.MT5_LOGIN,
                           password=config.MT5_PASSWORD,
                           server=config.MT5_SERVER)
    if not authorized:
        mt5.shutdown()
        raise ConnectionError(f"MT5 login failed: {mt5.last_error()}")
    info = mt5.account_info()
    logger.info(f"Connected to MT5: {info.name} | Balance: {info.balance} {info.currency}")
    return True


def disconnect() -> None:
    if MT5_AVAILABLE:
        mt5.shutdown()
        logger.info("MT5 connection closed.")


# ── Market data ────────────────────────────────────────────────────────────────

def get_latest_candles(symbol: str, n: int = 200):
    """Fetch last `n` closed H1 candles for the given symbol."""
    import pandas as pd
    _require_mt5()
    mt5_symbol = config.PAIR_CONFIGS[symbol]["mt5_symbol"]
    rates = mt5.copy_rates_from_pos(mt5_symbol, config.MT5_TIMEFRAME, 0, n + 1)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no candles for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"time": "Datetime", "open": "Open", "high": "High",
                             "low": "Low", "close": "Close", "tick_volume": "Volume"})
    df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
    df = df.iloc[:-1]  # drop forming candle
    return df.sort_index()


def get_current_price(symbol: str) -> tuple[float, float]:
    """Returns (ask, bid) for the given symbol."""
    _require_mt5()
    mt5_symbol = config.PAIR_CONFIGS[symbol]["mt5_symbol"]
    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is None:
        raise RuntimeError(f"Cannot get tick for {symbol}")
    return tick.ask, tick.bid


# ── Order management ───────────────────────────────────────────────────────────

def place_order(symbol: str,
                direction: int,   # 2=Buy, 0=Sell
                lots: float,
                sl: float,
                tp: float,
                comment: str = "bot") -> int:
    """Place a market order. Returns ticket number on success."""
    _require_mt5()
    mt5_symbol = config.PAIR_CONFIGS[symbol]["mt5_symbol"]
    ask, bid   = get_current_price(symbol)
    order_type = mt5.ORDER_TYPE_BUY  if direction == 2 else mt5.ORDER_TYPE_SELL
    price      = ask                 if direction == 2 else bid

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       mt5_symbol,
        "volume":       lots,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    10,
        "magic":        20260304,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        raise RuntimeError(f"Order failed (retcode={code}): {mt5.last_error()}")

    logger.info(f"Order placed | {symbol} {'BUY' if direction==2 else 'SELL'} "
                f"lots={lots} sl={sl} tp={tp} ticket={result.order}")
    return result.order


def get_open_positions(symbol: str = None) -> list[dict]:
    """
    Returns open positions. If symbol is given, filters to that pair only.
    """
    _require_mt5()
    if symbol:
        mt5_symbol = config.PAIR_CONFIGS[symbol]["mt5_symbol"]
        positions  = mt5.positions_get(symbol=mt5_symbol)
    else:
        positions = mt5.positions_get()

    if positions is None:
        return []
    return [
        {
            "ticket":     p.ticket,
            "symbol":     p.symbol,
            "type":       2 if p.type == 0 else 0,
            "volume":     p.volume,
            "price_open": p.price_open,
            "sl":         p.sl,
            "tp":         p.tp,
            "profit":     p.profit,
            "time_open":  p.time,   # Unix timestamp (seconds)
        }
        for p in positions
    ]


def close_position(ticket: int) -> bool:
    """Close an open position by ticket."""
    _require_mt5()

    # Determine symbol from the open position
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning(f"close_position: ticket {ticket} not found.")
        return False

    p          = positions[0]
    symbol_str = p.symbol
    # Find matching config key
    symbol = next((k for k, v in config.PAIR_CONFIGS.items()
                   if v["mt5_symbol"] == symbol_str), None)
    if symbol is None:
        logger.error(f"Symbol {symbol_str} not in PAIR_CONFIGS.")
        return False

    ask, bid    = get_current_price(symbol)
    close_type  = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
    close_price = bid if p.type == 0 else ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol_str,
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
        logger.error(f"Close failed (retcode={result.retcode if result else 'None'})")
        return False

    logger.info(f"Position {ticket} closed.")
    return True


def modify_position(ticket: int, sl: float, tp: float = None) -> bool:
    """Modify SL (and optionally TP) of an open position."""
    _require_mt5()
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning(f"modify_position: ticket {ticket} not found.")
        return False
    p = positions[0]
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl":       sl,
        "tp":       tp if tp is not None else p.tp,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        logger.error(f"modify_position failed (retcode={code}): {mt5.last_error()}")
        return False
    logger.info(f"Position {ticket} modified: sl={sl}")
    return True


def get_account_balance() -> float:
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
