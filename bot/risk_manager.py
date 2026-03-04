# risk_manager.py — Position sizing, SL/TP, and daily loss guard

import config


def calculate_position_size(
    account_balance: float,
    atr: float,
    sl_multiplier: float = config.SL_ATR_MULT,
) -> float:
    """
    Risk-based position sizing.

    Risk $   = account_balance × RISK_PER_TRADE
    SL pips  = sl_multiplier × ATR / PIP_SIZE
    Lot size = Risk $ / (SL pips × pip_value_per_lot)

    For EURUSD standard lot (100,000 units):
        pip_value_per_lot ≈ $10 (USD account)

    Returns lot size rounded to 2 decimal places (minimum 0.01).
    """
    risk_usd       = account_balance * config.RISK_PER_TRADE
    sl_pips        = (sl_multiplier * atr) / config.PIP_SIZE
    pip_value      = config.LOT_SIZE * config.PIP_SIZE   # ≈ $10 per standard lot for EURUSD/USD
    lots           = risk_usd / (sl_pips * pip_value)

    lots = round(max(lots, 0.01), 2)
    return lots


def calculate_sl_tp(
    entry_price: float,
    direction: int,        # 2=Buy, 0=Sell
    atr: float,
    sl_multiplier: float = config.SL_ATR_MULT,
    tp_multiplier: float = config.TP_ATR_MULT,
) -> tuple[float, float]:
    """
    Calculate Stop Loss and Take Profit price levels.

    Returns
    -------
    (sl_price, tp_price)
    """
    sl_distance = sl_multiplier * atr
    tp_distance = tp_multiplier * atr

    if direction == 2:   # Buy
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:                # Sell
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance

    return round(sl, 5), round(tp, 5)


def check_daily_limit(
    daily_pnl_usd: float,
    account_balance: float = config.ACCOUNT_BALANCE,
) -> bool:
    """
    Returns True if trading should be halted (daily loss limit hit).

    Parameters
    ----------
    daily_pnl_usd : cumulative P&L for the current trading day (negative = loss)
    """
    drawdown_pct = daily_pnl_usd / account_balance
    if drawdown_pct < -config.DAILY_LOSS_LIMIT:
        return True   # halt trading
    return False
