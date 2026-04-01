# risk_manager.py — Position sizing, SL/TP, and daily loss guard

import config


def calculate_position_size(
    account_balance: float,
    atr: float,
    symbol: str,
    sl_multiplier: float = config.SL_ATR_MULT,
    open_trade_count: int = 0,
) -> float:
    """
    Risk-based position sizing with dynamic scaling (Option A portfolio cap).

    open_trade_count controls how much of RISK_PER_TRADE is used:
      0 open trades → 100% risk  (1.5%)
      1 open trade  →  50% risk  (0.75%)
      2+ open trades→  33% risk  (0.5%)

    Risk $   = account_balance × RISK_PER_TRADE × scale
    SL pips  = sl_multiplier × ATR / pip_size
    Lots     = Risk $ / (SL pips × pip_value_per_lot)
    """
    pip_size      = config.PAIR_CONFIGS[symbol]["pip_size"]
    pip_value_usd = config.PAIR_CONFIGS[symbol]["pip_value_usd"]
    scale_idx     = min(open_trade_count, len(config.RISK_SCALE) - 1)
    scale         = config.RISK_SCALE[scale_idx]
    risk_usd      = account_balance * config.RISK_PER_TRADE * scale
    sl_pips       = (sl_multiplier * atr) / pip_size
    lots          = risk_usd / (sl_pips * pip_value_usd)
    if lots < 0.01:
        return 0.0
    return round(lots, 2)


def calculate_sl_tp(
    entry_price: float,
    direction: int,        # 2=Buy, 0=Sell
    atr: float,
    symbol: str,
    sl_multiplier: float = config.SL_ATR_MULT,
    tp_multiplier: float = config.TP_ATR_MULT,
) -> tuple[float, float]:
    """
    Calculate Stop Loss and Take Profit price levels, rounded per pair decimals.
    """
    decimals    = config.PAIR_CONFIGS[symbol]["decimals"]
    sl_distance = sl_multiplier * atr
    tp_distance = tp_multiplier * atr

    if direction == 2:   # Buy
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:                # Sell
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance

    return round(sl, decimals), round(tp, decimals)


def check_daily_limit(
    daily_pnl_usd: float,
    account_balance: float = config.ACCOUNT_BALANCE,
) -> bool:
    """Returns True if trading should be halted (daily loss limit hit)."""
    return (daily_pnl_usd / account_balance) < -config.DAILY_LOSS_LIMIT
