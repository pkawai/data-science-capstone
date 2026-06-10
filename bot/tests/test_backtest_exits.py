# test_backtest_exits.py — the trade simulator must evaluate TP/SL starting
# from the FIRST bar after entry.
#
# Bug: _simulate_trades entered at loop index i0, and the next iteration
# called _check_exit(current_idx=i0+1) which looked at bar i0+2 — so bar
# i0+1 (the most likely bar to hit a 1.0-ATR stop) was never examined.
# Labels check bars i+1..i+horizon and live SL/TP orders are active
# immediately, so the backtest disagreed with both.

import numpy as np
import pandas as pd

import config
from backtest import _simulate_trades
from risk_manager import calculate_sl_tp

SYMBOL = "EURUSD"
N_BARS = 25  # loop scans range(N - TB_HORIZON); horizon exit needs i0+10 in range


def _flat_df(n=N_BARS, price=1.0000, atr=0.0010):
    """Flat synthetic H1 frame; individual bars overridden per scenario."""
    idx = pd.date_range("2026-01-05", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({
        "Datetime": idx,
        "Open":  price,
        "High":  price + 0.0001,
        "Low":   price - 0.0001,
        "Close": price,
        "ATR":   atr,
    })
    return df


def _run(df, signal_at_0):
    predictions = np.ones(len(df), dtype=int)
    predictions[0] = signal_at_0
    confidences = np.full(len(df), 0.9)
    active = np.zeros(len(df), dtype=bool)
    active[0] = True  # exactly one entry, at bar 0
    trades, _ = _simulate_trades(df, predictions, confidences, active, SYMBOL)
    return trades


def test_buy_tp_hit_on_first_bar_after_entry_is_a_tp_exit():
    df = _flat_df()
    # Entry: Buy at close of bar 0 → tp=1.0015, sl=0.9990 (1.5/1.0 × ATR)
    sl, tp = calculate_sl_tp(1.0000, 2, 0.0010, SYMBOL)
    df.loc[1, "High"] = tp + 0.0005   # TP hit on the FIRST bar after entry
    trades = _run(df, signal_at_0=2)

    assert len(trades) == 1
    assert trades[0]["close_reason"] == "TP", (
        "TP touched on the first bar after entry must close the trade — "
        "the simulator skipped that bar and fell through to a HORIZON exit."
    )
    # 15 pips to TP minus (1.5 spread + 0.5 slippage) costs
    assert trades[0]["pnl"] == (tp - 1.0000) / 0.0001 - 2.0


def test_sell_sl_hit_on_first_bar_after_entry_is_an_sl_exit():
    df = _flat_df()
    # Entry: Sell at close of bar 0 → sl=1.0010, tp=0.9985
    sl, tp = calculate_sl_tp(1.0000, 0, 0.0010, SYMBOL)
    df.loc[1, "High"] = sl + 0.0005   # SL hit on the FIRST bar after entry
    trades = _run(df, signal_at_0=0)

    assert len(trades) == 1
    assert trades[0]["close_reason"] == "SL", (
        "SL touched on the first bar after entry must close the trade — "
        "missing it understates losses in every backtest metric."
    )


def test_entry_bar_itself_is_never_exit_checked():
    # The entry happens at the CLOSE of bar 0 — a TP-looking high earlier in
    # that same bar must not count as an exit.
    df = _flat_df()
    df.loc[0, "High"] = 1.0030  # would be above any TP, but it's the entry bar
    trades = _run(df, signal_at_0=2)

    assert len(trades) == 1
    assert trades[0]["close_reason"] == "HORIZON"


def test_horizon_exit_matches_label_horizon():
    # No barrier ever hit → time exit after TB_HORIZON bars, same window the
    # triple-barrier labels use (bars i+1 .. i+TB_HORIZON).
    df = _flat_df()
    trades = _run(df, signal_at_0=2)

    assert len(trades) == 1
    assert trades[0]["close_reason"] == "HORIZON"
