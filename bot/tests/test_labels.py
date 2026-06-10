# test_labels.py — triple-barrier labels must match the geometry the bot
# actually trades, on BOTH sides.
#
# Bug: a single barrier pair (+TB_TP_MULT / −TB_SL_MULT around every bar)
# labeled both directions. "Sell" meant "price fell 1.0×ATR before rising
# 1.5×ATR" — but a live short has TP 1.5×ATR BELOW and SL 1.0×ATR ABOVE, so
# the model was trained to predict an event the bot never trades on the short
# side. The closer down-barrier also made Sell labels structurally easier to
# trigger than Buy labels (class skew by construction, not by the market).
#
# Expected semantics (two-sided):
#   Buy  (2): long trade wins  — +1.5×ATR hit before −1.0×ATR
#   Sell (0): short trade wins — −1.5×ATR hit before +1.0×ATR
#   Hold (1): neither wins (or ambiguous same-bar hit)

import numpy as np
import pandas as pd

import config
from features import create_labels

ATR = 0.0010
ENTRY = 1.0000


def _frame(bars):
    """bars: list of (high, low) for bars 1..k after the entry bar; the rest
    of the frame is flat at ENTRY so only the given bars can hit barriers."""
    n = 15
    highs = [ENTRY] + [h for h, _ in bars] + [ENTRY] * (n - 1 - len(bars))
    lows  = [ENTRY] + [l for _, l in bars] + [ENTRY] * (n - 1 - len(bars))
    idx = pd.date_range("2026-01-05", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "Close": ENTRY, "High": highs, "Low": lows, "ATR": ATR,
    }, index=idx)


def _label_of_bar0(df):
    return create_labels(df, method="triple_barrier").iloc[0]


def test_shallow_drop_is_hold_not_sell():
    # Falls 1.2×ATR (crosses the OLD 1.0 down-barrier, but NOT a short's
    # 1.5×ATR TP), then recovers 1.2×ATR. A real short from this bar loses;
    # labeling it Sell trained the model on phantom wins.
    df = _frame([
        (ENTRY,           ENTRY - 1.2 * ATR),   # bar 1: shallow drop
        (ENTRY + 0.5*ATR, ENTRY - 1.0 * ATR),   # bar 2: drifting back
        (ENTRY + 1.2*ATR, ENTRY - 0.2 * ATR),   # bar 3: shallow rise
    ])
    assert _label_of_bar0(df) == 1, (
        "A 1.2-ATR dip that never reaches a short's 1.5-ATR take-profit must "
        "be Hold — the one-sided labeler called it Sell."
    )


def test_clean_short_win_is_sell():
    # Falls 1.6×ATR before any 1.0×ATR rise → a real short wins.
    df = _frame([(ENTRY, ENTRY - 1.6 * ATR)])
    assert _label_of_bar0(df) == 0


def test_clean_long_win_is_buy():
    # Rises 1.6×ATR before any 1.0×ATR drop → a real long wins.
    df = _frame([(ENTRY + 1.6 * ATR, ENTRY - 0.2 * ATR)])
    assert _label_of_bar0(df) == 2


def test_ambiguous_giant_bar_is_hold():
    # One bar spans both ±1.5×ATR — intra-bar order is unknowable from OHLC,
    # so neither side may claim a win.
    df = _frame([(ENTRY + 1.6 * ATR, ENTRY - 1.6 * ATR)])
    assert _label_of_bar0(df) == 1, (
        "A bar touching both barriers is ambiguous; the old TP-first check "
        "optimistically called it Buy."
    )


def test_last_horizon_rows_are_nan():
    df = _frame([])
    labels = create_labels(df, method="triple_barrier")
    assert labels.iloc[-config.TB_HORIZON:].isna().all()
    assert labels.iloc[:-config.TB_HORIZON].notna().all()
