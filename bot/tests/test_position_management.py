# test_position_management.py — stop management must survive a bot restart.
#
# trade_state lives only in memory. After a restart, _manage_open_positions
# re-detects open positions with an empty trade_state and used to re-capture
# "initial_sl" from the CURRENT broker SL. For a position whose SL had already
# been moved to breakeven or trailed into profit, that meant:
#   * SL trailed beyond entry  → initial_risk looked tiny → the breakeven block
#     re-fired and moved the SL BACKWARD to entry (loosening a winning stop);
#   * SL exactly at entry      → an identical-values modify → MT5 NO_CHANGES →
#     treated as failure → breakeven_active never set → trailing permanently
#     dead for that position.
# Detection must instead resume such positions directly in the trailing phase.

import pandas as pd
import pytest

import bot
import mt5_executor as mt5ex

TICKET = 111
ATR = 0.0010


@pytest.fixture
def mt5_stub(monkeypatch):
    """Fake the MT5 surface _manage_open_positions touches; record modifies."""
    calls = {"modify": [], "close": []}

    def fake_modify(ticket, sl, tp=None):
        # Mimic MT5: a request identical to the current position values fails
        # with TRADE_RETCODE_NO_CHANGES. The stub's "current" SL is the last
        # successfully set one (or the position's starting SL).
        if sl == fake_modify.current_sl:
            return False
        calls["modify"].append({"ticket": ticket, "sl": sl, "tp": tp})
        fake_modify.current_sl = sl
        return True

    fake_modify.current_sl = None  # set per-test from the position dict

    monkeypatch.setattr(mt5ex, "get_server_now", lambda symbol=None: 1_750_000_000)
    monkeypatch.setattr(mt5ex, "get_latest_candles", lambda s, n=12000: "raw")
    monkeypatch.setattr(bot, "build_features", lambda raw: pd.DataFrame({"ATR": [ATR]}))
    monkeypatch.setattr(mt5ex, "modify_position", fake_modify)
    monkeypatch.setattr(mt5ex, "close_position",
                        lambda t: calls["close"].append(t) or True)
    return calls, fake_modify


def _pos(direction, price_open, sl, tp):
    return {
        "ticket":     TICKET,
        "symbol":     "EURUSD",
        "type":       direction,          # 2=Buy, 0=Sell
        "price_open": price_open,
        "sl":         sl,
        "tp":         tp,
        "profit":     0.0,
        "time_open":  1_750_000_000 - 3600,   # 1 bar old → no time exit
    }


def test_restart_never_moves_a_trailed_sl_backward(mt5_stub, monkeypatch):
    calls, fake_modify = mt5_stub
    # Buy from 1.1000 whose SL was trailed up to 1.1050 before a restart.
    pos = _pos(direction=2, price_open=1.1000, sl=1.1050, tp=1.1200)
    fake_modify.current_sl = pos["sl"]
    monkeypatch.setattr(mt5ex, "get_current_price", lambda s: (1.1081, 1.1080))

    trade_state = {}
    bot._manage_open_positions([pos], trade_state)

    assert trade_state[TICKET]["breakeven_active"] is True, (
        "A position whose SL is already beyond entry must resume in the "
        "trailing phase after a restart."
    )
    backward = [c for c in calls["modify"] if c["sl"] < pos["sl"]]
    assert backward == [], (
        f"Restart moved a trailed SL backward toward entry: {backward} — "
        "that loosens a stop that had already locked in profit."
    )


def test_restart_with_sl_at_entry_keeps_trailing_alive(mt5_stub, monkeypatch):
    calls, fake_modify = mt5_stub
    # Sell from 1.1000 already at breakeven (SL == entry) before a restart.
    pos = _pos(direction=0, price_open=1.1000, sl=1.1000, tp=1.0900)
    fake_modify.current_sl = pos["sl"]
    # In profit: ask used for sell P&L and trailing
    monkeypatch.setattr(mt5ex, "get_current_price", lambda s: (1.0951, 1.0950))

    trade_state = {}
    bot._manage_open_positions([pos], trade_state)

    assert trade_state[TICKET]["breakeven_active"] is True
    trail = [c for c in calls["modify"] if c["sl"] < pos["sl"]]
    assert trail, (
        "Trailing never ran: the identical-values breakeven modify fails with "
        "NO_CHANGES, so breakeven_active stayed False forever after a restart."
    )
    # Sell trail = ask + TRAIL_ATR_MULT × ATR
    assert trail[0]["sl"] == round(1.0951 + 1.5 * ATR, 5)


def test_fresh_position_with_normal_sl_is_not_treated_as_breakeven(mt5_stub, monkeypatch):
    calls, fake_modify = mt5_stub
    # Fresh Buy, SL still below entry, profit < initial risk → nothing happens.
    pos = _pos(direction=2, price_open=1.1000, sl=1.0990, tp=1.1015)
    fake_modify.current_sl = pos["sl"]
    monkeypatch.setattr(mt5ex, "get_current_price", lambda s: (1.1004, 1.1003))

    trade_state = {}
    bot._manage_open_positions([pos], trade_state)

    assert trade_state[TICKET]["breakeven_active"] is False
    assert calls["modify"] == []
