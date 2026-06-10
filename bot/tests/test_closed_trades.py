# test_closed_trades.py — realised-P&L logging.
#
# trades.csv records ENTRIES only, so real win-rate/profit-factor was
# unmeasurable from live data. Each cycle the bot now pulls the bot's exit
# deals (closed positions) from MT5 history and appends them to
# closed_trades.csv, deduplicated by deal ticket so restarts/overlapping
# windows never double-log.

from types import SimpleNamespace

import pytest

import bot
import mt5_executor as mt5ex


def _deal(ticket, profit=10.0, reason="TP"):
    return {
        "deal_ticket":     ticket,
        "position_ticket": 1000 + ticket,
        "close_time":      "2026-06-10T08:00:00+00:00",
        "symbol":          "EURUSD",
        "volume":          0.10,
        "close_price":     1.1010,
        "profit":          profit,
        "swap":            0.0,
        "commission":      -0.7,
        "reason":          reason,
    }


def test_append_writes_header_and_rows(tmp_path):
    path = tmp_path / "closed_trades.csv"
    n = bot._append_closed_trades([_deal(1), _deal(2)], path=str(path))
    assert n == 2
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3                       # header + 2 rows
    assert lines[0].startswith("close_time,")    # header present


def test_append_dedups_by_deal_ticket(tmp_path):
    path = tmp_path / "closed_trades.csv"
    bot._append_closed_trades([_deal(1), _deal(2)], path=str(path))
    # Overlapping fetch window re-returns deal 2, plus a genuinely new deal 3.
    n = bot._append_closed_trades([_deal(2), _deal(3)], path=str(path))
    assert n == 1, "already-logged deals must not be re-appended"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 4                       # header + deals 1, 2, 3


def test_get_closed_trades_filters_bot_exit_deals(monkeypatch):
    """Only THIS bot's deals (magic match) that CLOSE a position (entry=OUT)
    may be returned — manual trades and entry deals must be filtered out."""
    OUT, IN = 1, 0   # mt5.DEAL_ENTRY_OUT / DEAL_ENTRY_IN

    def fake_deal(ticket, magic, entry, reason=5):
        return SimpleNamespace(
            ticket=ticket, position_id=1000 + ticket, time=1_750_000_000,
            symbol="EURUSD", volume=0.1, price=1.1010, profit=12.5,
            swap=0.0, commission=-0.7, reason=reason, magic=magic, entry=entry,
        )

    fake_mt5 = SimpleNamespace(
        DEAL_ENTRY_OUT=OUT,
        history_deals_get=lambda frm, to: (
            fake_deal(1, magic=mt5ex.MAGIC, entry=OUT),       # ours, exit  ✓
            fake_deal(2, magic=mt5ex.MAGIC, entry=IN),        # ours, entry ✗
            fake_deal(3, magic=0,           entry=OUT),       # manual      ✗
        ),
        terminal_info=lambda: True,
    )
    monkeypatch.setattr(mt5ex, "MT5_AVAILABLE", True)
    monkeypatch.setattr(mt5ex, "mt5", fake_mt5, raising=False)

    deals = mt5ex.get_closed_trades(days=7)
    assert [d["deal_ticket"] for d in deals] == [1]
    assert deals[0]["profit"] == 12.5
    assert deals[0]["reason"] == "TP"            # DEAL_REASON 5 → "TP"
    assert deals[0]["symbol"] == "EURUSD"
