#!/usr/bin/env python
# check_tick.py — Pinpoint a "GUI live but Python stale" MT5 feed problem.
#
# For each pair it prints THREE clocks:
#   - now (UTC)          : your computer's UTC time
#   - last TICK time     : the live quote MT5 has (should be within seconds)
#   - last CANDLE time   : the newest closed H1 bar the bot uses
#
# If the TICK is current but the CANDLE is hours old, the terminal isn't
# syncing H1 history for the Python API. check_feed.py / bot.py now force
# symbol_select() before fetching, which fixes that. This tool confirms it.
#
# Usage (Windows, MT5 open + logged in):
#   python check_tick.py

from datetime import datetime, timezone

import config
import mt5_executor as m

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


def main():
    if mt5 is None:
        print("MetaTrader5 package not installed — run this on Windows.")
        return

    m.connect()
    now = datetime.now(timezone.utc)
    print("=" * 70)
    print(f"  TICK vs CANDLE CHECK     now: {now:%Y-%m-%d %H:%M:%S} UTC")
    print("=" * 70)

    for s in config.PAIRS:
        mt5_symbol = config.PAIR_CONFIGS[s]["mt5_symbol"]
        try:
            mt5.symbol_select(mt5_symbol, True)        # subscribe + sync
            tick = mt5.symbol_info_tick(mt5_symbol)
            tick_t = datetime.fromtimestamp(tick.time, tz=timezone.utc)

            raw = m.get_latest_candles(s, 200)
            candle_t = raw.index[-1].to_pydatetime()

            tick_age = (now - tick_t).total_seconds()
            candle_age = (now - candle_t).total_seconds() / 3600

            tick_ok = "LIVE" if tick_age < 120 else f"{tick_age:.0f}s old (?)"
            print(f"  {s}")
            print(f"     last TICK   : {tick_t:%H:%M:%S}   ({tick_ok}, bid={tick.bid})")
            print(f"     last CANDLE : {candle_t:%Y-%m-%d %H:%M}   ({candle_age:.1f}h old)")
        except Exception as e:
            print(f"  {s}  ERROR: {e}")

    print("=" * 70)
    print("  TICK live + CANDLE current (<3h) -> feed fully healthy.")
    print("  TICK live + CANDLE stale (hours) -> history-sync issue; the")
    print("    symbol_select() fix should clear it. Re-run after pulling.")
    print("=" * 70)
    m.disconnect()


if __name__ == "__main__":
    main()
