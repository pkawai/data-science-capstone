#!/usr/bin/env python
# check_feed.py — Is the live MT5 data actually advancing, or frozen?
#
# Prints, for each pair, the newest closed candle's timestamp + ADX (3 dp).
# Run it now, then again in ~1 hour:
#   - timestamp jumps by 1h  -> data is LIVE (ADX is just slow, not frozen)
#   - timestamp does NOT move -> MT5 lost its broker connection (reconnect MT5)
#
# Usage (Windows, MT5 open + logged in):
#   python check_feed.py

import config
import mt5_executor as m
from features import build_features


def main():
    m.connect()
    print("=" * 60)
    print(f"  LIVE FEED CHECK   (ADX gate is > {config.ADX_THRESHOLD})")
    print("=" * 60)
    for s in config.PAIRS:
        try:
            raw = m.get_latest_candles(s, 12000)
            df = build_features(raw)
            ts = raw.index[-1]
            adx = round(float(df.iloc[-1]["ADX"]), 3)
            bars = len(raw)
            status = "PASS (would consider trading)" if adx > config.ADX_THRESHOLD \
                else "ranging (blocked)"
            print(f"  {s}   last candle: {ts}   ADX: {adx:<8} {status}   [{bars} bars]")
        except Exception as e:
            print(f"  {s}   ERROR: {e}")
    print("=" * 60)
    print("  Run again in ~1 hour. If 'last candle' time moves up by 1h,")
    print("  the feed is LIVE. If it's stuck, reconnect MetaTrader 5.")
    print("=" * 60)
    m.disconnect()


if __name__ == "__main__":
    main()
