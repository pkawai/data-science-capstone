#!/usr/bin/env python
# check_feed.py — Is the live MT5 data fresh, and would the bot trade?
#
# Shows current UTC time + how many hours behind each pair's newest candle is,
# so you don't have to do timezone math. A healthy live feed is <2h behind
# during the trading week (the most recent candle is still forming, and we drop
# it, so ~1h behind is normal). Weekends: the feed is closed, so it can be days
# behind — that's expected, not a bug.
#
# Usage (Windows, MT5 open + logged in):
#   python check_feed.py

from datetime import datetime, timezone

import config
import mt5_executor as m
from features import build_features


def main():
    now = datetime.now(timezone.utc)
    m.connect()
    print("=" * 64)
    print(f"  LIVE FEED CHECK   (ADX gate is > {config.ADX_THRESHOLD})")
    print(f"  Current UTC time : {now:%Y-%m-%d %H:%M} UTC   "
          f"(weekday {now.strftime('%a')})")
    print("=" * 64)
    for s in config.PAIRS:
        try:
            raw = m.get_latest_candles(s, 12000)
            df = build_features(raw)
            ts = raw.index[-1].to_pydatetime()
            adx = round(float(df.iloc[-1]["ADX"]), 3)
            bars = len(raw)
            behind_h = (now - ts).total_seconds() / 3600

            # Newest CLOSED candle is always 1-2h old (we drop the forming one),
            # and broker server-time offsets can add an hour, so <=3h = healthy.
            if behind_h <= 3:
                fresh = "FRESH"
            elif now.weekday() >= 5:        # Sat/Sun UTC — market closed
                fresh = "weekend (market closed — normal)"
            elif behind_h <= 8:
                fresh = f"{behind_h:.1f}h behind — re-run in 1h; OK if the time advances"
            else:
                fresh = f"STALE — {behind_h:.1f}h behind! reconnect MT5"

            gate = "PASS (would consider trading)" if adx > config.ADX_THRESHOLD \
                else "ranging (blocked)"
            print(f"  {s}  last candle {ts:%Y-%m-%d %H:%M}  "
                  f"({behind_h:4.1f}h ago, {fresh})")
            print(f"           ADX {adx:<8} {gate}   [{bars} bars]")
        except Exception as e:
            print(f"  {s}   ERROR: {e}")
    print("=" * 64)
    print("  FRESH + a pair says PASS  -> bot will trade when a signal lines up.")
    print("  FRESH + all 'ranging'     -> market is just flat right now (normal).")
    print("  STALE on a weekday        -> MT5 feed dropped; reconnect MetaTrader 5.")
    print("=" * 64)
    m.disconnect()


if __name__ == "__main__":
    main()
