#!/usr/bin/env python
# mt5_doctor.py — Full MT5 connection diagnosis. Run this when data looks stale.
#
# Answers, in plain language:
#   1. Is the Python API actually CONNECTED to the broker? (the usual culprit)
#   2. What account / server is Python attached to? (is it even your account?)
#   3. Is the live TICK current? (in broker time AND your local time)
#   4. Is the latest CANDLE current?
#   5. Is the candle stale vs the tick? (history-sync problem)
#
# Key insight this tool encodes: MT5 timestamps are in BROKER SERVER TIME,
# not UTC. We measure staleness as (server_now - data_time) using the broker's
# own clock, so timezone never confuses the verdict.
#
# Usage (Windows, MT5 open):  python mt5_doctor.py

from datetime import datetime, timezone

import config

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


def main():
    if mt5 is None:
        print("MetaTrader5 package not installed — run this on Windows.")
        return

    print("=" * 72)
    print("  MT5 DOCTOR")
    print("=" * 72)

    if not mt5.initialize():
        print(f"  [FAIL] mt5.initialize() failed: {mt5.last_error()}")
        print("         -> MT5 desktop app isn't running, or Python can't attach.")
        return
    print("  [ok]  mt5.initialize() succeeded")

    # ── 0. Timeframe sanity (MT5 encodes hourly frames as 16384 + hours) ─────
    tf = config.MT5_TIMEFRAME
    tf_label = f"H{tf - 16384}" if 16385 <= tf <= 16400 else f"code {tf}"
    tf_ok = (tf == 16385)
    print(f"  [{'ok' if tf_ok else 'FAIL'}]  Timeframe: MT5_TIMEFRAME={tf} -> {tf_label}"
          + ("" if tf_ok else "  <-- NOT H1! candles will look 'frozen' between closes"))

    # ── 1. Terminal link state (THE key check) ──────────────────────────────
    term = mt5.terminal_info()
    if term is None:
        print("  [FAIL] terminal_info() is None — cannot read terminal state.")
        mt5.shutdown(); return
    print(f"  [{'ok' if term.connected else 'FAIL'}]  Broker link connected: {term.connected}")
    if not term.connected:
        print("         ^^^ THIS is the problem. The Python/terminal API link is")
        print("             NOT connected to the trade server, so all data is stale")
        print("             even though the GUI may tick. Fix: log into your account")
        print("             in MT5 (bottom-right must be green), or set")
        print("             MT5_LOGIN/MT5_PASSWORD/MT5_SERVER in config.py.")

    # ── 2. Which account is Python attached to? ──────────────────────────────
    acct = mt5.account_info()
    if acct is None:
        print("  [FAIL] account_info() is None — no account logged in for the API.")
    else:
        print(f"  [ok]  Account: {acct.login}  |  server: {acct.server}  |  "
              f"{acct.name}  |  bal: {acct.balance} {acct.currency}  |  "
              f"trade_allowed: {acct.trade_allowed}")

    # ── 3+4. Tick vs candle, measured in BROKER time ─────────────────────────
    local_now = datetime.now()                       # your wall clock
    utc_now = datetime.now(timezone.utc)
    print("-" * 72)
    print(f"  Your local time : {local_now:%Y-%m-%d %H:%M:%S}")
    print(f"  UTC time        : {utc_now:%Y-%m-%d %H:%M:%S}")
    print("-" * 72)

    server_now = None
    for s in config.PAIRS:
        sym = config.PAIR_CONFIGS[s]["mt5_symbol"]
        mt5.symbol_select(sym, True)
        tick = mt5.symbol_info_tick(sym)
        rates = mt5.copy_rates_from_pos(sym, config.MT5_TIMEFRAME, 0, 2)

        if tick is None or tick.time == 0:
            print(f"  {s}: no tick (symbol not available / not subscribed)")
            continue

        # The tick time IS the broker's 'now' (last quote). Use it as server clock.
        tick_srv = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        server_now = tick_srv
        # Broker offset vs UTC (e.g. +0200 / +0300) — informational.
        offset_h = round((tick_srv - utc_now.replace(tzinfo=timezone.utc)).total_seconds() / 3600)

        if rates is not None and len(rates):
            candle_srv = datetime.fromtimestamp(rates[-1]["time"], tz=timezone.utc)
            # Stale measured in broker's OWN clock: tick_now - candle_time.
            candle_lag_h = (tick_srv - candle_srv).total_seconds() / 3600
            verdict = "FRESH" if candle_lag_h <= 2.5 else f"STALE ({candle_lag_h:.1f}h behind last tick)"
        else:
            candle_srv, verdict = None, "no candles returned"

        print(f"  {s}  (broker clock ~UTC{offset_h:+d})")
        print(f"     last TICK   : {tick_srv:%Y-%m-%d %H:%M:%S} (broker)   bid={tick.bid}")
        if candle_srv:
            print(f"     last CANDLE : {candle_srv:%Y-%m-%d %H:%M} (broker)   {verdict}")

    print("=" * 72)
    print("  HOW TO READ THIS:")
    print("   - 'Broker link connected: False'  -> reconnect/login in MT5. Root cause.")
    print("   - link True + TICK old             -> terminal not receiving quotes.")
    print("   - TICK current + CANDLE stale      -> history sync issue (symbol_select")
    print("                                          fix handles it; re-run bot).")
    print("   - TICK current + CANDLE current    -> feed healthy; bot will trade.")
    print("=" * 72)
    mt5.shutdown()


if __name__ == "__main__":
    main()
