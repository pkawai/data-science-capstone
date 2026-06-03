#!/usr/bin/env python
# why_no_trade.py — Show EXACTLY why each pair is or isn't trading right now.
#
# Runs the same 12-gate pipeline as the live bot, but prints a verdict per pair:
# which gate blocked it, and the full Buy/Hold/Sell probability breakdown so a
# "Signal: HOLD 66%" is unambiguous (is Hold the model's pick, or did the meta
# model veto a Buy/Sell?).
#
# Usage (Windows, MT5 open):  python why_no_trade.py

from datetime import datetime, timezone

import numpy as np

import config
import news_calendar
import mt5_executor as mt5ex
from features import build_features, get_feature_columns
from model import load, predict_proba

LABEL = {0: "SELL", 1: "HOLD", 2: "BUY"}


def analyse(symbol, bundle, now_utc):
    raw = mt5ex.get_latest_candles(symbol, 12000)
    df = build_features(raw)
    row = df.iloc[-1]
    adx = float(row.get("ADX", 0))
    vol = float(row.get("Vol_ratio", 1))

    # Full ensemble probabilities (before any filter)
    feat_cols = bundle.get("feature_cols") or get_feature_columns(df)
    avail = [f for f in feat_cols if f in df.columns]
    proba = predict_proba(bundle["primary_models"], df.iloc[[-1]][avail])[0]
    top = int(np.argmax(proba))
    conf = float(proba[top])
    conf_thr = bundle.get("confidence_threshold", config.CONFIDENCE_THRESHOLD)
    meta_thr = bundle.get("meta_threshold", config.META_CONFIDENCE_THRESHOLD)

    # Meta-model win-probability (only meaningful if top is Buy/Sell)
    meta_win = None
    meta_model = bundle.get("meta_model")
    if meta_model is not None and top in (0, 2):
        X_meta = np.concatenate([df.iloc[[-1]][avail].values, proba.reshape(1, -1)], axis=1)
        meta_win = float(meta_model.predict_proba(X_meta)[0][1])

    print(f"\n[{symbol}]  probs -> BUY {proba[2]*100:4.1f}%  "
          f"HOLD {proba[1]*100:4.1f}%  SELL {proba[0]*100:4.1f}%")
    print(f"   model's pick: {LABEL[top]} @ {conf*100:.1f}% "
          f"(needs >= {conf_thr*100:.0f}%)")

    checks = []
    checks.append(("session 07-17 UTC", now_utc.hour in config.ACTIVE_HOURS,
                   f"hour={now_utc.hour:02d} UTC"))
    checks.append(("no news blackout",
                   not news_calendar.is_news_blackout(now_utc, config.NEWS_BLACKOUT_MINUTES),
                   "±%dmin" % config.NEWS_BLACKOUT_MINUTES))
    checks.append((f"ADX > {config.ADX_THRESHOLD}", adx > config.ADX_THRESHOLD,
                   f"ADX={adx:.1f}"))
    checks.append((f"vol < {config.VOL_RATIO_MAX}", vol < config.VOL_RATIO_MAX,
                   f"vol={vol:.2f}"))
    checks.append(("model picks Buy/Sell (not Hold)", top in (0, 2),
                   f"pick={LABEL[top]}"))
    checks.append((f"confidence >= {conf_thr*100:.0f}%", conf >= conf_thr,
                   f"{conf*100:.1f}%"))
    if meta_win is not None:
        checks.append((f"meta >= {meta_thr*100:.0f}%", meta_win >= meta_thr,
                       f"meta={meta_win*100:.1f}%"))

    blocked_by = None
    for name, ok, detail in checks:
        mark = "OK " if ok else "NO "
        print(f"      [{mark}] {name:<32} ({detail})")
        if not ok and blocked_by is None:
            blocked_by = name

    if blocked_by is None:
        print(f"   ==> WOULD TRADE: {LABEL[top]}")
    else:
        print(f"   ==> NO TRADE — blocked by: {blocked_by}")


def main():
    now_utc = datetime.now(timezone.utc)
    mt5ex.connect()
    print("=" * 64)
    print(f"  WHY-NO-TRADE   {now_utc:%Y-%m-%d %H:%M} UTC  "
          f"(active 07-17 UTC = 15:00-01:00 Ulaanbaatar)")
    print("=" * 64)
    for s in config.PAIRS:
        try:
            bundle = load(s)
            analyse(s, bundle, now_utc)
        except Exception as e:
            print(f"\n[{s}]  ERROR: {e}")
    print("\n" + "=" * 64)
    print("  'model's pick: HOLD' = the model itself sees no clean setup (normal).")
    print("  A 'NO' on confidence/meta = a Buy/Sell idea wasn't strong enough.")
    print("=" * 64)
    mt5ex.disconnect()


if __name__ == "__main__":
    main()
