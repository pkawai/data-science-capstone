#!/usr/bin/env python
# replay_live.py — Replay the EXACT live decision path over recent H1 data and
# count how often the bot WOULD trade. Answers "is 2 days no-trade normal, or is
# a gate strangling it?" Works on Mac (yfinance) or Windows (MT5).
#
# It runs the real predict_signal() (ensemble + confidence + meta veto) plus the
# session/ADX/vol/news gates, exactly like bot.py, on each of the last N days of
# H1 bars — and tallies where trades die.
#
# Usage:  python replay_live.py [days]   (default 30)

import sys
import warnings
warnings.filterwarnings("ignore")

from datetime import timezone

import numpy as np
import pandas as pd

import config
import news_calendar
from features import build_features, get_feature_columns
from model import load, predict_proba

LABEL = {0: "SELL", 1: "HOLD", 2: "BUY"}
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def _fetch(symbol):
    import yfinance as yf
    yf_sym = config.PAIR_CONFIGS[symbol]["yf_symbol"]
    df = yf.Ticker(yf_sym).history(period="700d", interval="1h", auto_adjust=True)
    if df.empty:
        df = yf.Ticker(yf_sym.replace("=X", "")).history(period="700d", interval="1h", auto_adjust=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df[~df.index.duplicated(keep="last")].sort_index()


def replay(symbol):
    bundle = load(symbol)
    raw = _fetch(symbol)
    df = build_features(raw)

    cutoff = df.index[-1] - pd.Timedelta(days=DAYS)
    recent = df[df.index >= cutoff]

    feat_cols = bundle.get("feature_cols") or get_feature_columns(df)
    avail = [f for f in feat_cols if f in df.columns]
    conf_thr = bundle.get("confidence_threshold", config.CONFIDENCE_THRESHOLD)
    meta_thr = bundle.get("meta_threshold", config.META_CONFIDENCE_THRESHOLD)
    meta_model = bundle.get("meta_model")

    # Batch ensemble probabilities for all recent bars
    proba = predict_proba(bundle["primary_models"], recent[avail])
    picks = proba.argmax(axis=1)
    confs = proba.max(axis=1)

    n = len(recent)
    tally = dict(total=n, off_session=0, news=0, adx=0, vol=0,
                 hold=0, low_conf=0, meta_veto=0, WOULD_TRADE=0)
    conf_when_dir = []   # confidence when model picked Buy/Sell

    for i, (ts, r) in enumerate(recent.iterrows()):
        if ts.hour not in config.ACTIVE_HOURS:
            tally["off_session"] += 1; continue
        if news_calendar.is_news_blackout(ts.to_pydatetime(), config.NEWS_BLACKOUT_MINUTES):
            tally["news"] += 1; continue
        if r.get("ADX", 0) <= config.ADX_THRESHOLD:
            tally["adx"] += 1; continue
        if r.get("Vol_ratio", 1) >= config.VOL_RATIO_MAX:
            tally["vol"] += 1; continue

        pick, conf = int(picks[i]), float(confs[i])
        if pick == 1:
            tally["hold"] += 1; continue
        conf_when_dir.append(conf)
        if conf < conf_thr:
            tally["low_conf"] += 1; continue
        if meta_model is not None:
            X_meta = np.concatenate([recent.iloc[[i]][avail].values, proba[i].reshape(1, -1)], axis=1)
            if float(meta_model.predict_proba(X_meta)[0][1]) < meta_thr:
                tally["meta_veto"] += 1; continue
        tally["WOULD_TRADE"] += 1

    print(f"\n[{symbol}]  last {DAYS}d = {n} H1 bars   "
          f"(conf_thr={conf_thr:.2f}, meta_thr={meta_thr:.2f})")
    active = n - tally["off_session"]
    print(f"   in-session bars: {active}")
    print(f"   died at: ADX {tally['adx']}, vol {tally['vol']}, "
          f"model=HOLD {tally['hold']}, low_conf {tally['low_conf']}, "
          f"meta_veto {tally['meta_veto']}, news {tally['news']}")
    if conf_when_dir:
        cd = np.array(conf_when_dir)
        print(f"   when model picked a DIRECTION ({len(cd)}x): "
              f"conf min/median/max = {cd.min()*100:.0f}/{np.median(cd)*100:.0f}/{cd.max()*100:.0f}%")
    print(f"   ==> WOULD TRADE {tally['WOULD_TRADE']}x in {DAYS} days "
          f"(~{tally['WOULD_TRADE']/DAYS:.2f}/day)")
    return tally


def main():
    print("=" * 66)
    print(f"  LIVE-PATH REPLAY — last {DAYS} days, real predict_signal pipeline")
    print(f"  NOTE: uses the model .pkl files in THIS folder.")
    print("=" * 66)
    grand = 0
    for s in config.PAIRS:
        try:
            grand += replay(s)["WOULD_TRADE"]
        except Exception as e:
            print(f"\n[{s}] ERROR: {e}")
    print("\n" + "=" * 66)
    print(f"  TOTAL would-trade across pairs in {DAYS}d: {grand}")
    print("  If this is ~0, a gate is too strict (see 'died at' — usually meta_veto")
    print("  or low_conf). If it's several per week, 2 quiet days is just variance.")
    print("=" * 66)


if __name__ == "__main__":
    main()
