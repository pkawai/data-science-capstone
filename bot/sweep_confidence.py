#!/usr/bin/env python
# sweep_confidence.py — How does the CONFIDENCE threshold trade off frequency
# vs quality? Trains once per walk-forward fold, then re-applies several
# confidence cutoffs to the SAME predictions (with the meta-model veto intact)
# and reports trades / win% / profit factor at each. ADX>20 + session + vol
# gates held constant, matching the live bot.
#
# Usage:  python sweep_confidence.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
config.USE_OPTUNA = False

from features import build_features, create_labels, get_feature_columns
from model import train, train_meta, predict_proba
import backtest as bt

CONF_LEVELS = [0.55, 0.60, 0.65, 0.70]
LOOKBACK_DAYS = 729


def _fetch(symbol):
    import yfinance as yf
    yf_sym = config.PAIR_CONFIGS[symbol]["yf_symbol"]
    df = yf.Ticker(yf_sym).history(period=f"{LOOKBACK_DAYS}d", interval="1h", auto_adjust=True)
    if df.empty:
        df = yf.Ticker(yf_sym.replace("=X", "")).history(period=f"{LOOKBACK_DAYS}d", interval="1h", auto_adjust=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df[~df.index.duplicated(keep="last")].sort_index()


def sweep_pair(symbol):
    raw = _fetch(symbol)
    df = build_features(raw)
    df["Signal"] = create_labels(df, method="triple_barrier")
    df = df.dropna(subset=["Signal"])
    df["Signal"] = df["Signal"].astype(int)

    folds = bt._generate_folds(df)
    if not folds:
        raise ValueError(f"{symbol}: not enough data")

    trades_by_conf = {c: [] for c in CONF_LEVELS}

    for train_df, test_df in folds:
        feat_cols = get_feature_columns(train_df)
        model = train(train_df[feat_cols], train_df["Signal"])
        proba = predict_proba(model, test_df[feat_cols])
        y_pred = proba.argmax(axis=1)
        conf = proba.max(axis=1)

        mask_sess = test_df.index.hour.isin(config.ACTIVE_HOURS)
        mask_adx = test_df["ADX"] > config.ADX_THRESHOLD
        mask_vol = test_df["Vol_ratio"] < config.VOL_RATIO_MAX
        base = (mask_sess & mask_adx & mask_vol).values

        for c in CONF_LEVELS:
            active = base & (conf >= c)
            trades, _ = bt._simulate_trades(
                test_df.reset_index(), y_pred, conf, active, symbol,
                probas=proba, feat_cols=feat_cols)
            trades_by_conf[c].extend(trades)

    pip_value = config.PAIR_CONFIGS[symbol]["pip_value_usd"]
    print(f"\n[{symbol}]  {len(folds)} folds, "
          f"test {folds[0][1].index[0].date()} -> {folds[-1][1].index[-1].date()}")
    print(f"   {'conf>=':<8}{'trades':>8}{'win%':>8}{'PF':>8}{'net pips':>11}{'net USD':>11}")
    print(f"   {'-'*52}")
    for c in CONF_LEVELS:
        t = trades_by_conf[c]
        wr, pf = bt._trade_metrics(t)
        net = sum(x["pnl"] for x in t)
        print(f"   {c:<8.2f}{len(t):>8}{wr*100:>7.1f}{pf:>8.2f}{net:>11.0f}{net*pip_value:>11.0f}")


def main():
    print("=" * 64)
    print("  CONFIDENCE SWEEP — walk-forward, ADX>%d held constant" % config.ADX_THRESHOLD)
    print("  (meta-model veto applied inside _simulate_trades)")
    print("=" * 64)
    for s in config.PAIRS:
        try:
            sweep_pair(s)
        except Exception as e:
            print(f"\n[{s}] ERROR: {e}")
    print("\n" + "=" * 64)
    print("  Pick the lowest conf where PF stays healthy (>~1.6) and trade count")
    print("  is usable. That becomes the per-pair cap.")
    print("=" * 64)


if __name__ == "__main__":
    main()
