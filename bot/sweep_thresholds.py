#!/usr/bin/env python
# sweep_thresholds.py — ADX threshold sensitivity test.
#
# Trains the model ONCE per walk-forward fold, then re-applies several ADX
# thresholds to the SAME predictions. This isolates the effect of the ADX gate
# on trade frequency and profit factor — cheaply and apples-to-apples.
#
# Usage:  python sweep_thresholds.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
config.USE_OPTUNA = False   # disable slow search; we only compare thresholds

from features import build_features, create_labels, get_feature_columns
from model import train, predict_proba
import backtest as bt

THRESHOLDS = [25, 20, 18, 15]
LOOKBACK_DAYS = 729   # yfinance 1h cap is ~730d


def _fetch(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    yf_sym = config.PAIR_CONFIGS[symbol]["yf_symbol"]
    df = yf.Ticker(yf_sym).history(period=f"{LOOKBACK_DAYS}d", interval="1h",
                                   auto_adjust=True)
    if df.empty:
        df = yf.Ticker(yf_sym.replace("=X", "")).history(
            period=f"{LOOKBACK_DAYS}d", interval="1h", auto_adjust=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df[~df.index.duplicated(keep="last")].sort_index()


def sweep_pair(symbol: str) -> dict:
    raw = _fetch(symbol)
    df = build_features(raw)
    df["Signal"] = create_labels(df, method="triple_barrier")
    df = df.dropna(subset=["Signal"])
    df["Signal"] = df["Signal"].astype(int)

    folds = bt._generate_folds(df)
    if not folds:
        raise ValueError(f"{symbol}: not enough data for a fold "
                         f"({len(df)} rows, {df.index[0].date()}→{df.index[-1].date()})")

    # threshold -> list of all trades across folds
    trades_by_thr = {t: [] for t in THRESHOLDS}

    for train_df, test_df in folds:
        feat_cols = get_feature_columns(train_df)
        model = train(train_df[feat_cols], train_df["Signal"])      # train ONCE
        proba = predict_proba(model, test_df[feat_cols])
        y_pred = proba.argmax(axis=1)
        conf = proba.max(axis=1)

        mask_conf = conf >= config.CONFIDENCE_THRESHOLD
        mask_sess = test_df.index.hour.isin(config.ACTIVE_HOURS)
        mask_vol = test_df["Vol_ratio"] < config.VOL_RATIO_MAX

        for thr in THRESHOLDS:
            mask_adx = test_df["ADX"] > thr
            active = (mask_conf & mask_sess & mask_adx & mask_vol).values
            trades, _ = bt._simulate_trades(
                test_df.reset_index(), y_pred, conf, active, symbol,
                probas=proba, feat_cols=feat_cols)
            trades_by_thr[thr].extend(trades)

    out = {"symbol": symbol, "folds": len(folds),
           "test_from": folds[0][1].index[0].date(),
           "test_to": folds[-1][1].index[-1].date(), "rows": {}}
    pip_value = config.PAIR_CONFIGS[symbol]["pip_value_usd"]
    for thr in THRESHOLDS:
        trades = trades_by_thr[thr]
        wr, pf = bt._trade_metrics(trades)
        net_pips = sum(t["pnl"] for t in trades)
        out["rows"][thr] = {
            "n": len(trades), "wr": wr, "pf": pf,
            "net_pips": net_pips, "net_usd": net_pips * pip_value,
        }
    return out


def main():
    print("=" * 74)
    print("  ADX THRESHOLD SWEEP — walk-forward, model trained once per fold")
    print(f"  Filters held constant: conf>={config.CONFIDENCE_THRESHOLD}, "
          f"vol<{config.VOL_RATIO_MAX}, hours {min(config.ACTIVE_HOURS)}-{max(config.ACTIVE_HOURS)} UTC")
    print("=" * 74)

    for sym in config.PAIRS:
        try:
            r = sweep_pair(sym)
        except Exception as e:
            print(f"\n[{sym}] ERROR: {e}")
            continue
        print(f"\n[{r['symbol']}]  {r['folds']} folds, test {r['test_from']} → {r['test_to']}")
        print(f"   {'ADX>':<6}{'trades':>8}{'win%':>8}{'PF':>8}{'net pips':>11}{'net USD':>11}")
        print(f"   {'-'*50}")
        for thr in THRESHOLDS:
            d = r["rows"][thr]
            print(f"   {thr:<6}{d['n']:>8}{d['wr']*100:>7.1f}{d['pf']:>8.2f}"
                  f"{d['net_pips']:>11.0f}{d['net_usd']:>11.0f}")

    print("\n" + "=" * 74)
    print("  Read: lower ADX = more trades. Watch whether PF stays > ~1.5 and")
    print("  net USD keeps rising. If PF collapses as ADX drops, the gate is")
    print("  protecting the edge — don't loosen it. If PF holds, loosening is safe.")
    print("=" * 74)


if __name__ == "__main__":
    main()
