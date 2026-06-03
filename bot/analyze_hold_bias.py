#!/usr/bin/env python
# analyze_hold_bias.py — Is the model just too HOLD-biased to ever trade?
#
# Two questions:
#  1. Of in-session bars with ADX>threshold, how often is HOLD the argmax?
#  2. If instead we take the STRONGER of Buy/Sell whenever its probability
#     clears a floor (ignoring that HOLD is nominally higher), how many trades
#     do we get and what's the profit factor? -> tells us if a "directional
#     override" is a safe way to make a HOLD-heavy model actually trade.
#
# Walk-forward, model trained once per fold. Usage: python analyze_hold_bias.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
config.USE_OPTUNA = False

from features import build_features, create_labels, get_feature_columns
from model import train, predict_proba
import backtest as bt

FLOORS = [0.35, 0.40, 0.45, 0.50]
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


def analyse(symbol):
    raw = _fetch(symbol)
    df = build_features(raw)
    df["Signal"] = create_labels(df, method="triple_barrier")
    df = df.dropna(subset=["Signal"])
    df["Signal"] = df["Signal"].astype(int)

    folds = bt._generate_folds(df)
    if not folds:
        raise ValueError("no folds")

    n_active = 0
    n_hold_argmax = 0
    # directional override trades per floor
    trades_by_floor = {f: [] for f in FLOORS}

    for train_df, test_df in folds:
        feat_cols = get_feature_columns(train_df)
        model = train(train_df[feat_cols], train_df["Signal"])
        proba = predict_proba(model, test_df[feat_cols])   # (n,3): [SELL,HOLD,BUY]

        base = (np.asarray(test_df.index.hour.isin(config.ACTIVE_HOURS))
                & (test_df["ADX"] > config.ADX_THRESHOLD).to_numpy()
                & (test_df["Vol_ratio"] < config.VOL_RATIO_MAX).to_numpy())

        argmax = proba.argmax(axis=1)
        n_active += int(base.sum())
        n_hold_argmax += int((base & (argmax == 1)).sum())

        # directional: stronger of BUY(2)/SELL(0), ignore HOLD
        p_sell, p_buy = proba[:, 0], proba[:, 2]
        dir_sig = np.where(p_buy >= p_sell, 2, 0)
        p_dir = np.maximum(p_buy, p_sell)

        for f in FLOORS:
            active = base & (p_dir >= f)
            # build a y_pred that is the directional signal; probas=None since we
            # only need PnL here (skip meta-signal capture, which needs DataFrame)
            trades, _ = bt._simulate_trades(
                test_df.reset_index(), dir_sig, p_dir, active, symbol,
                probas=None, feat_cols=None)
            trades_by_floor[f].extend(trades)

    pct_hold = 100 * n_hold_argmax / max(n_active, 1)
    print(f"\n[{symbol}]  in-session ADX>{config.ADX_THRESHOLD} bars: {n_active}")
    print(f"   HOLD was the model's pick on {pct_hold:.0f}% of them "
          f"-> that's why argmax almost never trades")
    print(f"   DIRECTIONAL OVERRIDE (take stronger of Buy/Sell >= floor):")
    print(f"      {'floor':<8}{'trades':>8}{'win%':>8}{'PF':>8}{'net pips':>11}")
    pip_value = config.PAIR_CONFIGS[symbol]["pip_value_usd"]
    for f in FLOORS:
        t = trades_by_floor[f]
        wr, pf = bt._trade_metrics(t)
        net = sum(x["pnl"] for x in t)
        print(f"      {f:<8.2f}{len(t):>8}{wr*100:>7.1f}{pf:>8.2f}{net:>11.0f}")


def main():
    print("=" * 64)
    print("  HOLD-BIAS ANALYSIS + directional override backtest (2yr WF)")
    print("=" * 64)
    for s in config.PAIRS:
        try:
            analyse(s)
        except Exception as e:
            print(f"\n[{s}] ERROR: {e}")
    print("\n" + "=" * 64)
    print("  If HOLD% is very high AND a directional floor gives many trades at")
    print("  PF>~1.5, a trend-aware override is the right fix for a HOLD-heavy")
    print("  model. Pick the floor with good PF + usable trade count.")
    print("=" * 64)


if __name__ == "__main__":
    main()
