#!/usr/bin/env python
# calibrate_floor.py — Tune DIRECTIONAL_FLOOR to YOUR actual live models.
#
# WHY: the right floor depends on how confident each trained model gets. Models
# trained on different machines/library versions express confidence very
# differently (Mac models reach ~0.50 directional; the Windows ones may top out
# ~0.25). A floor copied from the wrong models = zero trades. This script loads
# the model_*.pkl files IN THIS FOLDER, measures their real directional
# confidence on recent data, backtests candidate floors, prints the trade-off,
# picks a sensible floor, and writes it to local_overrides.json so the live bot
# uses it automatically (no config edit, no git conflict).
#
# Run on the SAME machine as the live bot (Windows). MT5 not required — uses
# yfinance for the candles. Usage:  python calibrate_floor.py

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
from features import build_features, create_labels, get_feature_columns
from model import load, predict_proba
from risk_manager import calculate_sl_tp

# Target ~1-2 trades/day/pair in active hours; require a minimally positive edge.
TARGET_TRADES_PER_DAY = 1.0
MIN_WIN_RATE = 0.50
CANDIDATE_FLOORS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
LOOKBACK_DAYS = 180
OVERRIDES_FILE = "local_overrides.json"


def _fetch(symbol):
    import yfinance as yf
    yf_sym = config.PAIR_CONFIGS[symbol]["yf_symbol"]
    df = yf.Ticker(yf_sym).history(period=f"{LOOKBACK_DAYS}d", interval="1h", auto_adjust=True)
    if df.empty:
        df = yf.Ticker(yf_sym.replace("=X", "")).history(period=f"{LOOKBACK_DAYS}d", interval="1h", auto_adjust=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df[~df.index.duplicated(keep="last")].sort_index()


def _simulate(df, entries, symbol):
    """Quick TP/SL-first outcome sim for a list of (idx, signal) entries."""
    horizon = config.TB_HORIZON
    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
    atrs = df["ATR"].values
    wins = losses = 0
    pip = config.PAIR_CONFIGS[symbol]["pip_size"]
    net_pips = 0.0
    for i, sig in entries:
        if i + horizon >= len(df):
            continue
        entry = closes[i]
        atr = atrs[i]
        sl, tp = calculate_sl_tp(entry, sig, atr, symbol)
        outcome = None
        for j in range(i + 1, i + 1 + horizon):
            if sig == 2:   # BUY
                if highs[j] >= tp: outcome = (tp - entry) / pip; break
                if lows[j] <= sl:  outcome = (sl - entry) / pip; break
            else:          # SELL
                if lows[j] <= tp:  outcome = (entry - tp) / pip; break
                if highs[j] >= sl: outcome = (entry - sl) / pip; break
        if outcome is None:
            outcome = ((closes[i + horizon] - entry) if sig == 2
                       else (entry - closes[i + horizon])) / pip
        net_pips += outcome
        if outcome > 0: wins += 1
        else:           losses += 1
    n = wins + losses
    wr = wins / n if n else 0.0
    return n, wr, net_pips


def calibrate_pair(symbol):
    bundle = load(symbol)
    df = build_features(_fetch(symbol))
    feat_cols = bundle.get("feature_cols") or get_feature_columns(df)
    avail = [f for f in feat_cols if f in df.columns]
    proba = predict_proba(bundle["primary_models"], df[avail])

    p_sell, p_buy = proba[:, 0], proba[:, 2]
    dir_sig = np.where(p_buy >= p_sell, 2, 0)
    p_dir = np.maximum(p_buy, p_sell)

    base = (np.asarray(df.index.hour.isin(config.ACTIVE_HOURS))
            & (df["ADX"] > config.ADX_THRESHOLD).to_numpy()
            & (df["Vol_ratio"] < config.VOL_RATIO_MAX).to_numpy())

    days = max((df.index[-1] - df.index[0]).days, 1)
    print(f"\n[{symbol}]  directional confidence (p_dir) on tradeable bars:")
    pd_active = p_dir[base]
    if len(pd_active):
        print(f"   min/median/90th/max = {pd_active.min():.2f} / "
              f"{np.median(pd_active):.2f} / {np.quantile(pd_active,0.9):.2f} / "
              f"{pd_active.max():.2f}")
    print(f"   {'floor':<8}{'trades':>8}{'/day':>7}{'win%':>8}{'net pips':>11}")

    best = None
    for f in CANDIDATE_FLOORS:
        mask = base & (p_dir >= f)
        entries = [(i, int(dir_sig[i])) for i in np.where(mask)[0]]
        n, wr, net = _simulate(df, entries, symbol)
        per_day = n / days
        print(f"   {f:<8.2f}{n:>8}{per_day:>7.2f}{wr*100:>7.1f}{net:>11.0f}")
        # pick the LOWEST floor that still wins >= MIN_WIN_RATE and trades enough
        if wr >= MIN_WIN_RATE and per_day >= 0.3 and net > 0:
            if best is None or f < best[0]:
                best = (f, n, wr, per_day)

    # fallback: if nothing met the bar, take the floor with best win-rate*trades
    return best, symbol


def main():
    print("=" * 64)
    print("  CALIBRATE DIRECTIONAL_FLOOR to your local models")
    print("=" * 64)
    chosen = []
    for s in config.PAIRS:
        try:
            best, sym = calibrate_pair(s)
            if best:
                print(f"   -> {s}: pick floor {best[0]:.2f} "
                      f"({best[3]:.2f} trades/day, {best[2]*100:.0f}% win)")
                chosen.append(best[0])
            else:
                print(f"   -> {s}: no floor gave a positive edge — model may be weak")
        except Exception as e:
            print(f"\n[{s}] ERROR: {e}")

    print("\n" + "=" * 64)
    if chosen:
        # one global floor = the median of per-pair picks (robust)
        floor = float(np.median(chosen))
        overrides = {"DIRECTIONAL_OVERRIDE": True,
                     "DIRECTIONAL_FLOOR": round(floor, 2)}
        with open(OVERRIDES_FILE, "w") as f:
            json.dump(overrides, f, indent=2)
        print(f"  WROTE {OVERRIDES_FILE}: {overrides}")
        print(f"  The bot will now use DIRECTIONAL_FLOOR={floor:.2f} automatically.")
        print(f"  Restart the bot:  Ctrl+C  then  python bot.py")
    else:
        print("  No positive-edge floor found on any pair. The current models")
        print("  may be too weak — consider retraining. Tell Claude this output.")
    print("=" * 64)


if __name__ == "__main__":
    main()
