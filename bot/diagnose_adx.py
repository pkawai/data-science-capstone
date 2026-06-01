#!/usr/bin/env python
# diagnose_adx.py — Answer the question: "Is the market really ranging, or is
# something wrong?" Pulls recent H1 data (yfinance, works on Mac — no MT5 needed),
# runs it through the REAL features.py, and reports how often the live ADX gate
# would actually open.
#
# Usage:  python diagnose_adx.py

import sys

import numpy as np
import pandas as pd

import config
from features import build_features, get_feature_columns

LOOKBACK_DAYS = 120   # ~4 months of H1 history to profile recent conditions


def _fetch(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    yf_sym = config.PAIR_CONFIGS[symbol]["yf_symbol"]
    df = yf.Ticker(yf_sym).history(period=f"{LOOKBACK_DAYS}d", interval="1h",
                                   auto_adjust=True)
    if df.empty:
        df = yf.Ticker(yf_sym.replace("=X", "")).history(
            period=f"{LOOKBACK_DAYS}d", interval="1h", auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def diagnose(symbol: str) -> dict:
    raw = _fetch(symbol)
    df = build_features(raw)

    adx = df["ADX"]
    active = df.index.hour.isin(config.ACTIVE_HOURS)
    thr = config.ADX_THRESHOLD

    # Bars during active session only (when the bot would actually consider trading)
    adx_active = adx[active]

    pass_all = (adx > thr).mean() * 100
    pass_active = (adx_active > thr).mean() * 100

    # How many distinct days in the last ~30 days had at least one qualifying bar?
    last30 = df[df.index >= df.index[-1] - pd.Timedelta(days=30)]
    last30_active = last30[last30.index.hour.isin(config.ACTIVE_HOURS)]
    qualifying = last30_active[last30_active["ADX"] > thr]
    days_with_signal = qualifying.index.normalize().nunique()
    total_days = last30_active.index.normalize().nunique()

    return {
        "symbol": symbol,
        "bars": len(df),
        "from": df.index[0].date(),
        "to": df.index[-1].date(),
        "current_adx": float(adx.iloc[-1]),
        "median_adx": float(adx.median()),
        "p90_adx": float(adx.quantile(0.90)),
        "max_adx": float(adx.max()),
        "pct_pass_all": pass_all,
        "pct_pass_active": pass_active,
        "days_with_signal": days_with_signal,
        "total_days": total_days,
    }


def main():
    thr = config.ADX_THRESHOLD
    print("=" * 70)
    print(f"  ADX DIAGNOSTIC — gate is: ADX > {thr}  (active hours {min(config.ACTIVE_HOURS)}–{max(config.ACTIVE_HOURS)} UTC)")
    print(f"  Data: yfinance H1, last {LOOKBACK_DAYS} days")
    print("=" * 70)

    rows = []
    for sym in config.PAIRS:
        try:
            r = diagnose(sym)
            rows.append(r)
            print(f"\n[{r['symbol']}]  {r['from']} → {r['to']}  ({r['bars']} bars)")
            print(f"   Current ADX        : {r['current_adx']:6.1f}   "
                  f"({'PASS' if r['current_adx'] > thr else 'ranging — blocked'})")
            print(f"   Median ADX         : {r['median_adx']:6.1f}")
            print(f"   90th pct ADX       : {r['p90_adx']:6.1f}")
            print(f"   Max ADX            : {r['max_adx']:6.1f}")
            print(f"   % bars ADX>{thr}     : {r['pct_pass_all']:5.1f}%  (all hours)")
            print(f"   % bars ADX>{thr}     : {r['pct_pass_active']:5.1f}%  (active hours only)")
            print(f"   Last 30 days       : {r['days_with_signal']}/{r['total_days']} "
                  f"trading days had >=1 qualifying bar")
        except Exception as e:
            print(f"\n[{sym}]  ERROR: {e}")

    if rows:
        print("\n" + "=" * 70)
        print("  VERDICT")
        print("=" * 70)
        avg_active = np.mean([r["pct_pass_active"] for r in rows])
        total_sig_days = sum(r["days_with_signal"] for r in rows)
        if avg_active < 5:
            print(f"  Market IS genuinely ranging. Only {avg_active:.1f}% of active-hour")
            print(f"  bars clear ADX>{thr}. The bot is behaving correctly — the filter is")
            print(f"  just strict for current conditions.")
        elif total_sig_days == 0:
            print(f"  No qualifying bars in 30 days across all pairs despite ADX>{thr} being")
            print(f"  reachable historically — investigate data feed / threshold mismatch.")
        else:
            print(f"  ADX>{thr} IS being cleared regularly ({avg_active:.1f}% of active bars).")
            print(f"  If the live bot still isn't trading, the blocker is DOWNSTREAM of ADX")
            print(f"  (confidence / meta / news / USD-direction). Check trades.log.")
        print(f"\n  Suggested experiment: re-run with ADX_THRESHOLD=20 and 18 to see how")
        print(f"  many more days qualify, THEN re-run backtest.py at that threshold to")
        print(f"  confirm the edge survives before changing the live config.")


if __name__ == "__main__":
    main()
