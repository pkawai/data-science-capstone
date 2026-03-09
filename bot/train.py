#!/usr/bin/env python
# train.py — Train a model for one currency pair (run on Mac)
#
# Usage:
#   python train.py EURUSD
#   python train.py GBPUSD
#   python train.py USDJPY
#   python train.py AUDUSD
#
# Train all pairs at once:
#   for pair in EURUSD GBPUSD USDJPY AUDUSD; do python train.py $pair; done

import sys
import time
import warnings
warnings.filterwarnings("ignore")

import config
from data_pipeline import fetch_historical
from features      import build_features, create_labels, get_feature_columns
from model         import train, save, feature_importance_plot
from backtest      import walk_forward_backtest, print_summary, plot_equity_curve


def main(symbol: str):
    if symbol not in config.PAIRS:
        print(f"Unknown symbol '{symbol}'. Choose from: {config.PAIRS}")
        sys.exit(1)

    print("=" * 60)
    print(f"  {symbol} H1 — Walk-Forward Training Pipeline")
    print("=" * 60)

    # ── 1. Fetch data ──────────────────────────────────────────────────────
    print(f"\n[1/5] Fetching {config.YEARS_HISTORY}yr of {symbol} H1 data…")
    raw_df = fetch_historical(symbol, source="yfinance")

    # ── 2. Feature engineering ─────────────────────────────────────────────
    print("\n[2/5] Building features…")
    t0 = time.time()
    df = build_features(raw_df)
    df["Signal"] = create_labels(df, method="triple_barrier")
    df = df.dropna(subset=["Signal"])
    df["Signal"] = df["Signal"].astype(int)

    feat_cols = get_feature_columns(df)
    print(f"  Features : {len(feat_cols)}")
    print(f"  Samples  : {len(df):,}")
    print(f"  Class distribution:\n{df['Signal'].value_counts().sort_index().to_string()}")
    print(f"  Elapsed  : {time.time()-t0:.1f}s")

    # ── 3. Walk-forward backtest ───────────────────────────────────────────
    print("\n[3/5] Running walk-forward backtest…")
    results = walk_forward_backtest(raw_df, symbol)
    print_summary(results, symbol)
    plot_equity_curve(results["equity_curve"], symbol)

    # ── 4. Train final model on full dataset ───────────────────────────────
    print("\n[4/5] Training final model on full dataset…")
    final_model = train(df[feat_cols], df["Signal"])

    # ── 5. Save ────────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving model → {config.model_path(symbol)}")
    save(final_model, symbol)

    feature_importance_plot(final_model, feat_cols, symbol=symbol)

    print(f"\nDone. Model saved: {config.model_path(symbol)}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <SYMBOL>")
        print(f"Available: {config.PAIRS}")
        sys.exit(1)
    main(sys.argv[1].upper())
