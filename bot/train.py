#!/usr/bin/env python
# train.py — One-time training script (run on Mac)
#
# Usage:
#   cd /Users/orgilbk/Claude/Capstone/bot
#   python train.py

import time
import warnings
warnings.filterwarnings("ignore")

import config
from data_pipeline import fetch_historical
from features      import build_features, create_labels, get_feature_columns
from model         import train, save, feature_importance_plot
from backtest      import walk_forward_backtest, print_summary, plot_equity_curve


def main():
    print("=" * 60)
    print("  EUR/USD H1 — Walk-Forward Training Pipeline")
    print("=" * 60)

    # ── 1. Fetch data ──────────────────────────────────────────────────────
    print(f"\n[1/5] Fetching {config.YEARS_HISTORY}yr of {config.SYMBOL} H1 data…")
    raw_df = fetch_historical(source="yfinance")

    # ── 2. Feature engineering ─────────────────────────────────────────────
    print("\n[2/5] Building features…")
    t0 = time.time()
    df = build_features(raw_df)
    df["Signal"] = create_labels(df, method="atr")
    df = df.dropna(subset=["Signal"])
    df["Signal"] = df["Signal"].astype(int)

    feat_cols = get_feature_columns(df)
    print(f"  Features : {len(feat_cols)}")
    print(f"  Samples  : {len(df):,}")
    print(f"  Class distribution:\n{df['Signal'].value_counts().sort_index().to_string()}")
    print(f"  Elapsed  : {time.time()-t0:.1f}s")

    # ── 3. Walk-forward backtest ───────────────────────────────────────────
    print("\n[3/5] Running walk-forward backtest…")
    results = walk_forward_backtest(raw_df)
    print_summary(results)
    plot_equity_curve(results["equity_curve"], save_path="equity_curve.png")

    # ── 4. Train final model on full dataset ───────────────────────────────
    print("\n[4/5] Training final model on full dataset…")
    X = df[feat_cols]
    y = df["Signal"]
    final_model = train(X, y)

    # ── 5. Save ────────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving model → {config.MODEL_PATH}")
    save(final_model, config.MODEL_PATH)

    print("\n[done] Feature importance chart:")
    feature_importance_plot(final_model, feat_cols, save_path="feature_importance.png")

    print("\nAll done. Files written:")
    print(f"  {config.MODEL_PATH}")
    print(f"  equity_curve.png")
    print(f"  feature_importance.png")
    print("\nCopy the entire bot/ folder to Windows to run bot.py with MT5.\n")


if __name__ == "__main__":
    main()
