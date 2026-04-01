#!/usr/bin/env python
# train.py — Full training pipeline for one currency pair (run on Mac)
#
# Pipeline:
#   1. Fetch 2yr H1 data
#   2. Build features (now includes D1 EMA200, time features)
#   3. Walk-forward backtest → collect meta-signals
#   4. Optuna hyperparameter search (optional, ~5 min/pair)
#   5. Train meta-labeling model
#   6. Optimise confidence threshold on OOS data
#   7. Train final ensemble (XGBoost + LightGBM + RandomForest)
#   8. Save ModelBundle
#
# Usage:
#   python train.py EURUSD
#   python train.py GBPUSD
#   python train.py USDJPY
#
# Train all pairs:
#   for pair in EURUSD GBPUSD USDJPY; do python train.py $pair; done

import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import config
from data_pipeline import fetch_historical
from features      import build_features, create_labels, get_feature_columns
from model         import (train, train_meta, save, feature_importance_plot,
                           _train_xgb)
from backtest      import walk_forward_backtest, print_summary, plot_equity_curve


# ── Optuna hyperparameter search ───────────────────────────────────────────────

def _optimize_hyperparams(df, feat_cols: list[str]) -> dict:
    """
    Run Optuna on a single temporal train/val split (first 70% / last 30%).
    Returns the best XGBoost hyperparams dict.
    Falls back to config defaults if Optuna is unavailable.
    """
    try:
        import optuna
    except ImportError:
        print("[optuna] Not installed — using config defaults.")
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.metrics import f1_score as sk_f1
    from sklearn.model_selection import TimeSeriesSplit

    X_all = df[feat_cols]
    y_all = df["Signal"]
    tscv  = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        scores = []
        for train_idx, val_idx in tscv.split(X_all):
            X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
            X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]
            model  = _train_xgb(X_tr, y_tr, params=params)
            y_pred = model.predict_proba(X_val).argmax(axis=1)
            scores.append(sk_f1(y_val, y_pred, average="macro", zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.OPTUNA_TRIALS, show_progress_bar=True)
    best = study.best_params
    print(f"[optuna] Best F1: {study.best_value:.3f} | Params: {best}")
    return best


# ── Confidence threshold optimisation ─────────────────────────────────────────

def _optimize_threshold(meta_signals: list) -> float:
    """
    Sweep confidence thresholds on walk-forward OOS signals and pick the one
    with the best profit factor.  Returns the optimised threshold.
    """
    if not meta_signals:
        return config.CONFIDENCE_THRESHOLD

    # (max_proba_of_predicted_class, outcome)
    pairs = [
        (float(np.max(ms["primary_proba"])), ms["outcome"])
        for ms in meta_signals
        if ms.get("outcome") is not None
    ]
    if len(pairs) < 10:
        return config.CONFIDENCE_THRESHOLD

    best_thr, best_pf = config.CONFIDENCE_THRESHOLD, 0.0
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        filtered = [(c, o) for c, o in pairs if c >= thr]
        if len(filtered) < 10:
            continue
        wins   = sum(1 for _, o in filtered if o == 1)
        losses = sum(1 for _, o in filtered if o == 0)
        pf     = wins / max(losses, 1)
        if pf > best_pf:
            best_pf, best_thr = pf, thr

    print(f"[train] Optimised confidence threshold: {best_thr:.2f}  "
          f"(OOS profit factor: {best_pf:.2f}, "
          f"n_signals={len([p for p in pairs if p[0] >= best_thr])})")
    return best_thr


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(symbol: str):
    if symbol not in config.PAIRS:
        print(f"Unknown symbol '{symbol}'. Choose from: {config.PAIRS}")
        sys.exit(1)

    print("=" * 60)
    print(f"  {symbol} H1 — Walk-Forward Training Pipeline v2")
    print("=" * 60)

    # ── 1. Fetch data ──────────────────────────────────────────────────────
    print(f"\n[1/7] Fetching {config.YEARS_HISTORY}yr of {symbol} H1 data…")
    raw_df = fetch_historical(symbol, source="yfinance")

    # ── 2. Feature engineering ─────────────────────────────────────────────
    print("\n[2/7] Building features (incl. D1 EMA200 + time features)…")
    t0 = time.time()
    df = build_features(raw_df)
    df["Signal"] = create_labels(df, method="triple_barrier")
    df = df.dropna(subset=["Signal"])
    df["Signal"] = df["Signal"].astype(int)

    feat_cols = get_feature_columns(df)
    print(f"  Features : {len(feat_cols)}")
    print(f"  Samples  : {len(df):,}")
    print(f"  Class dist:\n{df['Signal'].value_counts().sort_index().to_string()}")
    print(f"  Elapsed  : {time.time()-t0:.1f}s")

    # ── 3. Walk-forward backtest (also collects meta-signal data) ──────────
    print("\n[3/7] Running walk-forward backtest…")
    results = walk_forward_backtest(raw_df, symbol)
    print_summary(results, symbol)
    plot_equity_curve(results["equity_curve"], symbol)

    meta_signals = results.get("meta_signals", [])
    print(f"  Meta-signals collected: {len(meta_signals)} "
          f"(with outcome: {sum(1 for m in meta_signals if m.get('outcome') is not None)})")

    # ── 4. Optuna hyperparameter search ────────────────────────────────────
    best_xgb_params = None
    if config.USE_OPTUNA:
        print(f"\n[4/7] Optuna search ({config.OPTUNA_TRIALS} trials)…")
        best_xgb_params = _optimize_hyperparams(df, feat_cols)
    else:
        print("\n[4/7] Optuna disabled (USE_OPTUNA=False). Using config defaults.")

    # Update config XGBOOST_PARAMS if Optuna found something better
    if best_xgb_params:
        merged = {**config.XGBOOST_PARAMS, **best_xgb_params}
        # Remove use_label_encoder (deprecated)
        merged.pop("use_label_encoder", None)
    else:
        merged = {k: v for k, v in config.XGBOOST_PARAMS.items()
                  if k != "use_label_encoder"}

    # ── 5. Train meta-labeling model ───────────────────────────────────────
    print("\n[5/7] Training meta-labeling model…")
    meta_model = None
    valid_meta = [m for m in meta_signals if m.get("outcome") is not None]
    if len(valid_meta) >= 20:
        features_arr = np.array([m["features"] for m in valid_meta])
        proba_arr    = np.array([m["primary_proba"] for m in valid_meta])
        X_meta       = np.concatenate([features_arr, proba_arr], axis=1)
        y_meta       = np.array([m["outcome"] for m in valid_meta])
        meta_model   = train_meta(X_meta, y_meta)
    else:
        print(f"  Only {len(valid_meta)} meta-signals — need ≥20. Skipping meta-model.")

    # ── 6. Optimise confidence threshold ───────────────────────────────────
    print("\n[6/7] Optimising confidence threshold…")
    best_threshold = _optimize_threshold(meta_signals)

    # ── 7. Train final ensemble on full dataset ────────────────────────────
    print("\n[7/7] Training final ensemble on full dataset…")
    final_ensemble = train(df[feat_cols], df["Signal"], xgb_params=merged)

    # ── Save ModelBundle ───────────────────────────────────────────────────
    bundle = {
        "primary_models":       final_ensemble,
        "meta_model":           meta_model,
        "feature_cols":         feat_cols,
        "confidence_threshold": best_threshold,
        "meta_threshold":       config.META_CONFIDENCE_THRESHOLD,
    }
    save(bundle, symbol)
    print(f"\nBundle saved → {config.model_path(symbol)}")
    print(f"  Ensemble size       : {len(final_ensemble)} models")
    print(f"  Meta-model          : {'yes' if meta_model else 'no (not enough OOS data)'}")
    print(f"  Confidence threshold: {best_threshold:.2f}")

    feature_importance_plot(bundle, feat_cols, symbol=symbol)
    print(f"\nDone. {symbol} ready.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <SYMBOL>")
        print(f"Available: {config.PAIRS}")
        sys.exit(1)
    main(sys.argv[1].upper())
