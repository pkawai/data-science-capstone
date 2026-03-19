# backtest.py — Walk-forward backtest with realistic costs and metrics

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import config
from features import build_features, create_labels, get_feature_columns
from model import train, predict_proba
from risk_manager import calculate_sl_tp

warnings.filterwarnings("ignore")


def _cost_per_trade(symbol: str) -> float:
    spread = config.PAIR_CONFIGS[symbol]["spread_pips"]
    return (spread + config.SLIPPAGE_PIPS) * config.PAIR_CONFIGS[symbol]["pip_size"]


# ── Public API ─────────────────────────────────────────────────────────────────

def walk_forward_backtest(raw_df: pd.DataFrame, symbol: str) -> dict:
    """Walk-forward backtest for a specific symbol."""
    df = build_features(raw_df)
    df["Signal"] = create_labels(df, method="triple_barrier")
    df = df.dropna(subset=["Signal"])
    df["Signal"] = df["Signal"].astype(int)

    folds = _generate_folds(df)
    if not folds:
        raise ValueError("Not enough data for even one walk-forward fold.")

    fold_results     = []
    all_trades       = []
    all_meta_signals = []

    for i, (train_df, test_df) in enumerate(folds):
        feat_cols = get_feature_columns(train_df)
        model     = train(train_df[feat_cols], train_df["Signal"])
        proba     = predict_proba(model, test_df[feat_cols])
        y_pred    = proba.argmax(axis=1)
        conf      = proba.max(axis=1)

        mask_conf    = conf >= config.CONFIDENCE_THRESHOLD
        mask_session = test_df.index.hour.isin(config.ACTIVE_HOURS)
        mask_adx     = test_df["ADX"]      > config.ADX_THRESHOLD
        mask_vol     = test_df["Vol_ratio"] < config.VOL_RATIO_MAX
        active       = (mask_conf & mask_session & mask_adx & mask_vol).values

        trades, fold_meta = _simulate_trades(
            test_df.reset_index(), y_pred, conf, active, symbol,
            probas=proba, feat_cols=feat_cols,
        )
        all_trades.extend(trades)
        all_meta_signals.extend(fold_meta)

        f1 = f1_score(test_df["Signal"], y_pred, average="macro", zero_division=0)
        win_rate, pf = _trade_metrics(trades)

        result = {
            "fold":          i + 1,
            "train_from":    train_df.index[0].date(),
            "train_to":      train_df.index[-1].date(),
            "test_from":     test_df.index[0].date(),
            "test_to":       test_df.index[-1].date(),
            "n_trades":      len(trades),
            "win_rate":      win_rate,
            "profit_factor": pf,
            "f1_macro":      f1,
        }
        fold_results.append(result)
        print(f"  Fold {i+1}: trades={len(trades):3d}  "
              f"WR={win_rate:.1%}  PF={pf:.2f}  F1={f1:.3f}  "
              f"({result['test_from']} → {result['test_to']})")

    equity_curve = _build_equity_curve(all_trades, config.ACCOUNT_BALANCE, symbol)
    summary      = _summarise(fold_results, equity_curve)
    return {
        "fold_results":  fold_results,
        "equity_curve":  equity_curve,
        "all_trades":    all_trades,
        "summary":       summary,
        "meta_signals":  all_meta_signals,
    }


def plot_equity_curve(equity_curve: pd.Series,
                      symbol: str = "",
                      save_path: str | None = None) -> None:
    path = save_path or f"equity_curve_{symbol}.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    equity_curve.plot(ax=ax, color="steelblue", linewidth=1.5)
    ax.axhline(config.ACCOUNT_BALANCE, color="grey", linestyle="--",
               linewidth=0.8, label="Starting balance")
    ax.set_title(f"Walk-Forward Equity Curve — {symbol} H1")
    ax.set_ylabel("Account Balance (USD)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[backtest] Equity curve saved → {path}")
    plt.show()


def print_summary(results: dict, symbol: str = "") -> None:
    s = results["summary"]
    print(f"\n{'='*55}")
    print(f"  WALK-FORWARD BACKTEST SUMMARY — {symbol}")
    print(f"{'='*55}")
    for r in results["fold_results"]:
        print(f"  Fold {r['fold']:>2}  "
              f"{str(r['test_from'])+'→'+str(r['test_to']):>23}  "
              f"{r['n_trades']:>5}  {r['win_rate']:>5.1%}  "
              f"{r['profit_factor']:>5.2f}  {r['f1_macro']:>5.3f}")
    print(f"{'='*55}")
    print(f"  Total trades   : {s['total_trades']}")
    print(f"  Avg win rate   : {s['avg_win_rate']:.1%}")
    print(f"  Avg profit fac : {s['avg_profit_factor']:.2f}")
    print(f"  Avg F1 macro   : {s['avg_f1']:.3f}")
    print(f"  Sharpe ratio   : {s['sharpe']:.2f}")
    print(f"  Max drawdown   : {s['max_drawdown']:.1%}")
    print(f"  Final balance  : ${s['final_balance']:,.0f}")
    print(f"{'='*55}\n")


# ── Trade simulation ───────────────────────────────────────────────────────────

def _simulate_trades(test_df, predictions, confidences, active_mask, symbol,
                     probas=None, feat_cols=None):
    trades       = []
    meta_signals = []
    pending_meta = None
    in_trade     = False
    entry_data   = {}
    cost         = _cost_per_trade(symbol)

    for i in range(len(test_df) - config.TB_HORIZON):
        row = test_df.iloc[i]

        if in_trade:
            result = _check_exit(entry_data, test_df, i, symbol)
            if result is not None:
                pnl, reason, entry_data = result
                trades.append({
                    "entry_date":   entry_data["date"],
                    "exit_date":    row.get("Datetime", row.name),
                    "direction":    entry_data["direction"],
                    "entry":        entry_data["entry"],
                    "pnl":          pnl - cost / config.PAIR_CONFIGS[symbol]["pip_size"],
                    "close_reason": reason,
                })
                # Finalise meta-signal with trade outcome
                if pending_meta is not None:
                    pending_meta["outcome"] = 1 if reason == "TP" else 0
                    meta_signals.append(pending_meta)
                    pending_meta = None
                in_trade = False

        if not in_trade and active_mask[i]:
            signal = int(predictions[i])
            if signal in (0, 2):
                atr   = row.get("ATR", 0.001)
                entry = row["Close"]
                sl, tp = calculate_sl_tp(entry, signal, atr, symbol)
                in_trade   = True
                entry_data = {
                    "date":         row.get("Datetime", row.name),
                    "direction":    signal,
                    "entry":        entry,
                    "sl":           sl,
                    "tp":           tp,
                    "atr":          atr,
                    "initial_risk": abs(entry - sl),
                    "breakeven":    False,
                    "start_idx":    i,
                }
                # Capture meta-signal entry (features + primary probabilities)
                if probas is not None and feat_cols is not None:
                    available   = [f for f in feat_cols if f in test_df.columns]
                    feat_vals   = test_df.iloc[i][available].values.astype(float).copy()
                    pending_meta = {
                        "features":      feat_vals,
                        "primary_proba": probas[i].copy(),
                        "signal":        signal,
                        "outcome":       None,   # filled on exit
                    }

    return trades, meta_signals


def _check_exit(entry_data, test_df, current_idx, symbol):
    pip_size  = config.PAIR_CONFIGS[symbol]["pip_size"]
    direction = entry_data["direction"]
    entry     = entry_data["entry"]
    sl        = entry_data["sl"]
    tp        = entry_data["tp"]
    init_risk = entry_data["initial_risk"]
    atr       = entry_data["atr"]
    breakeven = entry_data["breakeven"]

    next_idx = current_idx + 1
    if next_idx >= len(test_df):
        return None

    bar   = test_df.iloc[next_idx]
    high  = bar["High"]
    low   = bar["Low"]
    close = bar["Close"]

    if direction == 2:
        if high >= tp: return (tp - entry) / pip_size, "TP", entry_data
        if low  <= sl: return (sl - entry) / pip_size, "SL", entry_data
        if not breakeven and (close - entry) >= init_risk:
            entry_data["sl"] = entry; entry_data["breakeven"] = True
        if breakeven:
            trail = close - config.SL_ATR_MULT * atr
            if trail > entry_data["sl"]: entry_data["sl"] = trail
    else:
        if low  <= tp: return (entry - tp) / pip_size, "TP", entry_data
        if high >= sl: return (entry - sl) / pip_size, "SL", entry_data
        if not breakeven and (entry - close) >= init_risk:
            entry_data["sl"] = entry; entry_data["breakeven"] = True
        if breakeven:
            trail = close + config.SL_ATR_MULT * atr
            if trail < entry_data["sl"]: entry_data["sl"] = trail

    if (next_idx - entry_data["start_idx"]) >= config.TB_HORIZON:
        pnl = (close - entry) / pip_size if direction == 2 else (entry - close) / pip_size
        return pnl, "HORIZON", entry_data

    return None


def _generate_folds(df):
    folds       = []
    start       = df.index[0]
    end         = df.index[-1]
    train_delta = timedelta(days=config.TRAIN_MONTHS * 30)
    test_delta  = timedelta(days=config.TEST_MONTHS  * 30)
    step_delta  = timedelta(days=config.STEP_MONTHS  * 30)
    fold_start  = start
    while True:
        train_end = fold_start + train_delta
        test_end  = train_end  + test_delta
        if test_end > end: break
        train_df = df[(df.index >= fold_start) & (df.index < train_end)]
        test_df  = df[(df.index >= train_end)  & (df.index < test_end)]
        if len(train_df) > 100 and len(test_df) > 20:
            folds.append((train_df, test_df))
        fold_start += step_delta
    return folds


def _trade_metrics(trades):
    if not trades: return 0.0, 0.0
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls)
    pf = sum(wins) / abs(sum(losses)) if losses else 0.0
    return win_rate, pf


def _build_equity_curve(trades, start_balance, symbol):
    if not trades: return pd.Series([start_balance])
    pip_value = config.PAIR_CONFIGS[symbol]["pip_value_usd"]
    records   = sorted(trades, key=lambda t: t["exit_date"])
    balance  = start_balance
    dates    = [records[0]["entry_date"]]
    values   = [balance]
    for t in records:
        balance += t["pnl"] * pip_value
        dates.append(t["exit_date"])
        values.append(balance)
    return pd.Series(values, index=dates)


def _summarise(fold_results, equity_curve):
    total_trades = sum(r["n_trades"] for r in fold_results)
    avg_wr  = np.mean([r["win_rate"]      for r in fold_results])
    avg_pf  = np.mean([r["profit_factor"] for r in fold_results])
    avg_f1  = np.mean([r["f1_macro"]      for r in fold_results])
    eq      = equity_curve
    daily   = eq.pct_change().dropna()
    sharpe  = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    max_dd  = ((eq - eq.cummax()) / eq.cummax()).min()
    return {
        "total_trades":      total_trades,
        "avg_win_rate":      avg_wr,
        "avg_profit_factor": avg_pf,
        "avg_f1":            avg_f1,
        "sharpe":            sharpe,
        "max_drawdown":      max_dd,
        "final_balance":     float(eq.iloc[-1]),
    }
