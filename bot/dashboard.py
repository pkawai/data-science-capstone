#!/usr/bin/env python
# dashboard.py — Live monitoring dashboard for the EUR/USD trading bot
#
# Usage:
#   streamlit run dashboard.py

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure bot/ directory is importable and used for relative file paths,
# whether invoked from inside bot/ or from the repo root (Streamlit Cloud).
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)
os.chdir(BOT_DIR)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EUR/USD Bot Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config

TRADES_CSV = "trades.csv"
STATE_JSON = "state.json"
REFRESH_SECONDS = 60
START_BALANCE = config.ACCOUNT_BALANCE

# ── Auto-refresh ───────────────────────────────────────────────────────────────
st.markdown(
    f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>",
    unsafe_allow_html=True,
)


# ── Data loaders ───────────────────────────────────────────────────────────────

def _empty_state() -> dict:
    return {
        "bot_running":    False,
        "balance":        config.ACCOUNT_BALANCE,
        "daily_pnl":      0,
        "in_trade":       False,
        "open_position":  None,
        "last_signal":    "N/A",
        "last_updated":   "N/A",
        "open_positions": [],
    }


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "time", "direction", "entry", "sl", "tp",
        "lots", "ticket", "confidence", "adx", "atr",
    ])


def load_state() -> dict:
    if not os.path.exists(STATE_JSON):
        return _empty_state()
    try:
        with open(STATE_JSON) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        st.warning(f"Could not read {STATE_JSON}: {e}. Showing defaults.")
        return _empty_state()


def load_trades() -> pd.DataFrame:
    if not os.path.exists(TRADES_CSV):
        return _empty_trades()
    try:
        df = pd.read_csv(TRADES_CSV, parse_dates=["time"])
        if df.empty:
            return _empty_trades()
        return df.sort_values("time", ascending=False).reset_index(drop=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, ValueError) as e:
        st.warning(f"Could not parse {TRADES_CSV}: {e}. Showing empty trade list.")
        return _empty_trades()


def _row_pip_size(symbol: str) -> float:
    """Pip size for a symbol, defaulting to non-JPY (0.0001)."""
    cfg = config.PAIR_CONFIGS.get(str(symbol), {})
    return cfg.get("pip_size", 0.0001)


def _potential_pnl(row) -> float:
    """
    USD value of a trade IF it reaches TP, using the correct per-pair pip size
    and pip value. NOTE: this is a *potential* (target) figure — the live CSV
    has no exit price, so realised P&L is unknown until a trade closes. Used
    only for relative sizing of the chart, clearly labelled in the UI.
    """
    try:
        sym      = row.get("symbol", "EURUSD")
        cfg      = config.PAIR_CONFIGS.get(str(sym), config.PAIR_CONFIGS["EURUSD"])
        pip      = cfg["pip_size"]
        pip_val  = cfg["pip_value_usd"]
        if row["direction"] == "BUY":
            pips = (row["tp"] - row["entry"]) / pip
        else:
            pips = (row["entry"] - row["tp"]) / pip
        return pips * pip_val * row["lots"]
    except Exception:
        return 0.0


def compute_metrics(trades: pd.DataFrame, start_balance: float = 10_000) -> dict:
    if trades.empty:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "total_pnl": 0, "avg_confidence": 0,
        }

    # IMPORTANT: the live trades.csv records ENTRY/SL/TP at order time, not the
    # exit. We therefore cannot compute realised win rate / profit factor from
    # it. Report only what's truthful: trade count, avg confidence, and total
    # *target* P&L (sum of TP-distance values) as an aspirational figure.
    trades = trades.copy()
    trades["target_pnl"] = trades.apply(_potential_pnl, axis=1)

    return {
        "total_trades":   len(trades),
        "win_rate":       None,   # unknown without exit prices
        "profit_factor":  None,   # unknown without exit prices
        "total_pnl":      trades["target_pnl"].sum(),
        "avg_confidence": trades["confidence"].mean() if "confidence" in trades.columns else 0,
    }


def build_equity_curve(trades: pd.DataFrame, start_balance: float = 10_000) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame({"time": [datetime.now(timezone.utc)], "balance": [start_balance]})

    df = trades.sort_values("time").copy()
    df["pnl"]     = df.apply(_potential_pnl, axis=1)
    df["balance"] = start_balance + df["pnl"].cumsum()
    return df[["time", "balance"]]


# ── Main dashboard ─────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📋 How It Works")
        st.markdown("""
**Pipeline:**
1. **Data** — MT5 H1 candles (EUR/USD, GBP/USD, USD/JPY)
2. **Features** — 30+ indicators (RSI, MACD, BB, ATR, ADX, MA, lagged)
3. **Labels** — Triple Barrier (TP/SL/time horizon)
4. **Model** — XGBoost + LightGBM + RandomForest ensemble
5. **Meta-model** — Binary classifier filters out low-quality signals
6. **Validation** — Walk-forward (15mo train / 3mo test)
7. **Execution** — ATR-based SL/TP, dynamic position sizing
        """)

        st.divider()

        st.markdown("## 🔄 Recent Updates")
        st.markdown("""
**Latest changes (Jun 2026):**
- 🐛 Fixed timeframe bug: bot was pulling H4, not H1 candles
- 🔭 Live feature window 200 → 12,000 bars (Daily-EMA needs it)
- 🎯 Confidence cap 0.65 — old models stored 0.75–0.80 the
  model could never reach (USD/JPY was stuck at 0 trades)
- 📉 ADX gate 25 → 20 (more trades, equal-or-better profit factor)
- 🩺 Added diagnostics: mt5_doctor, check_feed, why_no_trade

**Earlier upgrades:**
- ✨ Ensemble model (XGBoost + LightGBM + RF)
- 🧠 Meta-labeling for signal quality
- 📰 News blackout filter
- 💵 USD direction correlation filter
- ⚖️ Dynamic risk scaling per portfolio
        """)

        st.divider()

        st.markdown("## 🛠️ Tech Stack")
        st.markdown("""
- **Python** (pandas, numpy, scikit-learn)
- **ML:** XGBoost, LightGBM, RandomForest
- **Tuning:** Optuna
- **Broker:** MetaTrader 5 (MT5)
- **Dashboard:** Streamlit + Plotly
- **Source:** [GitHub](https://github.com/pkawai/data-science-capstone)
        """)

        st.divider()

        st.markdown("## ⚠️ Known Limitations")
        st.markdown("""
- Trend-following model — sits out genuinely ranging markets
- MT5 is Windows-only (Mac uses demo data)
- $5K demo limits position sizing
- Live win-rate/PF need recorded exits — see Backtest tab
        """)


BACKTEST_JSON = "backtest_results.json"


def load_backtest_results() -> dict | None:
    """Load saved walk-forward backtest results. Returns None if missing/invalid."""
    if not os.path.exists(BACKTEST_JSON):
        return None
    try:
        with open(BACKTEST_JSON) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        st.warning(f"Could not read {BACKTEST_JSON}: {e}")
        return None


def render_backtest_tab():
    st.header("📊 Walk-Forward Backtest Results")

    results = load_backtest_results()

    if results is None:
        st.info(
            "No saved backtest results found. Run `python train.py` on Windows to "
            "generate `backtest_results.json` with real walk-forward metrics."
        )
        return

    st.markdown(results.get("description", ""))
    st.caption(
        f"Method: {results.get('method', 'N/A')}  |  "
        f"Train: {results.get('train_months', '?')} mo  |  "
        f"Test: {results.get('test_months', '?')} mo  |  "
        f"Step: {results.get('step_months', '?')} mo  |  "
        f"Last updated: {results.get('last_updated', 'N/A')}"
    )

    st.divider()

    pairs_data = results.get("pairs", [])
    if not pairs_data:
        st.warning("No pair results in backtest_results.json")
        return

    summary_df = pd.DataFrame([
        {
            "Pair":             p["display"],
            "Trades":           p["trades"],
            "Win Rate":         f"{p['win_rate']:.1%}",
            "Profit Factor":    f"{p['profit_factor']:.2f}",
            "Sharpe Ratio":     f"{p['sharpe_ratio']:.2f}",
            "Max Drawdown":     f"{p['max_drawdown_pct']:.1f}%",
            "Avg Trade (pips)": f"{p['avg_trade_pips']:.1f}",
        }
        for p in pairs_data
    ])

    st.subheader("Performance Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Equity Curves")
    for p in pairs_data:
        symbol, display = p["symbol"], p["display"]
        st.markdown(f"#### {display}")
        eq_path = f"equity_curve_{symbol}.png"
        if os.path.exists(eq_path):
            st.image(eq_path, use_container_width=True)
        else:
            st.caption(f"_(Equity curve image not found: {eq_path})_")

    st.divider()

    st.subheader("Feature Importance")
    st.markdown("Top features driving model decisions for each pair.")
    cols = st.columns(len(pairs_data))
    for i, p in enumerate(pairs_data):
        with cols[i]:
            st.markdown(f"**{p['display']}**")
            fi_path = f"feature_importance_{p['symbol']}.png"
            if os.path.exists(fi_path):
                st.image(fi_path, use_container_width=True)
            else:
                st.caption(f"_(Image not found: {fi_path})_")


def render_about_tab():
    st.header("ℹ️ About This Project")

    st.markdown("""
    ### What is this?
    A machine-learning–powered forex trading bot that predicts Buy / Sell / Hold
    signals on three major currency pairs (EUR/USD, GBP/USD, USD/JPY) and
    executes trades through the MetaTrader 5 broker API.

    ### Why?
    To answer: **Can supervised learning models reliably predict short-term
    forex direction with positive risk-adjusted returns?**
    Capstone project for AUM Data Science, Spring 2026.

    ### How does it work?
    """)

    st.markdown("""
    | Stage | What happens |
    |-------|--------------|
    | **1. Data** | Fetch H1 OHLC candles from MetaTrader 5 |
    | **2. Features** | Build 30+ technical indicators (RSI, MACD, BB, ATR, ADX, MA, lagged returns) |
    | **3. Labels** | Triple Barrier method — label each candle by which barrier (TP / SL / time) hits first |
    | **4. Train** | Ensemble: XGBoost + LightGBM + Random Forest, averaged probabilities |
    | **5. Meta-Model** | Binary classifier predicts "will this signal actually win?" — filters out weak signals |
    | **6. Validate** | Walk-forward: 15 months train → 3 months test → slide forward |
    | **7. Live** | Pass 6 real-time filters (ADX, session, news, USD direction, vol, max trades), then execute |
    | **8. Manage** | Trailing stop + breakeven + time-based exit |
    """)

    st.divider()

    st.subheader("📐 Risk Management Rules")
    st.markdown("""
    - **Risk per trade:** 1.5% (scales down to 0.75% / 0.5% with more open trades)
    - **Stop-loss:** 1.0× ATR from entry
    - **Take-profit:** 1.5× ATR from entry (1.5:1 reward:risk)
    - **Daily loss limit:** 3% of account
    - **Max bars in trade:** 20 H1 bars (~20 hours)
    - **Trailing stop:** activates after profit ≥ initial risk, trails at 1.5× ATR
    """)

    st.divider()

    st.subheader("🔗 Resources")
    st.markdown("""
    - 💻 [GitHub Repository](https://github.com/pkawai/data-science-capstone)
    - 📄 [Project Proposal (DOCX)](https://github.com/pkawai/data-science-capstone/blob/main/Proposal%201.docx)
    - 📓 [EDA Notebook (Checkpoint 2)](https://github.com/pkawai/data-science-capstone/blob/main/Checkpoint%202/eda_checkpoint2.ipynb)
    """)

    st.divider()

    st.caption("Built by Orgil BK · American University of Mongolia · Spring 2026")


def render_monitor_tab(state: dict, trades: pd.DataFrame, START_BALANCE: float):
    bot_status  = "🟢 Running" if state.get("bot_running") else "🔴 Offline"
    last_update = state.get("last_updated", "N/A")
    pairs_str   = "  |  ".join(state.get("pairs_trading", config.PAIRS))
    st.caption(f"Bot status: {bot_status}  |  Pairs: {pairs_str}  |  Last updated: {last_update}  |  Auto-refreshes every {REFRESH_SECONDS}s")
    st.divider()

    # ── Top KPI cards ─────────────────────────────────────────────────────────
    balance   = state.get("balance", START_BALANCE)
    daily_pnl = state.get("daily_pnl", 0)
    metrics   = compute_metrics(trades, START_BALANCE)

    open_positions = state.get("open_positions", [])
    n_open         = len(open_positions) if isinstance(open_positions, list) else 0

    total_pnl     = balance - START_BALANCE
    total_pnl_pct = (total_pnl / START_BALANCE) * 100

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("💰 Balance", f"${balance:,.0f}",
                  delta=f"{total_pnl:+,.0f} ({total_pnl_pct:+.1f}%)")

    with col2:
        st.metric("📅 Today's P&L", f"${daily_pnl:+,.0f}")

    with col3:
        st.metric("📊 Open Trades", f"{n_open} / {len(config.PAIRS)}")

    with col4:
        # Win rate/PF need exit prices the live CSV doesn't have — show trade
        # count instead of a fabricated 100% win rate.
        st.metric("📈 Trades Logged", f"{metrics['total_trades']}")

    with col5:
        st.metric("🎯 Avg Confidence",
                  f"{metrics['avg_confidence']:.1%}" if metrics['total_trades'] > 0 else "N/A")

    st.divider()

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.subheader("Equity Curve (target projection)")
    st.caption("Projected from each trade's take-profit target — live exits "
               "aren't recorded yet, so this shows potential, not realised P&L.")

    eq_df = build_equity_curve(trades, START_BALANCE)
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_df["time"], y=eq_df["balance"],
        mode="lines", name="Balance",
        line=dict(color="#00b4d8", width=2),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.08)",
    ))
    fig.add_hline(y=START_BALANCE, line_dash="dash",
                  line_color="gray", annotation_text="Starting balance")
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Date",
        yaxis_title="Balance (USD)",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Open position + metrics row ───────────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Open Positions")
        open_positions = state.get("open_positions", [])
        if open_positions:
            for pos in open_positions:
                direction_icon = "🟢 BUY" if pos.get("type") == 2 else "🔴 SELL"
                profit = pos.get("profit", 0)
                profit_color = "green" if profit >= 0 else "red"
                st.markdown(
                    f"**{pos.get('symbol', '?')}** {direction_icon}  "
                    f"Entry: `{pos.get('price_open', 0)}`  "
                    f"SL: `{pos.get('sl', 0)}`  "
                    f"TP: `{pos.get('tp', 0)}`  "
                    f"P&L: :{profit_color}[${profit:+.2f}]"
                )
        else:
            st.info("No open positions right now.")

    with right:
        st.subheader("Strategy Metrics")
        m = metrics
        if m["total_trades"] > 0:
            col_a, col_b = st.columns(2)
            col_a.metric("Total Trades",    m["total_trades"])
            col_b.metric("Avg Confidence",  f"{m['avg_confidence']:.1%}")
            col_a.metric("Open Cap (pairs)", len(config.PAIRS))
            col_b.caption("Win rate / profit factor require trade exits — see "
                          "the Backtest tab for validated performance.")
        else:
            st.info("No trades logged yet.")

    st.divider()

    # ── Trade history ─────────────────────────────────────────────────────────
    st.subheader("Recent Trades")
    if not trades.empty:
        display = trades.head(20).copy()
        display["time"]       = display["time"].dt.strftime("%Y-%m-%d %H:%M")
        display["confidence"] = display["confidence"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        display["entry"]      = display["entry"].map(lambda x: f"{x:.5f}" if pd.notna(x) else "")
        display["sl"]         = display["sl"].map(lambda x: f"{x:.5f}" if pd.notna(x) else "")
        display["tp"]         = display["tp"].map(lambda x: f"{x:.5f}" if pd.notna(x) else "")
        display["adx"]        = display["adx"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")

        def color_direction(val):
            if val == "BUY":
                return "color: #2dc653"
            elif val == "SELL":
                return "color: #e63946"
            return ""

        styled = display[["time", "direction", "entry", "sl", "tp", "lots", "confidence", "adx", "ticket"]].style.map(
            color_direction, subset=["direction"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No trades placed yet. The bot will log trades here once it starts executing.")

    # ── Direction distribution ────────────────────────────────────────────────
    if not trades.empty and len(trades) >= 3:
        st.divider()
        st.subheader("Trade Distribution")
        col_pie, col_conf = st.columns(2)

        with col_pie:
            counts = trades["direction"].value_counts()
            fig_pie = px.pie(
                values=counts.values,
                names=counts.index,
                title="Buy vs Sell",
                color=counts.index,
                color_discrete_map={"BUY": "#2dc653", "SELL": "#e63946"},
                hole=0.4,
            )
            fig_pie.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=280)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_conf:
            if "confidence" in trades.columns:
                fig_hist = px.histogram(
                    trades, x="confidence", nbins=15,
                    title="Confidence Distribution",
                    color_discrete_sequence=["#00b4d8"],
                )
                fig_hist.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=280)
                st.plotly_chart(fig_hist, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.caption("Multi-Pair H1 Bot | XGBoost + Triple Barrier | Demo account only — not financial advice.")


def main():
    render_sidebar()

    state         = load_state()
    trades        = load_trades()
    START_BALANCE = config.ACCOUNT_BALANCE

    st.title("📈 Forex Auto-Trading Bot")
    st.markdown("ML-powered trading signals on EUR/USD, GBP/USD, USD/JPY · Real-time monitoring + backtest results")

    tab_monitor, tab_backtest, tab_about = st.tabs([
        "📊 Live Monitor",
        "📈 Backtest Results",
        "ℹ️ About",
    ])

    with tab_monitor:
        render_monitor_tab(state, trades, START_BALANCE)

    with tab_backtest:
        render_backtest_tab()

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
