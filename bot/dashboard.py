#!/usr/bin/env python
# dashboard.py — Live monitoring dashboard for the EUR/USD trading bot
#
# Usage:
#   streamlit run dashboard.py

import json
import os
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EUR/USD Bot Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
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

def load_state() -> dict:
    if not os.path.exists(STATE_JSON):
        return {"bot_running": False, "balance": 10000, "daily_pnl": 0,
                "in_trade": False, "open_position": None,
                "last_signal": "N/A", "last_updated": "N/A"}
    with open(STATE_JSON) as f:
        return json.load(f)


def load_trades() -> pd.DataFrame:
    if not os.path.exists(TRADES_CSV):
        return pd.DataFrame(columns=[
            "time", "direction", "entry", "sl", "tp",
            "lots", "ticket", "confidence", "adx", "atr",
        ])
    df = pd.read_csv(TRADES_CSV, parse_dates=["time"])
    return df.sort_values("time", ascending=False).reset_index(drop=True)


def compute_metrics(trades: pd.DataFrame, start_balance: float = 10_000) -> dict:
    if trades.empty:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "total_pnl": 0, "avg_confidence": 0,
        }

    # Estimate pnl per trade from TP/SL direction
    # (live trades don't have exit price yet — approximate from TP/SL distance)
    def est_pnl(row):
        try:
            if row["direction"] == "BUY":
                return (row["tp"] - row["entry"]) / 0.0001 * 10 * row["lots"]
            else:
                return (row["entry"] - row["tp"]) / 0.0001 * 10 * row["lots"]
        except Exception:
            return 0

    trades = trades.copy()
    trades["est_pnl"] = trades.apply(est_pnl, axis=1)

    wins         = trades[trades["est_pnl"] > 0]
    losses       = trades[trades["est_pnl"] < 0]
    win_rate     = len(wins) / len(trades) if len(trades) > 0 else 0
    gross_profit = wins["est_pnl"].sum()
    gross_loss   = abs(losses["est_pnl"].sum()) if len(losses) > 0 else 1
    pf           = gross_profit / gross_loss if gross_loss > 0 else 0
    total_pnl    = trades["est_pnl"].sum()

    return {
        "total_trades":   len(trades),
        "win_rate":       win_rate,
        "profit_factor":  pf,
        "total_pnl":      total_pnl,
        "avg_confidence": trades["confidence"].mean() if "confidence" in trades.columns else 0,
    }


def build_equity_curve(trades: pd.DataFrame, start_balance: float = 10_000) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame({"time": [datetime.now(timezone.utc)], "balance": [start_balance]})

    df = trades.sort_values("time").copy()

    def est_pnl(row):
        try:
            if row["direction"] == "BUY":
                return (row["tp"] - row["entry"]) / 0.0001 * 10 * row["lots"]
            else:
                return (row["entry"] - row["tp"]) / 0.0001 * 10 * row["lots"]
        except Exception:
            return 0

    df["pnl"]     = df.apply(est_pnl, axis=1)
    df["balance"] = start_balance + df["pnl"].cumsum()
    return df[["time", "balance"]]


# ── Main dashboard ─────────────────────────────────────────────────────────────

def main():
    state  = load_state()
    trades = load_trades()
    START_BALANCE = config.ACCOUNT_BALANCE

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("📈 EUR/USD Trading Bot — Live Monitor")

    bot_status = "🟢 Running" if state.get("bot_running") else "🔴 Offline"
    last_update = state.get("last_updated", "N/A")
    st.caption(f"Bot status: {bot_status}  |  Last updated: {last_update}  |  Auto-refreshes every {REFRESH_SECONDS}s")
    st.divider()

    # ── Top KPI cards ─────────────────────────────────────────────────────────
    balance    = state.get("balance", START_BALANCE)
    daily_pnl  = state.get("daily_pnl", 0)
    in_trade   = state.get("in_trade", False)
    last_signal = state.get("last_signal", "N/A")
    metrics    = compute_metrics(trades, START_BALANCE)

    total_pnl     = balance - START_BALANCE
    total_pnl_pct = (total_pnl / START_BALANCE) * 100

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("💰 Balance", f"${balance:,.0f}",
                  delta=f"{total_pnl:+,.0f} ({total_pnl_pct:+.1f}%)")

    with col2:
        daily_color = "normal" if daily_pnl >= 0 else "inverse"
        st.metric("📅 Today's P&L", f"${daily_pnl:+,.0f}")

    with col3:
        trade_label = "YES" if in_trade else "NO"
        trade_icon  = "🔵" if in_trade else "⚪"
        st.metric("📊 In Trade", f"{trade_icon} {trade_label}")

    with col4:
        st.metric("🎯 Win Rate",
                  f"{metrics['win_rate']:.1%}" if metrics['total_trades'] > 0 else "N/A",
                  delta=f"{metrics['total_trades']} trades")

    with col5:
        st.metric("⚡ Last Signal", last_signal)

    st.divider()

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.subheader("Equity Curve")

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
        st.subheader("Open Position")
        pos = state.get("open_position")
        if pos and in_trade:
            direction_icon = "🟢 BUY" if pos.get("type") == 2 else "🔴 SELL"
            st.markdown(f"**Direction:** {direction_icon}")
            st.markdown(f"**Entry price:** `{pos.get('price_open', 'N/A'):.5f}`")
            st.markdown(f"**Stop Loss:** `{pos.get('sl', 'N/A'):.5f}`")
            st.markdown(f"**Take Profit:** `{pos.get('tp', 'N/A'):.5f}`")
            st.markdown(f"**Volume:** `{pos.get('volume', 'N/A')} lots`")
            profit = pos.get("profit", 0)
            profit_color = "green" if profit >= 0 else "red"
            st.markdown(f"**Floating P&L:** :{profit_color}[${profit:+.2f}]")
        else:
            st.info("No open position right now.")

    with right:
        st.subheader("Strategy Metrics")
        m = metrics
        if m["total_trades"] > 0:
            col_a, col_b = st.columns(2)
            col_a.metric("Total Trades",   m["total_trades"])
            col_b.metric("Win Rate",        f"{m['win_rate']:.1%}")
            col_a.metric("Profit Factor",   f"{m['profit_factor']:.2f}")
            col_b.metric("Avg Confidence",  f"{m['avg_confidence']:.1%}")
        else:
            st.info("No completed trades yet.")

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
    st.caption("EUR/USD H1 Bot | XGBoost + Triple Barrier | Demo account only — not financial advice.")


if __name__ == "__main__":
    main()
