#!/usr/bin/env python
# bot.py — Main live trading loop (run on Windows with MT5)
#
# Usage (Windows):
#   1. Open MetaTrader 5 and log in to your demo account.
#   2. Fill in MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in config.py
#   3. python bot.py

import csv
import json
import logging
import os
import time
from datetime import datetime, timezone

import config
from features      import build_features, get_feature_columns
from model         import load, predict_signal
from risk_manager  import calculate_position_size, calculate_sl_tp, check_daily_limit
import mt5_executor as mt5ex

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

TRADES_CSV = "trades.csv"
STATE_JSON = "state.json"

TRADE_FIELDS = [
    "time", "direction", "entry", "sl", "tp",
    "lots", "ticket", "confidence", "adx", "atr",
]


# ── Data writers (for dashboard) ───────────────────────────────────────────────

def _init_trades_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()


def _write_trade(row: dict):
    with open(TRADES_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADE_FIELDS).writerow(row)


def _write_state(state: dict):
    with open(STATE_JSON, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_active_session(utc_hour: int) -> bool:
    return utc_hour in config.ACTIVE_HOURS


def _wait_for_candle_close() -> None:
    now = datetime.now(timezone.utc)
    seconds_to_next_hour = 3600 - (now.minute * 60 + now.second)
    wait = seconds_to_next_hour + 5
    logger.info(f"Waiting {wait}s for candle close…")
    time.sleep(wait)


def _get_daily_pnl(account_balance: float) -> float:
    try:
        return mt5ex.get_account_balance() - account_balance
    except Exception:
        return 0.0


# ── Main loop ──────────────────────────────────────────────────────────────────

def run():
    logger.info("=" * 55)
    logger.info("  EUR/USD H1 Trading Bot — Starting")
    logger.info("=" * 55)

    mt5ex.connect()
    logger.info("MT5 connected.")

    model = load(config.MODEL_PATH)
    logger.info(f"Model loaded from {config.MODEL_PATH}")

    _init_trades_csv()

    account_balance = config.ACCOUNT_BALANCE
    daily_start_day = datetime.now(timezone.utc).date()
    last_signal     = "STARTING"

    try:
        while True:
            _wait_for_candle_close()

            now_utc  = datetime.now(timezone.utc)
            utc_hour = now_utc.hour
            today    = now_utc.date()

            if today != daily_start_day:
                daily_start_day = today
                logger.info("New trading day — daily P&L reset.")

            # ── Get current state ─────────────────────────────────────────
            try:
                current_balance = mt5ex.get_account_balance()
            except Exception:
                current_balance = account_balance

            live_daily_pnl = current_balance - account_balance
            open_positions = mt5ex.get_open_positions()
            in_trade       = len(open_positions) > 0
            current_pos    = open_positions[0] if in_trade else None

            # ── Write state for dashboard ─────────────────────────────────
            _write_state({
                "last_updated":  now_utc.isoformat(),
                "balance":       current_balance,
                "daily_pnl":     live_daily_pnl,
                "in_trade":      in_trade,
                "open_position": current_pos,
                "last_signal":   last_signal,
                "utc_hour":      utc_hour,
                "bot_running":   True,
            })

            # ── Daily loss limit ──────────────────────────────────────────
            if check_daily_limit(live_daily_pnl, account_balance):
                logger.warning(f"Daily loss limit hit ({live_daily_pnl:.0f} USD). Skipping.")
                last_signal = "DAILY LIMIT HIT"
                continue

            # ── Session filter ────────────────────────────────────────────
            if not _is_active_session(utc_hour):
                logger.info(f"Outside active session (UTC {utc_hour:02d}:xx). Skipping.")
                last_signal = "OUT OF SESSION"
                continue

            # ── Already in trade ──────────────────────────────────────────
            if in_trade:
                logger.info(f"Already in trade (ticket={current_pos['ticket']}). Skipping.")
                continue

            # ── Fetch data + features ─────────────────────────────────────
            try:
                raw_df = mt5ex.get_latest_candles(n=200)
                df     = build_features(raw_df)
            except Exception as e:
                logger.error(f"Data/feature error: {e}")
                continue

            if len(df) < 10:
                logger.warning("Not enough rows after feature build. Skipping.")
                continue

            feat_cols = get_feature_columns(df)
            X         = df[feat_cols]
            last_row  = df.iloc[-1]
            adx_value = last_row.get("ADX", 0)
            vol_ratio = last_row.get("Vol_ratio", 1)

            # ── Regime / volatility filters ───────────────────────────────
            if adx_value < config.ADX_THRESHOLD:
                logger.info(f"ADX={adx_value:.1f} — ranging market, skip.")
                last_signal = f"SKIP (ADX={adx_value:.1f})"
                continue
            if vol_ratio > config.VOL_RATIO_MAX:
                logger.info(f"Vol_ratio={vol_ratio:.2f} — vol spike, skip.")
                last_signal = f"SKIP (vol spike)"
                continue

            # ── Prediction ────────────────────────────────────────────────
            signal, confidence = predict_signal(model, X)
            label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            last_signal = f"{label_map[signal]} ({confidence:.1%})"

            logger.info(f"Signal: {label_map[signal]}  Confidence: {confidence:.1%}  ADX: {adx_value:.1f}")

            if signal == 1:
                logger.info("Hold — no trade.")
                continue

            # ── Calculate SL / TP / lots ──────────────────────────────────
            atr      = last_row["ATR"]
            ask, bid = mt5ex.get_current_price()
            entry    = ask if signal == 2 else bid
            sl, tp   = calculate_sl_tp(entry, signal, atr)
            lots     = calculate_position_size(current_balance, atr)

            logger.info(f"Placing {label_map[signal]}  entry={entry:.5f}  sl={sl:.5f}  tp={tp:.5f}  lots={lots}")

            # ── Execute ───────────────────────────────────────────────────
            try:
                ticket = mt5ex.place_order(
                    direction=signal,
                    lots=lots,
                    sl=sl,
                    tp=tp,
                    comment=f"bot_{label_map[signal].lower()}_{confidence:.2f}",
                )
                logger.info(f"Order filled. Ticket: {ticket}")
                _write_trade({
                    "time":       now_utc.isoformat(),
                    "direction":  label_map[signal],
                    "entry":      entry,
                    "sl":         sl,
                    "tp":         tp,
                    "lots":       lots,
                    "ticket":     ticket,
                    "confidence": round(confidence, 4),
                    "adx":        round(adx_value, 2),
                    "atr":        round(atr, 5),
                })
            except RuntimeError as e:
                logger.error(f"Order failed: {e}")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down.")
    finally:
        _write_state({"bot_running": False, "last_updated": datetime.now(timezone.utc).isoformat()})
        mt5ex.disconnect()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    run()
