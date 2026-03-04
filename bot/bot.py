#!/usr/bin/env python
# bot.py — Main live trading loop (run on Windows with MT5)
#
# Usage (Windows):
#   1. Open MetaTrader 5 and log in to your demo account.
#   2. Fill in MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in config.py
#   3. python bot.py

import logging
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_active_session(utc_hour: int) -> bool:
    return utc_hour in config.ACTIVE_HOURS


def _wait_for_candle_close() -> None:
    """Sleep until the current H1 candle closes (top of the next hour)."""
    now = datetime.now(timezone.utc)
    seconds_to_next_hour = 3600 - (now.minute * 60 + now.second)
    # Add 5 seconds buffer to make sure the candle is fully formed on the broker side
    wait = seconds_to_next_hour + 5
    logger.info(f"Waiting {wait}s for candle close…")
    time.sleep(wait)


def _get_daily_pnl(account_balance: float) -> float:
    """Rough daily P&L: current balance minus the configured starting balance."""
    try:
        current_balance = mt5ex.get_account_balance()
        return current_balance - account_balance
    except Exception:
        return 0.0


# ── Main loop ──────────────────────────────────────────────────────────────────

def run():
    logger.info("=" * 55)
    logger.info("  EUR/USD H1 Trading Bot — Starting")
    logger.info("=" * 55)

    # Connect to MT5
    mt5ex.connect()
    logger.info("MT5 connected.")

    # Load pre-trained model
    model = load(config.MODEL_PATH)
    logger.info(f"Model loaded from {config.MODEL_PATH}")

    account_balance = config.ACCOUNT_BALANCE
    daily_pnl       = 0.0
    daily_start_day = datetime.now(timezone.utc).date()

    try:
        while True:
            _wait_for_candle_close()

            now_utc    = datetime.now(timezone.utc)
            utc_hour   = now_utc.hour
            today      = now_utc.date()

            # Reset daily P&L tracker at midnight UTC
            if today != daily_start_day:
                daily_pnl       = 0.0
                daily_start_day = today
                logger.info("New trading day — daily P&L reset.")

            # ── Daily loss limit check ────────────────────────────────────
            live_daily_pnl = _get_daily_pnl(account_balance)
            if check_daily_limit(live_daily_pnl, account_balance):
                logger.warning(
                    f"Daily loss limit hit ({live_daily_pnl:.0f} USD). "
                    "Skipping until tomorrow."
                )
                continue

            # ── Session filter ────────────────────────────────────────────
            if not _is_active_session(utc_hour):
                logger.info(f"Outside active session (UTC {utc_hour:02d}:xx). Skipping.")
                continue

            # ── Existing positions ────────────────────────────────────────
            open_positions = mt5ex.get_open_positions()
            if len(open_positions) >= config.MAX_OPEN_TRADES:
                logger.info(f"Max open trades reached ({len(open_positions)}). Skipping.")
                continue

            # ── Fetch data + features ─────────────────────────────────────
            try:
                raw_df = mt5ex.get_latest_candles(n=200)
            except Exception as e:
                logger.error(f"Failed to fetch candles: {e}")
                continue

            try:
                df = build_features(raw_df)
            except Exception as e:
                logger.error(f"Feature engineering failed: {e}")
                continue

            if len(df) < 10:
                logger.warning("Not enough feature rows after dropna. Skipping.")
                continue

            feat_cols = get_feature_columns(df)
            X = df[feat_cols]

            # ── Regime / volatility filters ───────────────────────────────
            last_row  = df.iloc[-1]
            adx_value = last_row.get("ADX", 0)
            vol_ratio = last_row.get("Vol_ratio", 1)

            if adx_value < config.ADX_THRESHOLD:
                logger.info(f"ADX={adx_value:.1f} < {config.ADX_THRESHOLD} — ranging market, skip.")
                continue
            if vol_ratio > config.VOL_RATIO_MAX:
                logger.info(f"Vol_ratio={vol_ratio:.2f} — volatility spike, skip.")
                continue

            # ── Prediction ────────────────────────────────────────────────
            signal, confidence = predict_signal(model, X)

            label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            logger.info(
                f"Signal: {label_map[signal]}  "
                f"Confidence: {confidence:.1%}  "
                f"ADX: {adx_value:.1f}  UTC hour: {utc_hour}"
            )

            if signal == 1:
                logger.info("Hold — no trade.")
                continue

            # ── Calculate SL / TP / position size ─────────────────────────
            atr        = last_row["ATR"]
            ask, bid   = mt5ex.get_current_price()
            entry      = ask if signal == 2 else bid

            sl, tp = calculate_sl_tp(entry, signal, atr)

            try:
                current_balance = mt5ex.get_account_balance()
            except Exception:
                current_balance = account_balance

            lots = calculate_position_size(current_balance, atr)

            logger.info(
                f"Placing {label_map[signal]}  "
                f"entry={entry:.5f}  sl={sl:.5f}  tp={tp:.5f}  "
                f"lots={lots}  ATR={atr:.5f}"
            )

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
            except RuntimeError as e:
                logger.error(f"Order failed: {e}")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down.")
    finally:
        mt5ex.disconnect()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    run()
