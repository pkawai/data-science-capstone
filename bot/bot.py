#!/usr/bin/env python
# bot.py — Multi-pair live trading loop (run on Windows with MT5)
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
    "time", "symbol", "direction", "entry", "sl", "tp",
    "lots", "ticket", "confidence", "adx", "atr",
]


# ── Data writers ───────────────────────────────────────────────────────────────

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

def _wait_for_candle_close() -> None:
    now = datetime.now(timezone.utc)
    wait = 3600 - (now.minute * 60 + now.second) + 5
    logger.info(f"Waiting {wait}s for candle close…")
    time.sleep(wait)


# ── Open-position management ───────────────────────────────────────────────────

def _manage_open_positions(all_positions: list[dict], trade_state: dict) -> None:
    """
    Called once per H1 bar for every open position.

    1. Time-based exit  — close if position has been open >= MAX_BARS_IN_TRADE bars.
    2. Breakeven        — move SL to entry once unrealized profit >= initial risk.
    3. Trailing stop    — once in breakeven, trail SL by TRAIL_ATR_MULT × ATR.
    """
    now_utc      = datetime.now(timezone.utc)
    open_tickets = {pos["ticket"] for pos in all_positions}

    # Clean up state for positions that were closed by SL/TP
    for stale in [t for t in list(trade_state) if t not in open_tickets]:
        del trade_state[stale]

    for pos in all_positions:
        ticket     = pos["ticket"]
        symbol_mt5 = pos["symbol"]
        direction  = pos["type"]      # 2=Buy, 0=Sell
        price_open = pos["price_open"]
        current_sl = pos["sl"]
        current_tp = pos["tp"]

        # Reverse-lookup config symbol key from MT5 symbol name
        symbol = next((k for k, v in config.PAIR_CONFIGS.items()
                       if v["mt5_symbol"] == symbol_mt5), None)
        if symbol is None:
            continue

        decimals = config.PAIR_CONFIGS[symbol]["decimals"]

        # ── Time-based exit ──────────────────────────────────────────────────
        time_open    = datetime.fromtimestamp(pos["time_open"], tz=timezone.utc)
        bars_elapsed = (now_utc - time_open).total_seconds() / 3600

        if bars_elapsed >= config.MAX_BARS_IN_TRADE:
            logger.info(f"[{symbol}] Time exit: {bars_elapsed:.0f} bars open >= "
                        f"{config.MAX_BARS_IN_TRADE}. Closing ticket {ticket}.")
            mt5ex.close_position(ticket)
            trade_state.pop(ticket, None)
            continue

        # ── Initialise state for newly detected positions ────────────────────
        if ticket not in trade_state:
            trade_state[ticket] = {
                "initial_sl":       current_sl,
                "breakeven_active": False,
            }

        ts           = trade_state[ticket]
        initial_sl   = ts["initial_sl"]
        initial_risk = abs(price_open - initial_sl)   # price units

        # ── Fetch current ATR for this symbol ────────────────────────────────
        try:
            raw_df = mt5ex.get_latest_candles(symbol, n=50)
            df     = build_features(raw_df)
            atr    = df.iloc[-1]["ATR"]
        except Exception as e:
            logger.warning(f"[{symbol}] Could not get ATR for position management: {e}")
            continue

        ask, bid = mt5ex.get_current_price(symbol)

        # Unrealized profit in price units (positive = winning)
        if direction == 2:   # Buy
            current_profit = bid - price_open
        else:                # Sell
            current_profit = price_open - ask

        # ── Breakeven ────────────────────────────────────────────────────────
        if not ts["breakeven_active"] and current_profit >= initial_risk:
            be_sl = round(price_open, decimals)
            if mt5ex.modify_position(ticket, sl=be_sl, tp=current_tp):
                ts["breakeven_active"] = True
                current_sl = be_sl
                logger.info(f"[{symbol}] Breakeven: SL moved to entry {be_sl} "
                            f"(ticket {ticket}).")

        # ── Trailing stop ─────────────────────────────────────────────────────
        if ts["breakeven_active"]:
            trail_dist = config.TRAIL_ATR_MULT * atr
            if direction == 2:   # Buy: trail SL up
                trail_sl = round(bid - trail_dist, decimals)
                if trail_sl > current_sl:
                    if mt5ex.modify_position(ticket, sl=trail_sl, tp=current_tp):
                        logger.info(f"[{symbol}] Trail SL → {trail_sl} (Buy, "
                                    f"ticket {ticket}).")
            else:                # Sell: trail SL down
                trail_sl = round(ask + trail_dist, decimals)
                if trail_sl < current_sl:
                    if mt5ex.modify_position(ticket, sl=trail_sl, tp=current_tp):
                        logger.info(f"[{symbol}] Trail SL → {trail_sl} (Sell, "
                                    f"ticket {ticket}).")


# ── Per-pair logic ─────────────────────────────────────────────────────────────

def _process_pair(symbol: str, model, current_balance: float, utc_hour: int) -> None:
    """Run one full cycle for a single pair."""
    label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

    # ── Check if already in trade for this pair ────────────────────────────
    open_positions = mt5ex.get_open_positions(symbol)
    if len(open_positions) >= config.MAX_OPEN_TRADES:
        logger.info(f"[{symbol}] Already in trade. Skipping.")
        return

    # ── Fetch data + features ──────────────────────────────────────────────
    try:
        raw_df = mt5ex.get_latest_candles(symbol, n=200)
        df     = build_features(raw_df)
    except Exception as e:
        logger.error(f"[{symbol}] Data error: {e}")
        return

    if len(df) < 10:
        logger.warning(f"[{symbol}] Not enough feature rows. Skipping.")
        return

    feat_cols = get_feature_columns(df)
    X         = df[feat_cols]
    last_row  = df.iloc[-1]
    adx_value = last_row.get("ADX", 0)
    vol_ratio = last_row.get("Vol_ratio", 1)

    # ── Regime / volatility filters ────────────────────────────────────────
    if adx_value < config.ADX_THRESHOLD:
        logger.info(f"[{symbol}] ADX={adx_value:.1f} — ranging, skip.")
        return
    if vol_ratio > config.VOL_RATIO_MAX:
        logger.info(f"[{symbol}] Vol spike ({vol_ratio:.2f}), skip.")
        return

    # ── Prediction ─────────────────────────────────────────────────────────
    signal, confidence = predict_signal(model, X)
    logger.info(f"[{symbol}] Signal: {label_map[signal]}  "
                f"Confidence: {confidence:.1%}  ADX: {adx_value:.1f}")

    if signal == 1:
        return   # Hold

    # ── Calculate SL / TP / lots ────────────────────────────────────────────
    atr      = last_row["ATR"]
    ask, bid = mt5ex.get_current_price(symbol)
    entry    = ask if signal == 2 else bid
    sl, tp   = calculate_sl_tp(entry, signal, atr, symbol)
    lots     = calculate_position_size(current_balance, atr, symbol)

    logger.info(f"[{symbol}] Placing {label_map[signal]}  "
                f"entry={entry}  sl={sl}  tp={tp}  lots={lots}")

    # ── Execute ────────────────────────────────────────────────────────────
    try:
        ticket = mt5ex.place_order(
            symbol=symbol,
            direction=signal,
            lots=lots,
            sl=sl,
            tp=tp,
            comment=f"bot_{symbol}_{label_map[signal].lower()}_{confidence:.2f}",
        )
        logger.info(f"[{symbol}] Order filled. Ticket: {ticket}")
        _write_trade({
            "time":       datetime.now(timezone.utc).isoformat(),
            "symbol":     symbol,
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
        logger.error(f"[{symbol}] Order failed: {e}")


# ── Main loop ──────────────────────────────────────────────────────────────────

def run():
    logger.info("=" * 55)
    logger.info("  Multi-Pair Trading Bot — Starting")
    logger.info(f"  Pairs: {config.PAIRS}")
    logger.info("=" * 55)

    mt5ex.connect()
    _init_trades_csv()

    # Load all models at startup
    models = {}
    for symbol in config.PAIRS:
        try:
            models[symbol] = load(symbol)
        except Exception as e:
            logger.error(f"Could not load model for {symbol}: {e}. Skipping this pair.")

    if not models:
        raise RuntimeError("No models loaded. Run train.py for each pair first.")

    logger.info(f"Models loaded: {list(models.keys())}")

    account_balance = config.ACCOUNT_BALANCE
    daily_start_day = datetime.now(timezone.utc).date()
    trade_state     = {}   # per-ticket state for trailing stop / breakeven

    try:
        while True:
            _wait_for_candle_close()

            now_utc  = datetime.now(timezone.utc)
            utc_hour = now_utc.hour
            today    = now_utc.date()

            if today != daily_start_day:
                daily_start_day = today
                logger.info("New trading day.")

            # ── Get current state ────────────────────────────────────────
            try:
                current_balance = mt5ex.get_account_balance()
            except Exception:
                current_balance = account_balance

            live_daily_pnl = current_balance - account_balance
            all_positions  = mt5ex.get_open_positions()

            # ── Manage open positions (time exit + trailing stop) ─────────
            try:
                _manage_open_positions(all_positions, trade_state)
            except Exception as e:
                logger.error(f"Position management error: {e}")

            # ── Write state for dashboard ────────────────────────────────
            _write_state({
                "last_updated":   now_utc.isoformat(),
                "balance":        current_balance,
                "daily_pnl":      live_daily_pnl,
                "open_positions": all_positions,
                "pairs_trading":  list(models.keys()),
                "utc_hour":       utc_hour,
                "bot_running":    True,
            })

            # ── Daily loss limit ─────────────────────────────────────────
            if check_daily_limit(live_daily_pnl, account_balance):
                logger.warning(f"Daily loss limit hit ({live_daily_pnl:.0f} USD). Skipping.")
                continue

            # ── Session filter ───────────────────────────────────────────
            if utc_hour not in config.ACTIVE_HOURS:
                logger.info(f"Outside active session (UTC {utc_hour:02d}:xx). Skipping.")
                continue

            # ── Process each pair ────────────────────────────────────────
            for symbol, model in models.items():
                try:
                    _process_pair(symbol, model, current_balance, utc_hour)
                except Exception as e:
                    logger.error(f"[{symbol}] Unexpected error: {e}")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down.")
    finally:
        _write_state({"bot_running": False,
                      "last_updated": datetime.now(timezone.utc).isoformat()})
        mt5ex.disconnect()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    run()
