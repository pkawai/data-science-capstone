"""
generate_demo_data.py — Creates trades.csv + state.json for demo purposes on Mac.
Run once before `streamlit run dashboard.py`.

Usage:
    python generate_demo_data.py
"""

import json
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
START_BALANCE = 5_000
N_TRADES = 42

# Realistic H1 price ranges (approximate current levels)
PAIR_BASE = {
    "EURUSD": 1.08500,
    "GBPUSD": 1.27000,
    "USDJPY": 150.500,
}
PAIR_PIP = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
}
PAIR_PIP_VALUE = {
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "USDJPY": 6.7,
}

# Generate trade history over the past ~3 months
start_dt = datetime.now(timezone.utc) - timedelta(days=90)

trades = []
current_price = {p: PAIR_BASE[p] for p in PAIRS}

for i in range(N_TRADES):
    symbol = random.choice(PAIRS)
    direction = random.choice(["BUY", "SELL"])
    entry = current_price[symbol] + np.random.normal(0, PAIR_PIP[symbol] * 5)
    atr = PAIR_PIP[symbol] * random.uniform(8, 18)
    sl_dist = atr * 1.0
    tp_dist = atr * 1.5

    if direction == "BUY":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    # Drift price slightly
    current_price[symbol] += np.random.normal(0, PAIR_PIP[symbol] * 3)

    confidence = random.uniform(0.65, 0.95)
    adx = random.uniform(26, 55)
    lots = round(random.uniform(0.05, 0.20), 2)
    ticket = 100_000 + i

    # Space trades ~1-3 days apart, within active hours
    hour_offset = timedelta(days=i * 90 / N_TRADES, hours=random.randint(7, 17))
    trade_time = start_dt + hour_offset

    trades.append({
        "time": trade_time.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "direction": direction,
        "entry": round(entry, 5 if PAIR_PIP[symbol] == 0.0001 else 3),
        "sl": round(sl, 5 if PAIR_PIP[symbol] == 0.0001 else 3),
        "tp": round(tp, 5 if PAIR_PIP[symbol] == 0.0001 else 3),
        "lots": lots,
        "ticket": ticket,
        "confidence": round(confidence, 4),
        "adx": round(adx, 2),
        "atr": round(atr, 6),
    })

trades_df = pd.DataFrame(trades)
trades_df.to_csv("trades.csv", index=False)
print(f"✅ Wrote {len(trades_df)} trades to trades.csv")

# Compute rough running balance (win ~55% of trades)
balance = START_BALANCE
for _, row in trades_df.iterrows():
    pip = PAIR_PIP[row["symbol"]]
    pip_val = PAIR_PIP_VALUE[row["symbol"]]
    tp_pips = abs(row["tp"] - row["entry"]) / pip
    sl_pips = abs(row["sl"] - row["entry"]) / pip
    won = random.random() < 0.55
    pnl = (tp_pips if won else -sl_pips) * pip_val * row["lots"]
    balance += pnl

daily_pnl = random.uniform(-80, 150)

# Build open positions (1-2 live positions)
open_positions = []
for sym in random.sample(PAIRS, k=random.randint(1, 2)):
    base = PAIR_BASE[sym]
    direction_type = random.choice([2, 1])  # 2=BUY, 1=SELL
    open_price = base + np.random.normal(0, PAIR_PIP[sym] * 10)
    current = open_price + np.random.normal(0, PAIR_PIP[sym] * 15)
    atr = PAIR_PIP[sym] * random.uniform(10, 20)
    if direction_type == 2:  # BUY
        sl = open_price - atr
        tp = open_price + atr * 1.5
        profit = (current - open_price) / PAIR_PIP[sym] * PAIR_PIP_VALUE[sym] * 0.10
    else:  # SELL
        sl = open_price + atr
        tp = open_price - atr * 1.5
        profit = (open_price - current) / PAIR_PIP[sym] * PAIR_PIP_VALUE[sym] * 0.10

    open_positions.append({
        "symbol": sym,
        "type": direction_type,
        "price_open": round(open_price, 5 if PAIR_PIP[sym] == 0.0001 else 3),
        "price_current": round(current, 5 if PAIR_PIP[sym] == 0.0001 else 3),
        "sl": round(sl, 5 if PAIR_PIP[sym] == 0.0001 else 3),
        "tp": round(tp, 5 if PAIR_PIP[sym] == 0.0001 else 3),
        "profit": round(profit, 2),
        "volume": 0.10,
        "ticket": 200_000 + len(open_positions),
    })

state = {
    "bot_running": True,
    "balance": round(balance, 2),
    "daily_pnl": round(daily_pnl, 2),
    "in_trade": len(open_positions) > 0,
    "open_positions": open_positions,
    "pairs_trading": PAIRS,
    "last_signal": random.choice(["BUY EURUSD", "SELL GBPUSD", "HOLD USDJPY", "BUY USDJPY"]),
    "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
}

with open("state.json", "w") as f:
    json.dump(state, f, indent=2)

print(f"✅ Wrote state.json  (balance=${balance:,.0f}, {len(open_positions)} open positions)")
print(f"\nNow run:  streamlit run dashboard.py")
