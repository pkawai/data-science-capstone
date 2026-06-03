# Forex Bot — Handoff / Context for a New Claude Session

**Read this first.** It is the full story of debugging this bot so a new session
can continue without re-deriving everything. Owner: Orgil (Ulaanbaatar, UTC+8).
School has ended — the bot is now used for real on a **demo account**, not graded.

---

## TL;DR — where things stand

- The bot is a multi-pair (EURUSD, GBPUSD, USDJPY) H1 ML trading bot on
  MetaTrader 5. Repo: **github.com/pkawai/data-science-capstone** (the `bot/`
  subfolder). Windows clone path: `C:\Users\my tech\Projects\data-science-capstone\bot`.
- It runs on **Windows** (MT5 is Windows-only). Broker: **ICMarketsSC-Demo**,
  account 52772690, broker clock **UTC+3**. Bot active hours 07–17 UTC =
  **15:00–01:00 Ulaanbaatar**.
- **The problem the whole time:** "bot never takes a trade." It was caused by a
  STACK of separate bugs, all now fixed (see history below).
- **The last/open issue:** the model is HOLD-heavy and its *directional*
  confidence is low, so it rarely trades. Fix = a directional override whose
  threshold must be **calibrated to the user's own Windows models** via
  `calibrate_floor.py`. As of this handoff the user needs to RUN that on Windows
  and report the output. The current default `DIRECTIONAL_FLOOR = 0.50` is from
  Mac models and is likely too high for the Windows models (their directional
  confidence looked ~0.20–0.30 in the live log).

---

## What to tell the user to do RIGHT NOW (Windows)

```cmd
cd "C:\Users\my tech\Projects\data-science-capstone\bot"
git pull origin main
python calibrate_floor.py      # loads THEIR models, writes local_overrides.json
python bot.py                  # Ctrl+C first if already running
```
- `calibrate_floor.py` prints a per-pair table (directional confidence + trades/
  day + win% per candidate floor) and writes the chosen floor to
  `local_overrides.json` (gitignored, machine-specific). `config.py` auto-loads
  that file at startup, so no config editing is needed.
- At bot startup, the line `[config] Applied local_overrides.json: [...]`
  confirms the calibrated floor is active.
- **Ask the user to paste the `calibrate_floor.py` table.** If it says
  "No positive-edge floor found on any pair", the Windows models are too weak →
  the right move is a clean retrain (`python train.py EURUSD/GBPUSD/USDJPY`),
  NOT more threshold fiddling.

---

## How to check if it traded (don't watch live — it stresses the user)

```cmd
python -c "import pandas as pd; d=pd.read_csv('trades.csv'); print('TOTAL:', len(d)); print(d.tail())"
```
Expected once calibrated: **~1–2 trades/day/pair**. Zero hours of HOLD are
normal in between. Judge over days, not hours.

---

## The bug history (each was real, each is fixed & pushed)

1. **200-bar live fetch** broke a Daily-200-EMA feature (`D1_EMA200_dist`) →
   model fed garbage → defaulted to HOLD. Fixed: live fetch **12,000** H1 bars.
2. **ADX gate too high (25).** Lowered to **20** (sweep showed more trades, equal/
   better profit factor).
3. **THE BIG ONE — wrong timeframe.** `MT5_TIMEFRAME` was `16388` (**H4**),
   mislabeled H1. MT5 encodes H1=16385, H4=16388. Bot fetched 4-hour candles
   while waking hourly → "frozen candle" (timestamps always 08:00/12:00/16:00).
   Fixed: `MT5_TIMEFRAME = 16385`.
4. **MT5 feed diagnosis.** Added `mt5_doctor.py` / `check_feed.py` /
   `check_tick.py`. Confirmed feed healthy; verified H1 candles FRESH on broker
   clock (UTC+3). connect() now reports link state.
5. **Confidence threshold unreachable.** Optuna stored per-model thresholds of
   0.75–0.80; USDJPY's 0.80 is ABOVE the model's max confidence (~0.76) → could
   NEVER trade. The stored value also silently overrode config edits. Fixed:
   `CONFIDENCE_CAP = 0.65`; `model.load()` clamps to it.
6. **Audit fixes:** (a) position-management fetched only 50 candles →
   build_features collapsed → **breakeven/trailing stops silently never ran**;
   now 12,000. (b) `current_balance` NameError if first cycle crosses midnight
   UTC; seeded upfront. (c) dashboard P&L assumed every trade hit TP + used
   EURUSD pip math for all pairs (USDJPY ~100x off, win rate always 100%); now
   honest (no fake win-rate/PF from the exit-less live CSV).
7. **HOLD-collapse (current).** Model leans Buy/Sell ~85% of bars but at ~0.45–
   0.55 directional confidence (LOWER on the Windows models), so argmax+0.65
   gate collapses to HOLD → 4 live days, 0 trades. Fix: `DIRECTIONAL_OVERRIDE`
   (take stronger of Buy/Sell over `DIRECTIONAL_FLOOR`, ignore HOLD, bypass meta
   veto). Floor must be calibrated per-machine (`calibrate_floor.py`).

---

## Current config (bot/config.py key values)

```
MT5_TIMEFRAME        = 16385   # H1 (was 16388 = H4)
ADX_THRESHOLD        = 20      # was 25
CONFIDENCE_CAP       = 0.65    # clamps Optuna's 0.75-0.80 stored thresholds
DIRECTIONAL_OVERRIDE = True
DIRECTIONAL_FLOOR    = 0.50    # DEFAULT — calibrate per-machine via local_overrides.json
```
`local_overrides.json` (gitignored) overrides these at load; written by
`calibrate_floor.py`.

---

## Diagnostic tools (all in bot/, run on Windows with MT5 open unless noted)

| Tool | What it answers |
|---|---|
| `mt5_doctor.py` | Is the broker link up? account/server? tick vs candle freshness (broker time)? timeframe sane? |
| `check_feed.py` | Is each pair's candle FRESH, and would the ADX gate open? |
| `why_no_trade.py` | Per-pair: full Buy/Hold/Sell probs + every gate OK/NO + which gate blocked. **Best first tool when "why no trade?"** |
| `calibrate_floor.py` | Tunes DIRECTIONAL_FLOOR to the local models; writes local_overrides.json. |
| `replay_live.py` | (Mac/yfinance) replays the live decision path over N days, counts would-trades + where they die. |
| `sweep_thresholds.py` / `sweep_confidence.py` / `analyze_hold_bias.py` | Backtest sweeps behind the ADX / confidence / directional decisions. |
| `diagnose_adx.py` | (Mac) ADX distribution / how often the gate opens. |

---

## Honest caveats for the new session

- **Profit factor is realistically ~1.4–1.9**, NOT the ~2.9 the old README claims
  (those were inflated). Demo account only; past ≠ future.
- **Don't keep adding features.** The user (rightly) wants it WORKING and stable,
  not more tools. If trades still don't fire after calibration, the answer is
  likely **retrain the Windows models**, not another threshold.
- **Gotcha — nested git repo:** `Capstone/bot/` was itself an old git repo
  pointing at a now-deleted `FOREXBOT` remote. Commit/push from the Capstone
  ROOT (`/Users/orgilbk/Claude/Capstone`), staging `bot/<file>`. Pushing from
  inside `bot/` fails with "Repository not found".
- **Don't edit config.py on Windows** (causes pull conflicts). The bot uses the
  already-logged-in MT5 terminal (MT5_LOGIN=0). Use `local_overrides.json` for
  per-machine values.
- PC must stay awake + MT5 logged in during active hours, or no trades. A VPS is
  the eventual fix (not set up yet).
