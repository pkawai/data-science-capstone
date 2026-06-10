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
- **The override saga is OVER:** the directional override / `calibrate_floor.py`
  was a band-aid for the real bug (models trained on Yahoo, traded on MT5 —
  fixed in train.py). The override is disabled in config.py AND config now
  IGNORES the `DIRECTIONAL_*` keys a stale `local_overrides.json` may contain.
  Do NOT recommend `calibrate_floor.py` — it is retired.
- **Jun 10 2026 code review** found+fixed 4 more real bugs (branch
  `fix/review-bugfixes`): see "Review fixes" in the history below. Models must
  be RETRAINED on Windows to benefit from the label fix.

---

## What to tell the user to do RIGHT NOW (Windows)

```cmd
cd "C:\Users\my tech\Projects\data-science-capstone\bot"
git pull origin main
del local_overrides.json       # stale band-aid file, if present (now ignored anyway)
python train.py EURUSD
python train.py GBPUSD
python train.py USDJPY
python bot.py                  # Ctrl+C first if already running
```
- Retraining matters: the old labels were structurally Sell-skewed
  (EURUSD: 52% SELL / 34% BUY by construction; balanced ~33/34/33 after the
  two-sided label fix), and raw price-level features are dropped.
- The bot RUNS fine without retraining (old bundles stay compatible) — but it
  would still carry the old labels' Sell bias.
- Realised P&L now lands in `closed_trades.csv` (TP/SL/reason + profit) — use
  it, not trades.csv, to judge performance after a few days.

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
7. **HOLD-collapse.** Root cause was the train/serve FEED mismatch (trained on
   Yahoo, traded on MT5) — fixed in train.py (`source="auto"`). The directional
   override that was tried first is retired; config ignores its keys.
8. **Review fixes (Jun 10 2026, branch fix/review-bugfixes, w/ pytest suite):**
   (a) stale `local_overrides.json` could silently re-enable the retired
   override — config now blocklists `DIRECTIONAL_*` keys; (b) backtest off-by-
   one: the FIRST bar after entry was never TP/SL-checked → every sweep metric
   was biased; (c) restart wiped trade_state → breakeven re-fired and moved a
   trailed SL BACKWARD (or died on NO_CHANGES, killing trailing) → positions
   now resume in the trailing phase; (d) one-sided triple-barrier labels:
   "Sell" meant a 1.0-ATR fall before a 1.5-ATR rise — NOT the short trade the
   bot places (TP 1.5 below / SL 1.0 above) → labels now two-sided per trade
   geometry (old: 52% SELL / 34% BUY on EURUSD; new: balanced ~33/33/33);
   (e) raw price-level features (BB_upper/lower, MA_20/50) dropped from NEW
   training (date-proxy memorisation; old bundles unaffected); (f) realised
   P&L: bot logs its exit deals to `closed_trades.csv` every cycle.

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
