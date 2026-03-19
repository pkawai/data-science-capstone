# news_calendar.py — High-impact news event schedule and blackout checker
#
# Events covered: NFP (programmatic), FOMC, ECB, BOE
# Add new years / events by extending the lists below.
#
# NOTE: Verify exact dates against official sources:
#   FOMC: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
#   ECB:  https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html
#   BOE:  https://www.bankofengland.co.uk/monetary-policy/upcoming-dates

import calendar
from datetime import datetime, timedelta, timezone


# ── 2026 scheduled event times (UTC) ──────────────────────────────────────────
# Format: (month, day, hour, minute)

FOMC_2026 = [
    (1,  29, 19, 0),
    (3,  19, 18, 0),
    (5,   7, 18, 0),
    (6,  18, 18, 0),
    (7,  29, 18, 0),
    (9,  17, 18, 0),
    (10, 29, 18, 0),
    (12,  9, 19, 0),
]

ECB_2026 = [
    (1,  30, 13, 15),
    (3,   6, 13, 15),
    (4,  17, 12, 15),   # summer time → 12:15 UTC
    (6,   5, 12, 15),
    (7,  24, 12, 15),
    (9,  11, 12, 15),
    (10, 30, 13, 15),
    (12, 18, 13, 15),
]

BOE_2026 = [
    (2,   6, 12, 0),
    (3,  20, 12, 0),
    (5,   7, 11, 0),   # BST → 11:00 UTC
    (6,  18, 11, 0),
    (8,   6, 11, 0),
    (9,  17, 11, 0),
    (11,  5, 12, 0),
    (12, 17, 12, 0),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _first_friday(year: int, month: int) -> int:
    """Return the day-of-month of the first Friday in the given month."""
    cal = calendar.monthcalendar(year, month)
    for week in cal:
        if week[calendar.FRIDAY] != 0:
            return week[calendar.FRIDAY]
    raise ValueError(f"No Friday found in {year}-{month}")


def _build_event_list(year: int) -> list[datetime]:
    """Return all high-impact event datetimes for the given year (UTC)."""
    events = []

    # NFP — first Friday of every month at 12:30 UTC
    for month in range(1, 13):
        day = _first_friday(year, month)
        events.append(datetime(year, month, day, 12, 30, tzinfo=timezone.utc))

    # FOMC, ECB, BOE (only coded for 2026; extend as needed)
    if year == 2026:
        for (m, d, h, mn) in FOMC_2026:
            events.append(datetime(year, m, d, h, mn, tzinfo=timezone.utc))
        for (m, d, h, mn) in ECB_2026:
            events.append(datetime(year, m, d, h, mn, tzinfo=timezone.utc))
        for (m, d, h, mn) in BOE_2026:
            events.append(datetime(year, m, d, h, mn, tzinfo=timezone.utc))

    return events


# Cache per year so we don't recompute every candle
_event_cache: dict[int, list[datetime]] = {}


def _get_events(year: int) -> list[datetime]:
    if year not in _event_cache:
        _event_cache[year] = _build_event_list(year)
    return _event_cache[year]


# ── Public API ─────────────────────────────────────────────────────────────────

def is_news_blackout(dt: datetime, blackout_minutes: int = 60) -> bool:
    """
    Return True if `dt` falls within ±blackout_minutes of any high-impact
    news event.  `dt` must be timezone-aware (UTC).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    window = timedelta(minutes=blackout_minutes)
    events = _get_events(dt.year)
    # Also check adjacent years in case we're near a year boundary
    if dt.month == 12:
        events = events + _get_events(dt.year + 1)
    elif dt.month == 1:
        events = events + _get_events(dt.year - 1)

    for event_dt in events:
        if abs(dt - event_dt) <= window:
            return True
    return False


def next_event(dt: datetime) -> tuple[datetime, str]:
    """
    Return (event_datetime, description) of the next high-impact event after dt.
    Useful for logging.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    candidates = [(e, "event") for e in _get_events(dt.year) if e > dt]
    candidates += [(e, "event") for e in _get_events(dt.year + 1) if e > dt]
    if not candidates:
        return None, ""
    candidates.sort(key=lambda x: x[0])
    return candidates[0]
