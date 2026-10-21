"""
IV Rank and IV Percentile — two distinct standard metrics, both answering
"is current IV high or low relative to this symbol's own recent history,"
computed differently:

  IV Rank       = (current - min) / (max - min) * 100
                  Where does today sit between the lookback period's low
                  and high? Sensitive to outliers (one spike sets the max).

  IV Percentile = % of days in the lookback where IV was below today's level
                  More robust to a single outlier day than IV Rank.

Both require accumulated history (see logger.py's iv_history table). This
bot only snapshots once per weekly cycle, so useful history builds slowly —
see MIN_IV_HISTORY_POINTS in config.py. Both functions return None if there
isn't enough history yet rather than producing a number that looks
meaningful but isn't.
"""
from config import DAILY_IVR_MIN_HISTORY_POINTS, MIN_IV_HISTORY_POINTS


def iv_rank(current_iv: float, history: list[float]) -> float | None:
    if len(history) < MIN_IV_HISTORY_POINTS:
        return None

    lo, hi = min(history), max(history)
    if hi == lo:
        return None  # no variation to rank against

    return (current_iv - lo) / (hi - lo) * 100.0


def iv_percentile(current_iv: float, history: list[float]) -> float | None:
    if len(history) < MIN_IV_HISTORY_POINTS:
        return None

    below = sum(1 for h in history if h < current_iv)
    return below / len(history) * 100.0


def iv_rank_from_daily_history(history: list[float]) -> float | str:
    """
    IV Rank computed from iv_logger.py's daily history (atm_iv values,
    oldest first, most recent = history[-1] = "current"), gated on
    DAILY_IVR_MIN_HISTORY_POINTS (252, ~1 trading year) rather than
    MIN_IV_HISTORY_POINTS (8) — daily snapshots make a real 1-year lookback
    reachable in ~a year, so there's no reason to accept the weekly gate's
    noisier early threshold here. Returns the literal string
    "insufficient history" (not None) below that threshold, to make the two
    "not enough data" and "no variation in the data" cases distinguishable
    at the call site. Still returns None for the latter — current == lo ==
    hi, same degenerate case iv_rank() above already treats as unrankable.
    """
    if len(history) < DAILY_IVR_MIN_HISTORY_POINTS:
        return "insufficient history"

    current = history[-1]
    lo, hi = min(history), max(history)
    if hi == lo:
        return None  # no variation to rank against

    return (current - lo) / (hi - lo) * 100.0
