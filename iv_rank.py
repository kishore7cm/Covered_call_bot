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
from config import MIN_IV_HISTORY_POINTS


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
