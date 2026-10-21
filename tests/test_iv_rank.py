from config import DAILY_IVR_MIN_HISTORY_POINTS
from iv_rank import iv_rank_from_daily_history


def test_insufficient_history_below_threshold():
    history = [0.3] * (DAILY_IVR_MIN_HISTORY_POINTS - 1)
    assert iv_rank_from_daily_history(history) == "insufficient history"


def test_insufficient_history_empty():
    assert iv_rank_from_daily_history([]) == "insufficient history"


def test_exact_threshold_is_sufficient():
    # len == threshold should NOT be flagged insufficient (only < threshold is).
    history = [0.2 + 0.01 * (i % 5) for i in range(DAILY_IVR_MIN_HISTORY_POINTS)]
    result = iv_rank_from_daily_history(history)
    assert result != "insufficient history"
    assert isinstance(result, float)


def test_rank_at_max_is_100():
    history = [0.2] * (DAILY_IVR_MIN_HISTORY_POINTS - 1) + [0.5]
    assert iv_rank_from_daily_history(history) == 100.0


def test_rank_at_min_is_0():
    history = [0.5] * (DAILY_IVR_MIN_HISTORY_POINTS - 1) + [0.2]
    assert iv_rank_from_daily_history(history) == 0.0


def test_rank_uses_most_recent_as_current_not_max_of_series():
    # Current (last element) is below the series max -- rank should land
    # strictly between 0 and 100, not pinned to either end.
    history = [0.2] * (DAILY_IVR_MIN_HISTORY_POINTS - 2) + [0.5, 0.3]
    result = iv_rank_from_daily_history(history)
    assert 0.0 < result < 100.0


def test_no_variation_returns_none_not_insufficient():
    history = [0.25] * DAILY_IVR_MIN_HISTORY_POINTS
    assert iv_rank_from_daily_history(history) is None
