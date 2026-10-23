import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import backtest


def _synthetic_closes(n=400, start=100.0, drift=0.0003, vol=0.015, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return pd.Series(prices, index=idx)


# ── Fill haircut ──────────────────────────────────────────────────────────────

def test_haircut_reduces_net_premium_when_spread_is_nonzero():
    closes = _synthetic_closes()
    vol = backtest._realized_vol(closes)

    no_haircut = backtest.backtest_dte(closes, vol, 30, 0.25, spread_pct=0.10, haircut_fraction=0.0)
    with_haircut = backtest.backtest_dte(closes, vol, 30, 0.25, spread_pct=0.10, haircut_fraction=1.0)

    assert no_haircut is not None and with_haircut is not None
    assert with_haircut.net_premium_dollars < no_haircut.net_premium_dollars
    assert with_haircut.avg_annualized_yield < no_haircut.avg_annualized_yield


def test_zero_spread_means_haircut_fraction_has_no_effect():
    closes = _synthetic_closes()
    vol = backtest._realized_vol(closes)

    a = backtest.backtest_dte(closes, vol, 30, 0.25, spread_pct=0.0, haircut_fraction=0.0)
    b = backtest.backtest_dte(closes, vol, 30, 0.25, spread_pct=0.0, haircut_fraction=1.0)

    assert a.net_premium_dollars == pytest.approx(b.net_premium_dollars)


def test_too_short_series_returns_none():
    closes = _synthetic_closes(n=10)  # shorter than REALIZED_VOL_WINDOW_DAYS warmup
    vol = backtest._realized_vol(closes)
    assert backtest.backtest_dte(closes, vol, 30, 0.25) is None


# ── Discrete-lot dollar math vs. trader.py's actual state machine ────────────
#
# trader.py's real invariants: shares change hands only via buy_shares()
# (lot start, called when no shares held) and the assignment branch of
# main.py's run_symbol_cycle (lot end, immediately followed by the next
# buy_shares() call at CURRENT market price) -- never via roll_call(), which
# only replaces the short call leg and never touches the share position.
# This test hand-verifies backtest_dte's dollar accounting against that
# shape using a fully deterministic 3-event cycle sequence (bypassing the
# Black-Scholes solver so the expected numbers are exact, not approximate):
#   event 0: SELL_TO_OPEN, assigned  -> lot 1 realizes (strike - entry)*100
#   event 1: SELL_TO_OPEN, not assigned -> lot 2 opens at event 0's exit price
#            (the re-buy price a real buy_shares() call would fill at)
#   event 2: ROLL, not assigned -> lot 2 continues; NO share transaction,
#            only the call leg changes -- matches roll_call() exactly
def test_discrete_lot_dollar_math_matches_trader_state_machine(monkeypatch):
    events = [
        {"pos": 0, "exit_pos": 1, "spot": 100.0, "strike": 110.0, "premium": 2.0, "spot_t": 120.0, "assigned": True},
        {"pos": 1, "exit_pos": 2, "spot": 120.0, "strike": 130.0, "premium": 3.0, "spot_t": 125.0, "assigned": False},
        {"pos": 2, "exit_pos": 3, "spot": 125.0, "strike": 135.0, "premium": 1.0, "spot_t": 90.0, "assigned": False},
    ]
    monkeypatch.setattr(backtest, "_simulate_cycles", lambda *a, **k: events)

    stats = backtest.backtest_dte(pd.Series(dtype=float), pd.Series(dtype=float), dte=30, target_delta=0.25)

    # lot 1 (realized): called away at strike 110, bought at 100 -> +$1000
    # lot 2 (still open at sim end): re-bought at 120 (event 0's spot_t, the
    # price buy_shares() would actually have filled at), marked at the final
    # spot_t of 90 -> -$3000. A model that kept compounding through the
    # assignment instead of re-pricing the lot would get a different number.
    assert stats.stock_pnl_dollars == pytest.approx(1000.0 + (90.0 - 120.0) * 100)
    assert stats.net_premium_dollars == pytest.approx((2.0 + 3.0 + 1.0) * 100)
    assert stats.total_pnl_dollars == pytest.approx(stats.stock_pnl_dollars + stats.net_premium_dollars)
    assert stats.assignment_rate == pytest.approx(1 / 3)

    # Equity rose to a peak after event 1 (1.20x) then fell to 0.86x after
    # the crash in event 2 -- a drawdown that only exists because lot 2's
    # shares were still held (never sold/rebought) across the ROLL.
    assert stats.max_drawdown_pct == pytest.approx(1 - 0.86 / 1.20, abs=1e-6)
    assert stats.risk_adjusted_return == pytest.approx(stats.total_return_pct / stats.max_drawdown_pct)


def test_risk_adjusted_return_floors_near_zero_drawdown_instead_of_passthrough(monkeypatch):
    """Equity only ever rises -- max_drawdown_pct is exactly 0, nothing real
    to divide by. Must floor at MIN_DRAWDOWN_FLOOR_PCT, NOT silently fall
    back to the raw, unpenalized total_return_pct (which would put a
    one-cycle lucky combo on a different, more generous scale than a
    well-evidenced combo with genuine drawdown)."""
    events = [
        {"pos": 0, "exit_pos": 1, "spot": 100.0, "strike": 110.0, "premium": 2.0, "spot_t": 115.0, "assigned": True},
    ]
    monkeypatch.setattr(backtest, "_simulate_cycles", lambda *a, **k: events)

    stats = backtest.backtest_dte(pd.Series(dtype=float), pd.Series(dtype=float), dte=30, target_delta=0.25)

    assert stats.max_drawdown_pct == 0.0
    assert stats.risk_adjusted_return == pytest.approx(stats.total_return_pct / backtest.MIN_DRAWDOWN_FLOOR_PCT)
    assert stats.risk_adjusted_return != pytest.approx(stats.total_return_pct)


def test_risk_adjusted_return_unaffected_by_floor_when_drawdown_is_real(monkeypatch):
    """Sanity check the floor doesn't kick in (and doesn't change the
    answer) once there's a real, above-floor drawdown -- same scenario as
    the state-machine test above, which has a genuine ~28% drawdown."""
    events = [
        {"pos": 0, "exit_pos": 1, "spot": 100.0, "strike": 110.0, "premium": 2.0, "spot_t": 120.0, "assigned": True},
        {"pos": 1, "exit_pos": 2, "spot": 120.0, "strike": 130.0, "premium": 3.0, "spot_t": 125.0, "assigned": False},
        {"pos": 2, "exit_pos": 3, "spot": 125.0, "strike": 135.0, "premium": 1.0, "spot_t": 90.0, "assigned": False},
    ]
    monkeypatch.setattr(backtest, "_simulate_cycles", lambda *a, **k: events)

    stats = backtest.backtest_dte(pd.Series(dtype=float), pd.Series(dtype=float), dte=30, target_delta=0.25)

    assert stats.max_drawdown_pct > backtest.MIN_DRAWDOWN_FLOOR_PCT
    assert stats.risk_adjusted_return == pytest.approx(stats.total_return_pct / stats.max_drawdown_pct)


def test_single_lot_never_assigned_has_no_hidden_share_transaction(monkeypatch):
    """A run with zero assignments should price exactly like one continuous
    holding from the first entry to the final mark -- proving ROLL events
    never trigger a buy/sell of the underlying, matching roll_call()."""
    events = [
        {"pos": 0, "exit_pos": 1, "spot": 100.0, "strike": 115.0, "premium": 1.5, "spot_t": 105.0, "assigned": False},
        {"pos": 1, "exit_pos": 2, "spot": 105.0, "strike": 120.0, "premium": 1.2, "spot_t": 108.0, "assigned": False},
    ]
    monkeypatch.setattr(backtest, "_simulate_cycles", lambda *a, **k: events)

    stats = backtest.backtest_dte(pd.Series(dtype=float), pd.Series(dtype=float), dte=14, target_delta=0.25)

    assert stats.stock_pnl_dollars == pytest.approx((108.0 - 100.0) * 100)
    assert stats.assignment_rate == 0.0


# ── Ranking objective: risk-adjusted total return, not raw yield ────────────

def test_selection_prefers_risk_adjusted_return_over_raw_yield(monkeypatch):
    monkeypatch.setattr(backtest, "_price_history", lambda symbol, lookback=None: _synthetic_closes(n=500))

    def fake_backtest_dte(closes, vol, dte, target_delta, *a, **k):
        # DTE=7: highest raw yield, but a bad max_drawdown drags risk_adjusted
        # down. DTE=30: lower raw yield, but much better risk-adjusted return.
        # A yield-only ranking would pick 7; Task 2's ranking must pick 30.
        canned = {
            7: backtest.CycleStats(
                dte=7, num_cycles=20, total_premium_pct=0.5, avg_annualized_yield=0.80,
                assignment_rate=0.3, strategy_total_return=0.3, buy_and_hold_return=0.1,
                target_delta=target_delta, total_return_pct=0.15, max_drawdown_pct=0.30,
                risk_adjusted_return=0.5,
            ),
            30: backtest.CycleStats(
                dte=30, num_cycles=10, total_premium_pct=0.2, avg_annualized_yield=0.20,
                assignment_rate=0.2, strategy_total_return=0.18, buy_and_hold_return=0.1,
                target_delta=target_delta, total_return_pct=0.18, max_drawdown_pct=0.09,
                risk_adjusted_return=2.0,
            ),
        }.get(dte)
        return canned

    monkeypatch.setattr(backtest, "backtest_dte", fake_backtest_dte)

    result = backtest.backtest_symbol_total_return(
        "FAKE", dte_buckets=(7, 30), delta_buckets=(0.25,),
    )

    assert result.dte == 30
    assert result.risk_adjusted_return == 2.0


# ── Eligibility: MIN_CYCLES_FOR_SELECTION ────────────────────────────────────

def test_eligibility_excludes_zero_drawdown_fluke_in_favor_of_well_evidenced_combo(monkeypatch):
    """The MIN_DRAWDOWN_FLOOR_PCT floor alone isn't enough: a 2-cycle combo
    with zero observed drawdown and a HIGHER raw return would still score
    higher (total_return/floor) than a 12-cycle combo with a real drawdown
    and a LOWER raw return, and would win on score alone. Eligibility must
    exclude the 2-cycle combo from competing at all."""
    monkeypatch.setattr(backtest, "_price_history", lambda symbol, lookback=None: _synthetic_closes(n=500))

    fluke_score = 0.10 / backtest.MIN_DRAWDOWN_FLOOR_PCT  # = 5.0
    evidenced_score = 0.08 / 0.05                          # = 1.6
    assert fluke_score > evidenced_score  # sanity: without eligibility, the fluke would win

    def fake_backtest_dte(closes, vol, dte, target_delta, *a, **k):
        canned = {
            7: backtest.CycleStats(  # the fluke: 2 cycles, zero drawdown, HIGHER raw return
                dte=7, num_cycles=2, total_premium_pct=0.1, avg_annualized_yield=0.50,
                assignment_rate=0.5, strategy_total_return=0.10, buy_and_hold_return=0.05,
                target_delta=target_delta, total_return_pct=0.10, max_drawdown_pct=0.0,
                risk_adjusted_return=fluke_score,
            ),
            30: backtest.CycleStats(  # well-evidenced: 12 cycles, real drawdown, LOWER raw return
                dte=30, num_cycles=12, total_premium_pct=0.08, avg_annualized_yield=0.15,
                assignment_rate=0.2, strategy_total_return=0.08, buy_and_hold_return=0.05,
                target_delta=target_delta, total_return_pct=0.08, max_drawdown_pct=0.05,
                risk_adjusted_return=evidenced_score,
            ),
        }.get(dte)
        return canned

    monkeypatch.setattr(backtest, "backtest_dte", fake_backtest_dte)

    result = backtest.backtest_symbol_total_return(
        "FAKE", dte_buckets=(7, 30), delta_buckets=(0.25,), min_cycles_for_selection=5,
    )

    assert result.dte == 30
    assert result.num_cycles == 12


def test_ineligible_combo_is_logged_not_printed(monkeypatch, caplog):
    monkeypatch.setattr(backtest, "_price_history", lambda symbol, lookback=None: _synthetic_closes(n=500))

    def fake_backtest_dte(closes, vol, dte, target_delta, *a, **k):
        if dte == 7:
            return backtest.CycleStats(
                dte=7, num_cycles=2, total_premium_pct=0.1, avg_annualized_yield=0.50,
                assignment_rate=0.5, strategy_total_return=0.10, buy_and_hold_return=0.05,
                target_delta=target_delta, total_return_pct=0.10, max_drawdown_pct=0.01,
                risk_adjusted_return=5.0,
            )
        return backtest.CycleStats(
            dte=30, num_cycles=12, total_premium_pct=0.08, avg_annualized_yield=0.15,
            assignment_rate=0.2, strategy_total_return=0.08, buy_and_hold_return=0.05,
            target_delta=target_delta, total_return_pct=0.08, max_drawdown_pct=0.05,
            risk_adjusted_return=1.6,
        )

    monkeypatch.setattr(backtest, "backtest_dte", fake_backtest_dte)

    with caplog.at_level(logging.INFO, logger="backtest"):
        result = backtest.backtest_symbol_total_return(
            "FAKE", dte_buckets=(7, 30), delta_buckets=(0.25,), min_cycles_for_selection=5,
        )

    assert result.dte == 30
    messages = [rec.message for rec in caplog.records]
    assert any("DTE=7" in m and "ineligible for selection" in m for m in messages)


def test_symbol_dropped_with_warning_when_nothing_eligible(monkeypatch, caplog):
    monkeypatch.setattr(backtest, "_price_history", lambda symbol, lookback=None: _synthetic_closes(n=500))

    def fake_backtest_dte(closes, vol, dte, target_delta, *a, **k):
        return backtest.CycleStats(  # every combo has too few cycles
            dte=dte, num_cycles=1, total_premium_pct=0.05, avg_annualized_yield=0.30,
            assignment_rate=0.5, strategy_total_return=0.05, buy_and_hold_return=0.02,
            target_delta=target_delta, total_return_pct=0.05, max_drawdown_pct=0.0,
            risk_adjusted_return=0.05 / backtest.MIN_DRAWDOWN_FLOOR_PCT,
        )

    monkeypatch.setattr(backtest, "backtest_dte", fake_backtest_dte)

    with caplog.at_level(logging.WARNING, logger="backtest"):
        result = backtest.backtest_symbol_total_return(
            "FAKE", dte_buckets=(7, 30), delta_buckets=(0.25,), min_cycles_for_selection=5,
        )

    assert result is None
    messages = [rec.message for rec in caplog.records]
    assert any("dropping from the proposal" in m for m in messages)


# ── Walk-forward: in-sample vs. out-of-sample yield ──────────────────────────

def test_walk_forward_reports_is_oos_and_gap(monkeypatch):
    # Calm first 70%, then a regime shift (higher drift+vol) for the rest --
    # IS and OOS yields should genuinely differ, not just be duplicated.
    n = 600
    calm = _synthetic_closes(n=int(n * 0.7), start=100.0, drift=0.0001, vol=0.006, seed=1)
    wild_start = calm.iloc[-1]
    rng = np.random.default_rng(2)
    wild_idx = pd.date_range(calm.index[-1] + timedelta(days=1), periods=n - len(calm), freq="D")
    wild_prices = wild_start * np.exp(np.cumsum(rng.normal(0.004, 0.04, len(wild_idx))))
    wild = pd.Series(wild_prices, index=wild_idx)
    closes = pd.concat([calm, wild])

    monkeypatch.setattr(backtest, "_price_history", lambda symbol, lookback=None: closes)

    result = backtest.backtest_symbol_total_return("FAKE", dte_buckets=(14, 30), delta_buckets=(0.25,))

    assert result is not None
    assert result.in_sample_yield is not None
    assert result.out_of_sample_yield is not None
    assert result.oos_gap_pp == pytest.approx((result.in_sample_yield - result.out_of_sample_yield) * 100)
    # The regime shift should produce a real (non-zero) gap, not an
    # accidental exact match between the two windows.
    assert abs(result.oos_gap_pp) > 0.01


def test_oos_risk_adjusted_metrics_are_stored_not_just_yield(monkeypatch):
    """oos_gap_pp only validates avg_annualized_yield out of sample -- the
    OLD metric, not what selection actually optimizes for. These three
    fields must ALSO be populated so a reviewer (or apply_watchlist.py) can
    tell whether risk-adjusted performance itself held up out of sample."""
    n = 600
    calm = _synthetic_closes(n=int(n * 0.7), start=100.0, drift=0.0001, vol=0.006, seed=1)
    wild_start = calm.iloc[-1]
    rng = np.random.default_rng(2)
    wild_idx = pd.date_range(calm.index[-1] + timedelta(days=1), periods=n - len(calm), freq="D")
    wild_prices = wild_start * np.exp(np.cumsum(rng.normal(0.004, 0.04, len(wild_idx))))
    wild = pd.Series(wild_prices, index=wild_idx)
    closes = pd.concat([calm, wild])
    monkeypatch.setattr(backtest, "_price_history", lambda symbol, lookback=None: closes)

    result = backtest.backtest_symbol_total_return("FAKE", dte_buckets=(14, 30), delta_buckets=(0.25,))

    assert result.oos_total_return_pct is not None
    assert result.oos_max_drawdown_pct is not None
    assert result.oos_risk_adjusted_return is not None
    # Sanity: the OOS ratio should actually equal OOS return / OOS drawdown
    # (floored), not something derived from the in-sample numbers instead.
    assert result.oos_risk_adjusted_return == pytest.approx(
        result.oos_total_return_pct / max(result.oos_max_drawdown_pct, backtest.MIN_DRAWDOWN_FLOOR_PCT)
    )


def test_oos_metrics_none_when_test_slice_has_no_cycles(monkeypatch):
    monkeypatch.setattr(backtest, "_price_history", lambda symbol, lookback=None: _synthetic_closes(n=500))

    def fake_backtest_dte(closes, vol, dte, target_delta, *a, **k):
        # train slice (~350 rows, 70% of 500) gets a real result; test
        # slice (~150 rows) returns None, as if it had too little data for
        # even one cycle.
        if len(closes) > 200:
            return backtest.CycleStats(
                dte=dte, num_cycles=10, total_premium_pct=0.1, avg_annualized_yield=0.15,
                assignment_rate=0.2, strategy_total_return=0.1, buy_and_hold_return=0.05,
                target_delta=target_delta, total_return_pct=0.1, max_drawdown_pct=0.05,
                risk_adjusted_return=2.0,
            )
        return None

    monkeypatch.setattr(backtest, "backtest_dte", fake_backtest_dte)

    result = backtest.backtest_symbol_total_return("FAKE", dte_buckets=(30,), delta_buckets=(0.25,))

    assert result is not None
    assert result.out_of_sample_yield is None
    assert result.oos_total_return_pct is None
    assert result.oos_max_drawdown_pct is None
    assert result.oos_risk_adjusted_return is None


# ── DTE shrinkage toward the cross-symbol consensus ──────────────────────────

def _stats(dte, num_cycles, target_delta=0.25, intent=None, intent_is_default=None, **extra):
    return backtest.CycleStats(
        dte=dte, num_cycles=num_cycles, total_premium_pct=0.1, avg_annualized_yield=0.1,
        assignment_rate=0.2, strategy_total_return=0.1, buy_and_hold_return=0.1,
        target_delta=target_delta, spread_pct_used=0.05, spread_source="default",
        intent=intent, intent_is_default=intent_is_default, **extra,
    )


def test_shrinkage_pulls_low_cycle_symbol_toward_consensus_dte(monkeypatch):
    recommendations = {
        "AAA": _stats(dte=30, num_cycles=20),
        "BBB": _stats(dte=30, num_cycles=15),
        "CCC": _stats(dte=30, num_cycles=12),
        "OUTLIER": _stats(dte=7, num_cycles=2),  # below SHRINKAGE_MIN_CYCLES=8
    }

    captured = {}

    def fake_refresh(symbol, dte_buckets, delta_buckets, **kwargs):
        captured["dte_buckets"] = dte_buckets
        new_dte = dte_buckets[0]
        return _stats(dte=new_dte, num_cycles=2, target_delta=delta_buckets[0])

    monkeypatch.setattr(backtest, "backtest_symbol_total_return", fake_refresh)

    result = backtest.apply_dte_shrinkage(recommendations, min_cycles=8)

    # Trusted symbols (>= min_cycles) are untouched.
    assert result["AAA"].dte == 30 and result["AAA"].shrunk_from_dte is None
    assert result["BBB"].dte == 30
    assert result["CCC"].dte == 30

    # weight = 2/8 = 0.25 -> round(0.25*7 + 0.75*30) == 24
    assert result["OUTLIER"].dte == 24
    assert result["OUTLIER"].shrunk_from_dte == 7
    assert captured["dte_buckets"] == (24,)


def test_shrinkage_noop_when_no_symbol_is_trusted(caplog):
    recommendations = {"AAA": _stats(dte=7, num_cycles=1), "BBB": _stats(dte=30, num_cycles=2)}
    with caplog.at_level(logging.WARNING, logger="backtest"):
        result = backtest.apply_dte_shrinkage(recommendations, min_cycles=8)

    # Content-equal (nothing shrunk, dte/shrunk_from_dte unchanged) -- NOT
    # necessarily object-identical, since shrinkage is now grouped by
    # intent and always builds a fresh dict per group.
    assert result["AAA"].dte == 7 and result["AAA"].shrunk_from_dte is None
    assert result["BBB"].dte == 30 and result["BBB"].shrunk_from_dte is None
    assert any("skipping DTE shrinkage for this entire intent group" in rec.message for rec in caplog.records)


def test_shrinkage_logs_warning_and_keeps_original_dte_when_refresh_finds_no_cycles(monkeypatch, caplog):
    """If the refresh backtest at the shrunk DTE comes back empty, this must
    not silently keep the old stats with no trace -- it should log why, and
    leave the symbol explicitly unshrunk (shrunk_from_dte stays None, since
    the shrink didn't actually happen)."""
    recommendations = {
        "AAA": _stats(dte=30, num_cycles=20),
        "BBB": _stats(dte=30, num_cycles=15),
        "OUTLIER": _stats(dte=7, num_cycles=2),
    }
    monkeypatch.setattr(backtest, "backtest_symbol_total_return", lambda *a, **k: None)

    with caplog.at_level(logging.WARNING, logger="backtest"):
        result = backtest.apply_dte_shrinkage(recommendations, min_cycles=8)

    assert result["OUTLIER"].dte == 7  # unchanged -- refresh failed, fell back to the original
    assert result["OUTLIER"].shrunk_from_dte is None  # NOT marked as shrunk, since it wasn't
    messages = [rec.message for rec in caplog.records]
    assert any("OUTLIER" in m and "shrinkage wanted to move DTE" in m for m in messages)


# ── Intent: loading, resolution, delta bands ─────────────────────────────────

def test_load_intents_missing_file_returns_empty(tmp_path):
    assert backtest._load_intents(str(tmp_path / "nope.json")) == {}


def test_load_intents_reads_flat_symbol_to_intent_dict(tmp_path):
    path = tmp_path / "intents.json"
    path.write_text(json.dumps({"AAPL": "exit", "MSFT": "hold"}))
    assert backtest._load_intents(str(path)) == {"AAPL": "exit", "MSFT": "hold"}


def test_resolve_intent_explicit():
    intent, is_default = backtest._resolve_intent("AAPL", {"AAPL": "exit"})
    assert intent == "exit"
    assert is_default is False


def test_resolve_intent_missing_symbol_defaults(monkeypatch):
    intent, is_default = backtest._resolve_intent("MSFT", {"AAPL": "exit"})
    assert intent == backtest.DEFAULT_INTENT
    assert is_default is True


def test_resolve_intent_unrecognized_value_falls_back_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="backtest"):
        intent, is_default = backtest._resolve_intent("AAPL", {"AAPL": "sell_everything"})

    assert intent == backtest.DEFAULT_INTENT
    assert is_default is True
    assert any("unrecognized intent" in rec.message for rec in caplog.records)


def test_delta_buckets_for_intent_spans_the_band_inclusive():
    lo, hi = backtest.INTENT_DELTA_BANDS["exit"]
    buckets = backtest._delta_buckets_for_intent("exit")
    assert len(buckets) == backtest.INTENT_BAND_SWEEP_POINTS
    assert buckets[0] == pytest.approx(lo)
    assert buckets[-1] == pytest.approx(hi)


def test_delta_buckets_for_hold_and_exit_do_not_overlap():
    hold_buckets = backtest._delta_buckets_for_intent("hold")
    exit_buckets = backtest._delta_buckets_for_intent("exit")
    assert max(hold_buckets) < min(exit_buckets)


def test_delta_buckets_for_intent_raises_on_unrecognized_intent():
    """No silent fallback to DEFAULT_INTENT's band -- an invalid intent is
    a caller bug (it should have been caught by _resolve_intent already)
    and must surface loudly, not get quietly papered over."""
    with pytest.raises(ValueError, match="unrecognized intent"):
        backtest._delta_buckets_for_intent("sell_everything")


# ── Shrinkage: split consensus per intent group ──────────────────────────────

def test_shrinkage_computes_separate_consensus_per_intent_group():
    recommendations = {
        # hold group: consensus dte=30
        "H1": _stats(dte=30, num_cycles=20, intent="hold", intent_is_default=False),
        "H2": _stats(dte=30, num_cycles=15, intent="hold", intent_is_default=False),
        # exit group: consensus dte=7 -- a DIFFERENT consensus from hold's
        "E1": _stats(dte=7, num_cycles=20, intent="exit", intent_is_default=False),
        "E2": _stats(dte=7, num_cycles=15, intent="exit", intent_is_default=False),
    }
    result = backtest.apply_dte_shrinkage(recommendations, min_cycles=8)

    # All trusted (>= min_cycles) -- nothing should have moved, but this
    # proves the two groups were never blended into one global vote.
    assert result["H1"].dte == 30 and result["E1"].dte == 7


def test_shrinkage_skips_one_group_but_still_shrinks_the_other(monkeypatch, caplog):
    recommendations = {
        # hold group: plenty of trusted peers, consensus dte=30
        "H1": _stats(dte=30, num_cycles=20, intent="hold", intent_is_default=False),
        "H2": _stats(dte=30, num_cycles=15, intent="hold", intent_is_default=False),
        "H_OUTLIER": _stats(dte=7, num_cycles=2, intent="hold", intent_is_default=False),
        # exit group: NOBODY clears min_cycles -- nothing to anchor to.
        "E_OUTLIER": _stats(dte=14, num_cycles=1, intent="exit", intent_is_default=False),
    }

    captured = {}

    def fake_refresh(symbol, dte_buckets, delta_buckets, **kwargs):
        captured["intent"] = kwargs.get("intent")
        return _stats(dte=dte_buckets[0], num_cycles=2, target_delta=delta_buckets[0],
                      intent=kwargs.get("intent"), intent_is_default=False)

    monkeypatch.setattr(backtest, "backtest_symbol_total_return", fake_refresh)

    with caplog.at_level(logging.WARNING, logger="backtest"):
        result = backtest.apply_dte_shrinkage(recommendations, min_cycles=8)

    # hold's outlier shrank toward hold's own consensus (30).
    assert result["H_OUTLIER"].shrunk_from_dte == 7
    assert captured["intent"] == "hold"
    # exit group had nothing to anchor to -- skipped entirely, logged, and
    # left completely unshrunk (not pulled toward hold's consensus either).
    assert result["E_OUTLIER"].dte == 14
    assert result["E_OUTLIER"].shrunk_from_dte is None
    messages = [rec.message for rec in caplog.records]
    assert any("intent=exit" in m and "skipping DTE shrinkage for this entire intent group" in m for m in messages)


# ── Empirical outcome stats: synthetic path with KNOWN true probabilities ───

def _deterministic_drift_series(n=400, daily_drift=0.01, wiggle_amplitude=0.003, wiggle_period=17, start=100.0):
    """
    Fully deterministic (no randomness at all, reproducible bit-for-bit) --
    a strong constant daily drift plus a small periodic wiggle just large
    enough that realized vol is solvable (nonzero), small enough that it
    can never flip the sign of a single window's move. Every overlapping
    window therefore has the SAME deterministic outcome, so the "true"
    probabilities are known by construction, not estimated by sampling.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    i = np.arange(n)
    log_returns = daily_drift + wiggle_amplitude * np.sin(2 * np.pi * i / wiggle_period)
    prices = start * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=idx)


def test_empirical_stats_known_probabilities_on_deterministic_uptrend():
    closes = _deterministic_drift_series()
    vol = backtest._realized_vol(closes)
    dte = 30

    stats = backtest.compute_empirical_outcome_stats(closes, vol, dte, 0.25, 0.045, move_threshold_pct=0.05)

    assert stats is not None
    # The guaranteed-positive drift (~23% per 30-day window) vastly exceeds
    # any delta=0.25 strike's distance from spot at this (low, wiggle-only)
    # realized vol -- every window ends called away, by construction, not
    # by sampling luck. True P(called away) = 1.0, true P(small move) = 0.0.
    assert stats["p_called_away"] == pytest.approx(1.0)
    assert stats["p_expires_below_strike"] == pytest.approx(0.0)
    assert stats["p_small_move"] == pytest.approx(0.0)
    assert stats["median_move_pct"] == pytest.approx(0.2337, abs=1e-3)


def test_empirical_stats_effective_n_smaller_than_window_count():
    closes = _deterministic_drift_series()
    vol = backtest._realized_vol(closes)
    dte = 30
    step = max(1, round(dte * 5 / 7))

    stats = backtest.compute_empirical_outcome_stats(closes, vol, dte, 0.25, 0.045)

    assert stats["empirical_window_count"] > step
    assert stats["empirical_effective_n"] == stats["empirical_window_count"] // step
    assert stats["empirical_effective_n"] < stats["empirical_window_count"]


def test_empirical_stats_move_threshold_is_configurable():
    closes = _deterministic_drift_series()
    vol = backtest._realized_vol(closes)
    # The deterministic move is ~23% -- a huge threshold should catch it as "small".
    stats = backtest.compute_empirical_outcome_stats(closes, vol, 30, 0.25, 0.045, move_threshold_pct=0.50)
    assert stats["p_small_move"] == pytest.approx(1.0)
    assert stats["move_threshold_pct_used"] == 0.50


def test_empirical_stats_none_when_series_too_short():
    closes = _deterministic_drift_series(n=10)
    vol = backtest._realized_vol(closes)
    assert backtest.compute_empirical_outcome_stats(closes, vol, 30, 0.25, 0.045) is None


def test_backtest_dte_populates_empirical_fields(monkeypatch):
    closes = _synthetic_closes(n=400)
    vol = backtest._realized_vol(closes)
    stats = backtest.backtest_dte(closes, vol, 30, 0.25)

    assert stats.p_called_away is not None
    assert 0.0 <= stats.p_called_away <= 1.0
    assert stats.empirical_window_count is not None
    assert stats.empirical_effective_n is not None
    assert stats.empirical_effective_n <= stats.empirical_window_count
    assert stats.move_threshold_pct_used == backtest.EMPIRICAL_SMALL_MOVE_THRESHOLD_PCT


def test_backtest_symbol_total_return_populates_oos_empirical_fields(monkeypatch):
    closes = _deterministic_drift_series(n=600)
    monkeypatch.setattr(backtest, "_price_history", lambda symbol, lookback=None: closes)

    result = backtest.backtest_symbol_total_return("FAKE", dte_buckets=(30,), delta_buckets=(0.25,))

    assert result is not None
    assert result.p_called_away == pytest.approx(1.0)       # train
    assert result.oos_p_called_away == pytest.approx(1.0)    # OOS -- same deterministic series
    assert result.oos_empirical_window_count is not None
    assert result.oos_empirical_effective_n is not None


# ── Turnover score (exit-intent selection) ───────────────────────────────────

def test_realized_stock_gain_dollars_excludes_open_tail(monkeypatch):
    """Same 3-event scenario as the trader-state-machine test: lot 1 is
    assigned (realized), lot 2 is still open at the end (unrealized).
    realized_stock_gain_dollars must count ONLY lot 1 -- stock_pnl_dollars
    (already tested elsewhere) counts both and must differ from it."""
    events = [
        {"pos": 0, "exit_pos": 1, "spot": 100.0, "strike": 110.0, "premium": 2.0, "spot_t": 120.0, "assigned": True},
        {"pos": 1, "exit_pos": 2, "spot": 120.0, "strike": 130.0, "premium": 3.0, "spot_t": 125.0, "assigned": False},
        {"pos": 2, "exit_pos": 3, "spot": 125.0, "strike": 135.0, "premium": 1.0, "spot_t": 90.0, "assigned": False},
    ]
    monkeypatch.setattr(backtest, "_simulate_cycles", lambda *a, **k: events)

    stats = backtest.backtest_dte(pd.Series(dtype=float), pd.Series(dtype=float), dte=30, target_delta=0.25)

    assert stats.realized_stock_gain_dollars == pytest.approx((110.0 - 100.0) * 100)
    assert stats.entry_capital_dollars == pytest.approx(100.0 * 100)
    assert stats.stock_pnl_dollars != pytest.approx(stats.realized_stock_gain_dollars)


def test_turnover_score_formula():
    stats = _stats(
        dte=30, num_cycles=20, p_called_away=0.5,
        net_premium_dollars=500.0, realized_stock_gain_dollars=100.0, entry_capital_dollars=10000.0,
    )
    score = backtest._turnover_score(stats, dte=30)
    # expected_days_to_exit = 30/0.5 = 60; numerator = 600; denom = 600,000
    assert score == pytest.approx(600.0 / 600_000.0)


_EXIT_ARGS_TAIL = (30, 0.40, 0.045, 0.05, 0.5, 0.05, 20, 0.25, 0.15)  # dte, delta, rate, spread, haircut, move_thr, min_n, max_dd, max_tail


def test_check_exit_eligibility_rejects_zero_p_called_away():
    train_stats = _stats(dte=30, num_cycles=20, p_called_away=0.0, empirical_effective_n=50)
    eligible, reason, oos = backtest._check_exit_eligibility(
        train_stats, pd.Series(dtype=float), pd.Series(dtype=float), *_EXIT_ARGS_TAIL,
    )
    assert eligible is False
    assert "p_called_away is 0" in reason
    assert oos is None


def test_check_exit_eligibility_rejects_low_effective_n():
    train_stats = _stats(dte=30, num_cycles=20, p_called_away=0.3, empirical_effective_n=5)
    eligible, reason, oos = backtest._check_exit_eligibility(
        train_stats, pd.Series(dtype=float), pd.Series(dtype=float), *_EXIT_ARGS_TAIL,
    )
    assert eligible is False
    assert "empirical_effective_n=5 < 20" in reason


def test_check_exit_eligibility_rejects_high_oos_drawdown(monkeypatch):
    train_stats = _stats(dte=30, num_cycles=20, p_called_away=0.3, empirical_effective_n=50)
    oos_stats = _stats(
        dte=30, num_cycles=10, max_drawdown_pct=0.40, stock_pnl_dollars=0.0,
        realized_stock_gain_dollars=0.0, entry_capital_dollars=10000.0,
    )
    monkeypatch.setattr(backtest, "backtest_dte", lambda *a, **k: oos_stats)

    eligible, reason, oos = backtest._check_exit_eligibility(
        train_stats, pd.Series(dtype=float), pd.Series(dtype=float), *_EXIT_ARGS_TAIL,
    )
    assert eligible is False
    assert "OOS max_drawdown" in reason
    assert oos is None


def test_check_exit_eligibility_rejects_high_open_tail_loss(monkeypatch):
    train_stats = _stats(dte=30, num_cycles=20, p_called_away=0.3, empirical_effective_n=50)
    # max_drawdown passes (0.05 < 0.25 ceiling); open-tail loss doesn't:
    # all of stock_pnl_dollars (-$2000) is unrealized (realized gain=0) on
    # $10,000 capital -> 20% loss, above the 15% ceiling.
    oos_stats = _stats(
        dte=30, num_cycles=10, max_drawdown_pct=0.05, stock_pnl_dollars=-2000.0,
        realized_stock_gain_dollars=0.0, entry_capital_dollars=10000.0,
    )
    monkeypatch.setattr(backtest, "backtest_dte", lambda *a, **k: oos_stats)

    eligible, reason, oos = backtest._check_exit_eligibility(
        train_stats, pd.Series(dtype=float), pd.Series(dtype=float), *_EXIT_ARGS_TAIL,
    )
    assert eligible is False
    assert "open-tail loss" in reason
    assert oos is None


def test_check_exit_eligibility_passes_and_returns_oos_candidate(monkeypatch):
    train_stats = _stats(dte=30, num_cycles=20, p_called_away=0.3, empirical_effective_n=50)
    oos_stats = _stats(
        dte=30, num_cycles=10, max_drawdown_pct=0.05, stock_pnl_dollars=200.0,
        realized_stock_gain_dollars=200.0, entry_capital_dollars=10000.0,
    )
    monkeypatch.setattr(backtest, "backtest_dte", lambda *a, **k: oos_stats)

    eligible, reason, oos = backtest._check_exit_eligibility(
        train_stats, pd.Series(dtype=float), pd.Series(dtype=float), *_EXIT_ARGS_TAIL,
    )
    assert eligible is True
    assert reason == ""
    assert oos is oos_stats


def test_exit_intent_selection_uses_turnover_score_not_risk_adjusted(monkeypatch):
    monkeypatch.setattr(backtest, "_price_history", lambda symbol, lookback=None: _synthetic_closes(n=500))

    def fake_backtest_dte(closes, vol, dte, target_delta, *a, **k):
        base = dict(
            total_premium_pct=0.1, avg_annualized_yield=0.3, assignment_rate=0.3,
            strategy_total_return=0.1, buy_and_hold_return=0.05, target_delta=target_delta,
            total_return_pct=0.1, max_drawdown_pct=0.05, risk_adjusted_return=2.0,
            p_called_away=0.5, empirical_effective_n=40, empirical_window_count=100,
            entry_capital_dollars=10000.0, stock_pnl_dollars=0.0, realized_stock_gain_dollars=0.0,
        )
        # dte=7: turnover = 500/(10000*(7/0.5)) ~= 0.00357
        # dte=30: turnover = 100/(10000*(30/0.5)) ~= 0.000167 -- dte=7 wins
        premium = 500.0 if dte == 7 else 100.0
        return backtest.CycleStats(dte=dte, num_cycles=20, net_premium_dollars=premium, **base)

    monkeypatch.setattr(backtest, "backtest_dte", fake_backtest_dte)

    result = backtest.backtest_symbol_total_return(
        "FAKE", dte_buckets=(7, 30), delta_buckets=(0.40,), intent="exit",
    )

    assert result is not None
    assert result.dte == 7
    assert result.score_used == "turnover_score"
    assert result.turnover_score == pytest.approx(500.0 / (10000.0 * (7 / 0.5)))


def test_hold_intent_selection_still_uses_risk_adjusted_return(monkeypatch):
    monkeypatch.setattr(backtest, "_price_history", lambda symbol, lookback=None: _synthetic_closes(n=500))
    stats = _stats(dte=30, num_cycles=20, risk_adjusted_return=1.5)
    monkeypatch.setattr(backtest, "backtest_dte", lambda *a, **k: stats)

    result = backtest.backtest_symbol_total_return(
        "FAKE", dte_buckets=(30,), delta_buckets=(0.25,), intent="hold",
    )

    assert result.score_used == "risk_adjusted_return"
    assert result.turnover_score is None


def test_exit_intent_drops_symbol_when_nothing_passes_eligibility(monkeypatch, caplog):
    monkeypatch.setattr(backtest, "_price_history", lambda symbol, lookback=None: _synthetic_closes(n=500))
    # Every combo has p_called_away == 0 -- universally ineligible for exit.
    stats = _stats(dte=30, num_cycles=20, p_called_away=0.0, empirical_effective_n=50)
    monkeypatch.setattr(backtest, "backtest_dte", lambda *a, **k: stats)

    with caplog.at_level(logging.WARNING, logger="backtest"):
        result = backtest.backtest_symbol_total_return(
            "FAKE", dte_buckets=(30,), delta_buckets=(0.40,), intent="exit",
        )

    assert result is None
    assert any("dropping from the proposal" in rec.message for rec in caplog.records)


# ── ORACLE_MIN_DTE floor ──────────────────────────────────────────────────────

def test_apply_oracle_min_dte_floor_drops_buckets_below_floor():
    kept, dropped = backtest.apply_oracle_min_dte_floor((7, 14, 21, 30, 45), floor=14)
    assert kept == (14, 21, 30, 45)
    assert dropped == [7]


def test_apply_oracle_min_dte_floor_keeps_everything_when_all_clear():
    kept, dropped = backtest.apply_oracle_min_dte_floor((14, 21, 30), floor=14)
    assert kept == (14, 21, 30)
    assert dropped == []


def test_apply_oracle_min_dte_floor_drops_everything_below_floor():
    kept, dropped = backtest.apply_oracle_min_dte_floor((7, 10), floor=14)
    assert kept == ()
    assert dropped == [7, 10]


def test_apply_oracle_min_dte_floor_uses_config_default():
    import config
    kept, dropped = backtest.apply_oracle_min_dte_floor((7, 14, 21))
    assert dropped == [7]
    assert kept == (14, 21)
    assert config.ORACLE_MIN_DTE == 14
