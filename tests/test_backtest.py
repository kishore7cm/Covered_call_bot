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

def _stats(dte, num_cycles, target_delta=0.25):
    return backtest.CycleStats(
        dte=dte, num_cycles=num_cycles, total_premium_pct=0.1, avg_annualized_yield=0.1,
        assignment_rate=0.2, strategy_total_return=0.1, buy_and_hold_return=0.1,
        target_delta=target_delta, spread_pct_used=0.05, spread_source="default",
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


def test_shrinkage_noop_when_no_symbol_is_trusted():
    recommendations = {"AAA": _stats(dte=7, num_cycles=1), "BBB": _stats(dte=30, num_cycles=2)}
    result = backtest.apply_dte_shrinkage(recommendations, min_cycles=8)
    assert result is recommendations


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
