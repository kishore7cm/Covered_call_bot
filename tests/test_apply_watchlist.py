import json
from datetime import date, timedelta

import apply_watchlist as aw
from config import MAX_CAPITAL_PER_SYMBOL, MAX_LIVE_DELTA, MIN_LIVE_DELTA, ROLL_MIN_DTE, SHARES_PER_LOT, TARGET_DELTA


# ── _intent_summary ───────────────────────────────────────────────────────────

def test_intent_summary_explicit():
    stats = {"intent": "exit", "intent_is_default": False}
    assert aw._intent_summary(stats) == "intent=exit (explicit)"


def test_intent_summary_default():
    stats = {"intent": "hold", "intent_is_default": True}
    summary = aw._intent_summary(stats)
    assert "intent=hold" in summary
    assert "default" in summary
    assert "not listed in intents.json" in summary


def test_intent_summary_na_for_entries_predating_intent_tracking():
    assert aw._intent_summary({}) == "intent: n/a (predates intent tracking)"


# ── _diff() shows intent and flags a default ─────────────────────────────────

def test_diff_add_shows_explicit_intent(capsys):
    active = _watchlist({})
    proposed = _watchlist({
        "AAPL": {"dte": 30, "avg_annualized_yield": 0.18, "intent": "exit", "intent_is_default": False},
    })
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "intent=exit (explicit)" in out


def test_diff_add_shows_default_intent_distinctly(capsys):
    active = _watchlist({})
    proposed = _watchlist({
        "AAPL": {"dte": 30, "avg_annualized_yield": 0.18, "intent": "hold", "intent_is_default": True},
    })
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "intent=hold" in out
    assert "default" in out


def test_diff_change_fires_on_intent_change_even_if_dte_unchanged(capsys):
    active = _watchlist({
        "AAPL": {"dte": 30, "avg_annualized_yield": 0.18, "intent": "hold", "intent_is_default": True},
    })
    proposed = _watchlist({
        "AAPL": {"dte": 30, "avg_annualized_yield": 0.18, "intent": "exit", "intent_is_default": False},
    })
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "CHANGE:" in out
    assert "~ AAPL: intent hold -> exit" in out


# ── _is_oos_summary ───────────────────────────────────────────────────────────

def test_summary_full_stats_no_flag():
    stats = {"in_sample_yield": 0.20, "out_of_sample_yield": 0.18, "oos_gap_pp": 2.0}
    summary = aw._is_oos_summary(stats)
    assert "IS yield=20.0%" in summary
    assert "OOS yield=18.0%" in summary
    assert "gap=+2.0pp" in summary
    assert "LARGE GAP" not in summary


def test_summary_flags_large_positive_gap_as_overfit_risk():
    stats = {"in_sample_yield": 0.40, "out_of_sample_yield": 0.10, "oos_gap_pp": 30.0}
    summary = aw._is_oos_summary(stats)
    assert "LARGE GAP" in summary
    assert "in-sample beat out-of-sample" in summary


def test_summary_flags_large_negative_gap_with_correct_direction():
    stats = {"in_sample_yield": 0.05, "out_of_sample_yield": 0.25, "oos_gap_pp": -20.0}
    summary = aw._is_oos_summary(stats)
    assert "LARGE GAP" in summary
    assert "out-of-sample beat in-sample" in summary


def test_summary_at_threshold_boundary_not_flagged():
    stats = {"in_sample_yield": 0.20, "out_of_sample_yield": 0.10, "oos_gap_pp": aw.OOS_GAP_FLAG_THRESHOLD_PP}
    assert "LARGE GAP" not in aw._is_oos_summary(stats)


def test_summary_degrades_gracefully_for_pre_task2_watchlist_entries():
    """An older proposed/active_watchlist.json written before Task 2 has no
    in_sample_yield/out_of_sample_yield/oos_gap_pp keys at all."""
    stats = {"dte": 30, "avg_annualized_yield": 0.15}
    summary = aw._is_oos_summary(stats)
    assert summary == "IS yield=n/a OOS yield=n/a"


# ── _risk_adjusted_summary ────────────────────────────────────────────────────

def test_risk_adjusted_summary_no_flag_when_close():
    stats = {"risk_adjusted_return": 2.0, "oos_risk_adjusted_return": 1.8}
    summary = aw._risk_adjusted_summary(stats)
    assert "IS risk-adjusted=2.00 OOS risk-adjusted=1.80" == summary


def test_risk_adjusted_summary_flags_degradation_below_threshold():
    # OOS (0.8) is 40% of IS (2.0) -- below the 50% threshold.
    stats = {"risk_adjusted_return": 2.0, "oos_risk_adjusted_return": 0.8}
    summary = aw._risk_adjusted_summary(stats)
    assert "fell to 40% of in-sample" in summary
    assert "!!" in summary


def test_risk_adjusted_summary_not_flagged_at_exactly_half():
    stats = {"risk_adjusted_return": 2.0, "oos_risk_adjusted_return": 1.0}
    summary = aw._risk_adjusted_summary(stats)
    assert "!!" not in summary


def test_risk_adjusted_summary_flags_sign_flip():
    stats = {"risk_adjusted_return": 1.5, "oos_risk_adjusted_return": -0.3}
    summary = aw._risk_adjusted_summary(stats)
    assert "went non-positive" in summary


def test_risk_adjusted_summary_empty_when_either_side_missing():
    assert aw._risk_adjusted_summary({"risk_adjusted_return": 2.0}) == ""
    assert aw._risk_adjusted_summary({"oos_risk_adjusted_return": 1.0}) == ""
    assert aw._risk_adjusted_summary({}) == ""


def test_risk_adjusted_summary_no_flag_when_in_sample_already_non_positive():
    # Nothing positive to "lose" -- in-sample was already bad, this isn't
    # the "backtest doesn't generalize" story being flagged for.
    stats = {"risk_adjusted_return": -0.5, "oos_risk_adjusted_return": -2.0}
    summary = aw._risk_adjusted_summary(stats)
    assert "!!" not in summary


# ── _diff() output ────────────────────────────────────────────────────────────

def _watchlist(symbols: dict) -> dict:
    return {"generated_at": "2026-10-21", "symbols": symbols}


def test_diff_add_shows_is_oos_summary(capsys):
    active = _watchlist({})
    proposed = _watchlist({
        "AAPL": {"dte": 30, "avg_annualized_yield": 0.18, "in_sample_yield": 0.18,
                 "out_of_sample_yield": 0.16, "oos_gap_pp": 2.0},
    })
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "ADD:" in out
    assert "+ AAPL: DTE=30" in out
    assert "IS yield=18.0% OOS yield=16.0%" in out
    # No risk-adjusted fields on this entry -- line must not appear at all.
    assert "risk-adjusted" not in out


def test_diff_add_shows_risk_adjusted_degradation_flag(capsys):
    active = _watchlist({})
    proposed = _watchlist({
        "SOFI": {"dte": 30, "avg_annualized_yield": 0.18, "in_sample_yield": 0.18,
                 "out_of_sample_yield": 0.17, "oos_gap_pp": 1.0,
                 "risk_adjusted_return": 2.0, "oos_risk_adjusted_return": 0.5},
    })
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    # Yield gap is small (no flag there), but risk-adjusted degraded hard --
    # this is exactly the "passes the yield check, fails on risk-adjusted"
    # case this flag exists for.
    assert "IS yield=18.0% OOS yield=17.0%" in out
    assert "LARGE GAP" not in out
    assert "IS risk-adjusted=2.00 OOS risk-adjusted=0.50" in out
    assert "fell to 25% of in-sample" in out


def test_diff_remove_shows_summary_from_active_stats(capsys):
    active = _watchlist({
        "KO": {"dte": 45, "avg_annualized_yield": 0.05, "in_sample_yield": 0.05,
               "out_of_sample_yield": 0.30, "oos_gap_pp": -25.0},
    })
    proposed = _watchlist({})
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "REMOVE:" in out
    assert "- KO" in out
    assert "LARGE GAP" in out
    assert "out-of-sample beat in-sample" in out


def test_diff_change_dte_shows_proposed_summary_not_active(capsys):
    active = _watchlist({
        "SOFI": {"dte": 7, "avg_annualized_yield": 0.60, "in_sample_yield": 0.60,
                 "out_of_sample_yield": 0.05, "oos_gap_pp": 55.0},
    })
    proposed = _watchlist({
        "SOFI": {"dte": 30, "avg_annualized_yield": 0.20, "in_sample_yield": 0.20,
                 "out_of_sample_yield": 0.19, "oos_gap_pp": 1.0},
    })
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "CHANGE:" in out
    assert "~ SOFI: DTE 7 -> 30" in out
    # The NEW (proposed, 30 DTE) summary is shown, not the old 7-DTE one --
    # a reviewer deciding whether to approve the change needs to see what
    # they're switching TO.
    assert "IS yield=20.0% OOS yield=19.0%" in out
    assert "LARGE GAP" not in out


def test_diff_no_differences(capsys):
    shared = {"dte": 30, "avg_annualized_yield": 0.15}
    active = _watchlist({"AAPL": dict(shared)})
    proposed = _watchlist({"AAPL": dict(shared)})
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "No differences" in out


# ── --confirm gating is unaffected by this task ──────────────────────────────
#
# Runs the actual CLI as a subprocess (not just _diff()) so this genuinely
# exercises the __main__ gating logic, not just the new display code added
# in Task 3 -- a regression guard for the hard rule that apply_watchlist.py
# --confirm stays the ONLY writer of active_watchlist.json.

import subprocess
import sys
from pathlib import Path

_SCRIPT = str(Path(__file__).resolve().parent.parent / "apply_watchlist.py")


def _write_proposed(tmp_path) -> Path:
    path = tmp_path / "proposed.json"
    path.write_text(json.dumps({
        "generated_at": "2026-10-21",
        # Not AAPL: that's in OTHER_BOTS_SYMBOLS, which apply_watchlist.py
        # now strips from every proposal before diffing/applying.
        "symbols": {"KO": {"dte": 30, "avg_annualized_yield": 0.18,
                              "in_sample_yield": 0.18, "out_of_sample_yield": 0.16, "oos_gap_pp": 2.0}},
    }))
    return path


def test_no_confirm_does_not_write_active_file(tmp_path):
    proposed_path = _write_proposed(tmp_path)
    active_path = tmp_path / "active.json"

    result = subprocess.run(
        [sys.executable, _SCRIPT, "--proposed", str(proposed_path), "--active", str(active_path)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0
    assert "dry run" in result.stdout
    assert "IS yield=18.0% OOS yield=16.0%" in result.stdout
    assert not active_path.exists()


def test_confirm_writes_active_file_matching_proposed(tmp_path):
    proposed_path = _write_proposed(tmp_path)
    active_path = tmp_path / "active.json"

    result = subprocess.run(
        [sys.executable, _SCRIPT, "--proposed", str(proposed_path), "--active", str(active_path), "--confirm"],
        capture_output=True, text=True,
    )

    assert result.returncode == 0
    assert "Applied" in result.stdout
    assert active_path.exists()

    active = json.loads(active_path.read_text())
    proposed = json.loads(proposed_path.read_text())
    # active matches proposed except for the added confirmed_at stamp
    # (Phase 3, Task 5) -- a human-approval timestamp, distinct from
    # proposed's own generated_at (when the Oracle produced these numbers).
    assert "confirmed_at" in active
    assert active["confirmed_at"] != proposed.get("generated_at")
    assert active["approved_by"] == "human"
    active_without_stamps = {k: v for k, v in active.items() if k not in ("confirmed_at", "approved_by")}
    assert active_without_stamps == proposed


# ── _empirical_summary: missing sibling fields show "n/a", never 0.0% ───────

def test_empirical_summary_all_fields_present():
    stats = {
        "p_called_away": 0.3, "p_expires_below_strike": 0.7, "p_small_move": 0.5,
        "median_move_pct": 0.01, "move_threshold_pct_used": 0.05,
        "empirical_window_count": 100, "empirical_effective_n": 20,
    }
    summary = aw._empirical_summary(stats, "empirical (train)")
    assert "P(called away)=30.0%" in summary
    assert "P(expires below strike)=70.0%" in summary
    assert "median move=+1.0%" in summary
    assert "n=100 windows, effective_n=20" in summary
    assert "=0.0%" not in summary  # nothing here should look like a measured zero


def test_empirical_summary_missing_sibling_shows_na_not_zero():
    """p_called_away present, but a sibling field is missing (shouldn't
    normally happen since backtest_dte sets them as one atomic group, but
    this must never silently render as a measured 0.0%)."""
    stats = {"p_called_away": 0.4, "move_threshold_pct_used": 0.05}
    summary = aw._empirical_summary(stats, "empirical (train)")
    assert "P(called away)=40.0%" in summary
    assert "P(expires below strike)=n/a" in summary
    assert "P(|move|<5%)=n/a" in summary
    assert "median move=n/a" in summary
    assert "n=n/a windows, effective_n=n/a" in summary
    assert "=0.0%" not in summary


def test_empirical_summary_empty_when_p_called_away_missing():
    assert aw._empirical_summary({}, "empirical (train)") == ""


# ── _load_earnings_by_symbol / _earnings_warning ─────────────────────────────

def _screened(results):
    return {"generated_at": "2026-10-21", "results": results}


def test_load_earnings_by_symbol_parses_known_dates():
    screened = _screened([
        {"symbol": "AAPL", "earnings_date": "2026-11-05"},
        {"symbol": "MSFT", "earnings_date": None},
        {"symbol": "KO"},  # no earnings_date key at all
    ])
    result = aw._load_earnings_by_symbol(screened)
    assert result == {"AAPL": date(2026, 11, 5)}


def test_load_earnings_by_symbol_skips_malformed_date(caplog):
    screened = _screened([{"symbol": "AAPL", "earnings_date": "not-a-date"}])
    assert aw._load_earnings_by_symbol(screened) == {}


def test_load_earnings_by_symbol_missing_results_key_returns_empty():
    assert aw._load_earnings_by_symbol({}) == {}


def test_earnings_warning_fires_when_inside_implied_window():
    earnings_date = date.today() + timedelta(days=10)
    warning = aw._earnings_warning("AAPL", proposed_dte=14, earnings_by_symbol={"AAPL": earnings_date})
    assert "earnings on" in warning
    assert earnings_date.isoformat() in warning


def test_earnings_warning_empty_when_outside_implied_window():
    earnings_date = date.today() + timedelta(days=30)
    warning = aw._earnings_warning("AAPL", proposed_dte=14, earnings_by_symbol={"AAPL": earnings_date})
    assert warning == ""


def test_earnings_warning_empty_when_no_known_earnings_date():
    assert aw._earnings_warning("AAPL", proposed_dte=14, earnings_by_symbol={}) == ""


def test_earnings_warning_boundary_exactly_at_implied_expiration():
    earnings_date = date.today() + timedelta(days=14)
    warning = aw._earnings_warning("AAPL", proposed_dte=14, earnings_by_symbol={"AAPL": earnings_date})
    assert warning != ""  # inclusive of the implied expiration date itself


def test_earnings_warning_empty_when_earnings_is_today():
    warning = aw._earnings_warning("AAPL", proposed_dte=14, earnings_by_symbol={"AAPL": date.today()})
    assert warning == ""  # strictly AFTER today, not today itself


# ── _diff() wires the earnings warning into ADD/CHANGE, not REMOVE ──────────

def test_diff_add_shows_earnings_warning(capsys):
    earnings_date = date.today() + timedelta(days=5)
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": 14, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed, earnings_by_symbol={"AAPL": earnings_date})
    out = capsys.readouterr().out
    assert "earnings on" in out
    assert earnings_date.isoformat() in out


def test_diff_add_no_earnings_warning_when_outside_window(capsys):
    earnings_date = date.today() + timedelta(days=90)
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": 14, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed, earnings_by_symbol={"AAPL": earnings_date})
    out = capsys.readouterr().out
    assert "earnings on" not in out


def test_diff_change_uses_proposed_dte_for_earnings_check(capsys):
    # Active's old DTE (60) would put earnings outside the window; the
    # PROPOSED dte (14) is what must actually be used for the check.
    earnings_date = date.today() + timedelta(days=10)
    active = _watchlist({"AAPL": {"dte": 60, "avg_annualized_yield": 0.10}})
    proposed = _watchlist({"AAPL": {"dte": 14, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed, earnings_by_symbol={"AAPL": earnings_date})
    out = capsys.readouterr().out
    assert "earnings on" in out


def test_diff_remove_never_shows_earnings_warning(capsys):
    # No new position is being opened on a REMOVE -- the warning must not
    # appear there even if earnings_by_symbol has an entry for it.
    earnings_date = date.today() + timedelta(days=5)
    active = _watchlist({"AAPL": {"dte": 14, "avg_annualized_yield": 0.18}})
    proposed = _watchlist({})
    aw._diff(active, proposed, earnings_by_symbol={"AAPL": earnings_date})
    out = capsys.readouterr().out
    assert "REMOVE:" in out
    assert "earnings on" not in out


def test_diff_without_earnings_by_symbol_arg_is_backward_compatible(capsys):
    """_diff() must still work when called the old way (no third arg) --
    existing callers/tests from before this fix shouldn't break."""
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": 14, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed)  # no earnings_by_symbol at all
    out = capsys.readouterr().out
    assert "+ AAPL" in out
    assert "earnings on" not in out


# ── _load_ex_div_by_symbol / _ex_div_warning ─────────────────────────────────

def test_load_ex_div_by_symbol_parses_known_dates():
    screened = _screened([
        {"symbol": "AAPL", "ex_dividend_date": "2026-11-05"},
        {"symbol": "MSFT", "ex_dividend_date": None},
        {"symbol": "KO"},  # no ex_dividend_date key at all
    ])
    result = aw._load_ex_div_by_symbol(screened)
    assert result == {"AAPL": date(2026, 11, 5)}


def test_load_ex_div_by_symbol_skips_malformed_date():
    screened = _screened([{"symbol": "AAPL", "ex_dividend_date": "not-a-date"}])
    assert aw._load_ex_div_by_symbol(screened) == {}


def test_load_ex_div_by_symbol_missing_results_key_returns_empty():
    assert aw._load_ex_div_by_symbol({}) == {}


def test_ex_div_warning_fires_when_inside_implied_window():
    ex_div_date = date.today() + timedelta(days=10)
    warning = aw._ex_div_warning("AAPL", proposed_dte=14, ex_div_by_symbol={"AAPL": ex_div_date})
    assert "ex-dividend on" in warning
    assert ex_div_date.isoformat() in warning


def test_ex_div_warning_empty_when_outside_implied_window():
    ex_div_date = date.today() + timedelta(days=30)
    warning = aw._ex_div_warning("AAPL", proposed_dte=14, ex_div_by_symbol={"AAPL": ex_div_date})
    assert warning == ""


def test_ex_div_warning_empty_when_no_known_ex_div_date():
    assert aw._ex_div_warning("AAPL", proposed_dte=14, ex_div_by_symbol={}) == ""


def test_ex_div_warning_boundary_exactly_at_implied_expiration():
    ex_div_date = date.today() + timedelta(days=14)
    warning = aw._ex_div_warning("AAPL", proposed_dte=14, ex_div_by_symbol={"AAPL": ex_div_date})
    assert warning != ""  # inclusive of the implied expiration date itself


def test_ex_div_warning_fires_when_ex_div_is_today():
    # Unlike earnings (strictly after today), the approved ex-div window is
    # today <= ex-div <= implied expiration -- today itself DOES flag.
    warning = aw._ex_div_warning("AAPL", proposed_dte=14, ex_div_by_symbol={"AAPL": date.today()})
    assert warning != ""


def test_ex_div_warning_empty_when_date_is_in_the_past():
    ex_div_date = date.today() - timedelta(days=1)
    warning = aw._ex_div_warning("AAPL", proposed_dte=14, ex_div_by_symbol={"AAPL": ex_div_date})
    assert warning == ""


# ── _diff() wires the ex-div warning into ADD/CHANGE, not REMOVE ────────────

def test_diff_add_shows_ex_div_warning(capsys):
    ex_div_date = date.today() + timedelta(days=5)
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": 14, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed, ex_div_by_symbol={"AAPL": ex_div_date})
    out = capsys.readouterr().out
    assert "ex-dividend on" in out
    assert ex_div_date.isoformat() in out


def test_diff_add_no_ex_div_warning_when_outside_window(capsys):
    ex_div_date = date.today() + timedelta(days=90)
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": 14, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed, ex_div_by_symbol={"AAPL": ex_div_date})
    out = capsys.readouterr().out
    assert "ex-dividend on" not in out


def test_diff_change_uses_proposed_dte_for_ex_div_check(capsys):
    # Active's old DTE (60) would put the ex-div date outside the window;
    # the PROPOSED dte (14) is what must actually be used for the check.
    ex_div_date = date.today() + timedelta(days=10)
    active = _watchlist({"AAPL": {"dte": 60, "avg_annualized_yield": 0.10}})
    proposed = _watchlist({"AAPL": {"dte": 14, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed, ex_div_by_symbol={"AAPL": ex_div_date})
    out = capsys.readouterr().out
    assert "ex-dividend on" in out


def test_diff_remove_never_shows_ex_div_warning(capsys):
    ex_div_date = date.today() + timedelta(days=5)
    active = _watchlist({"AAPL": {"dte": 14, "avg_annualized_yield": 0.18}})
    proposed = _watchlist({})
    aw._diff(active, proposed, ex_div_by_symbol={"AAPL": ex_div_date})
    out = capsys.readouterr().out
    assert "REMOVE:" in out
    assert "ex-dividend on" not in out


def test_diff_without_ex_div_by_symbol_arg_is_backward_compatible(capsys):
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": 14, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed)  # no ex_div_by_symbol at all
    out = capsys.readouterr().out
    assert "+ AAPL" in out
    assert "ex-dividend on" not in out


def test_diff_shows_both_earnings_and_ex_div_warning_when_both_present(capsys):
    earnings_date = date.today() + timedelta(days=5)
    ex_div_date = date.today() + timedelta(days=8)
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": 14, "avg_annualized_yield": 0.18}})
    aw._diff(
        active, proposed,
        earnings_by_symbol={"AAPL": earnings_date},
        ex_div_by_symbol={"AAPL": ex_div_date},
    )
    out = capsys.readouterr().out
    assert "earnings on" in out
    assert "ex-dividend on" in out


# ── _load_spot_price_by_symbol / _capital_cap_summary ───────────────────────

def test_load_spot_price_by_symbol_parses_known_prices():
    screened = _screened([
        {"symbol": "AAPL", "spot_price": 348.34},
        {"symbol": "MSFT", "spot_price": None},
        {"symbol": "KO"},  # no spot_price key at all
    ])
    assert aw._load_spot_price_by_symbol(screened) == {"AAPL": 348.34}


def test_load_spot_price_by_symbol_missing_results_key_returns_empty():
    assert aw._load_spot_price_by_symbol({}) == {}


def test_capital_cap_summary_shows_unknown_when_no_spot_price_on_record():
    summary = aw._capital_cap_summary("AAPL", {})
    assert "unknown" in summary
    assert "OVER CAP" not in summary


def test_capital_cap_summary_shows_lot_cost_without_warning_when_under_cap():
    price = (MAX_CAPITAL_PER_SYMBOL / SHARES_PER_LOT) - 1
    summary = aw._capital_cap_summary("CHEAP", {"CHEAP": price})
    assert "lot cost" in summary
    assert f"{MAX_CAPITAL_PER_SYMBOL:,.2f}" in summary
    assert "OVER CAP" not in summary


def test_capital_cap_summary_warns_loudly_when_over_cap():
    # AAPL's real price at the time this was found, per the go/no-go report.
    summary = aw._capital_cap_summary("AAPL", {"AAPL": 348.34})
    assert "OVER CAP" in summary
    assert "34,834.00" in summary  # 100 x $348.34


def test_diff_add_shows_capital_cap_summary_for_every_entry(capsys):
    price = (MAX_CAPITAL_PER_SYMBOL / SHARES_PER_LOT) - 1
    active = _watchlist({})
    proposed = _watchlist({"CHEAP": {"dte": 30, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed, spot_price_by_symbol={"CHEAP": price})
    out = capsys.readouterr().out
    assert "lot cost" in out
    assert "OVER CAP" not in out


def test_diff_add_warns_loudly_when_proposed_symbol_is_over_cap(capsys):
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": 30, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed, spot_price_by_symbol={"AAPL": 348.34})
    out = capsys.readouterr().out
    assert "OVER CAP" in out


def test_diff_change_also_shows_capital_cap_summary(capsys):
    price = (MAX_CAPITAL_PER_SYMBOL / SHARES_PER_LOT) - 1
    active = _watchlist({"CHEAP": {"dte": 30, "avg_annualized_yield": 0.10}})
    proposed = _watchlist({"CHEAP": {"dte": 14, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed, spot_price_by_symbol={"CHEAP": price})
    out = capsys.readouterr().out
    assert "lot cost" in out


def test_diff_remove_never_shows_capital_cap_summary(capsys):
    active = _watchlist({"AAPL": {"dte": 30, "avg_annualized_yield": 0.18}})
    proposed = _watchlist({})
    aw._diff(active, proposed, spot_price_by_symbol={"AAPL": 348.34})
    out = capsys.readouterr().out
    assert "REMOVE:" in out
    assert "lot cost" not in out


def test_diff_without_spot_price_by_symbol_arg_is_backward_compatible(capsys):
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": 30, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed)  # no spot_price_by_symbol at all
    out = capsys.readouterr().out
    assert "+ AAPL" in out
    assert "unknown" in out  # falls back to "no spot price on record"


# ── _roll_threshold_warning ──────────────────────────────────────────────────

def test_roll_threshold_warning_fires_at_exactly_roll_min_dte():
    warning = aw._roll_threshold_warning(ROLL_MIN_DTE)
    assert "within the roll threshold" in warning
    assert f"DTE={ROLL_MIN_DTE}" in warning


def test_roll_threshold_warning_fires_below_roll_min_dte():
    warning = aw._roll_threshold_warning(ROLL_MIN_DTE - 1)
    assert "within the roll threshold" in warning


def test_roll_threshold_warning_empty_comfortably_above_roll_min_dte():
    assert aw._roll_threshold_warning(ROLL_MIN_DTE + 10) == ""


def test_diff_add_shows_roll_threshold_warning_for_a_dte7_pick(capsys):
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": ROLL_MIN_DTE, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "within the roll threshold" in out


def test_diff_add_no_roll_threshold_warning_for_a_safely_long_dte(capsys):
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": ROLL_MIN_DTE + 20, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "within the roll threshold" not in out


def test_diff_change_also_shows_roll_threshold_warning(capsys):
    active = _watchlist({"AAPL": {"dte": 30, "avg_annualized_yield": 0.10}})
    proposed = _watchlist({"AAPL": {"dte": ROLL_MIN_DTE, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "within the roll threshold" in out


# ── _closer_will_use_summary (Phase 3, Task 8) ───────────────────────────────

def test_closer_will_use_summary_shows_watchlist_delta_and_dte_window():
    summary = aw._closer_will_use_summary({"dte": 14, "target_delta": 0.22})
    assert "Closer will use" in summary
    assert "[9, 19]" in summary  # 14 -/+ CLOSER_DTE_TOLERANCE_DAYS (5)
    assert "0.22 (source=watchlist)" in summary
    assert "Closer will fall back" not in summary
    assert "intent does not affect live behavior" in summary


def test_closer_will_use_summary_warns_when_target_delta_missing():
    summary = aw._closer_will_use_summary({"dte": 30})
    assert f"{TARGET_DELTA:.2f} (source=global -- watchlist value missing)" in summary
    assert "Closer will fall back" in summary


def test_closer_will_use_summary_warns_when_target_delta_out_of_bounds():
    # Uses config.validate_live_delta directly -- same bounds the live
    # trader checks, no duplicated threshold here.
    bad_delta = MAX_LIVE_DELTA + 0.10
    summary = aw._closer_will_use_summary({"dte": 30, "target_delta": bad_delta})
    assert "out of bounds" in summary
    assert f"{TARGET_DELTA:.2f} (source=global" in summary
    assert "Closer will fall back" in summary


def test_closer_will_use_summary_warns_when_target_delta_non_numeric():
    summary = aw._closer_will_use_summary({"dte": 30, "target_delta": "not-a-number"})
    assert "invalid" in summary
    assert "Closer will fall back" in summary


def test_closer_will_use_summary_unknown_dte_window_when_dte_missing():
    summary = aw._closer_will_use_summary({"target_delta": MIN_LIVE_DELTA})
    assert "unknown (no dte on this entry)" in summary


def test_diff_add_shows_closer_will_use_summary(capsys):
    active = _watchlist({})
    proposed = _watchlist({"AAPL": {"dte": 14, "target_delta": 0.22, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "Closer will use" in out
    assert "0.22 (source=watchlist)" in out
    assert "intent does not affect live behavior" in out


def test_diff_add_warns_in_output_when_proposed_delta_out_of_bounds(capsys):
    active = _watchlist({})
    bad_delta = MIN_LIVE_DELTA - 0.05
    proposed = _watchlist({"AAPL": {"dte": 14, "target_delta": bad_delta, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "out of bounds" in out
    assert "Closer will fall back" in out


def test_diff_change_also_shows_closer_will_use_summary(capsys):
    active = _watchlist({"AAPL": {"dte": 30, "target_delta": 0.25, "avg_annualized_yield": 0.10}})
    proposed = _watchlist({"AAPL": {"dte": 14, "target_delta": 0.35, "avg_annualized_yield": 0.18}})
    aw._diff(active, proposed)
    out = capsys.readouterr().out
    assert "Closer will use" in out
    assert "0.35 (source=watchlist)" in out
