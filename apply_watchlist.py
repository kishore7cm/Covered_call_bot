"""
Human approval gate between research (screener.py + backtest.py) and the
live trader (main.py). The live trader never reads proposed_watchlist.json
directly — it reads active_watchlist.json, and this script is the only thing
that writes that file. Deliberately a separate, trivial script rather than a
flag on backtest.py, so "backtest produced new numbers" and "I'm approving
these for live trading" are always two distinct, explicit actions.

Task 2's backtest.py now attaches walk-forward in_sample_yield/
out_of_sample_yield/oos_gap_pp to every symbol's stats -- this script
surfaces those right next to each ADD/REMOVE/CHANGE line (not just a raw
DTE number) and flags a large in-sample/out-of-sample gap, so a symbol
whose backtest doesn't generalize is visible at the one point a human
actually reviews it, before it can affect live positions. Still changes
nothing without --confirm -- this is a display-only addition.

Also flags when OOS risk-adjusted return (premium + stock P&L, divided by
drawdown -- the thing selection actually optimizes for) degraded hard
relative to in-sample, separately from the yield gap above: yield ignores
drawdown/stock P&L entirely, so a combo can pass the yield-gap check while
its risk-adjusted performance quietly fell apart out of sample.

EARNINGS vs. PROPOSED DTE: Scout (screener.py) records each symbol's next
known earnings date in screened_symbols.json. For every ADD/CHANGE line,
this compares that date against the expiration IMPLIED by the PROPOSED
dte (today + dte days -- an approximation, same one main.py's own
DTE_TOLERANCE_DAYS window already makes) and warns if earnings falls
inside it. This is a SEPARATE, informational check from the live bot's
actual gate (options_data.pick_strike's REQUIRE_EXPIRY_BEFORE_EARNINGS,
which filters the REAL candidate contracts by their real expiration_date
at trade time) -- a clean pass here does not guarantee the live gate
sees no conflict either, and vice versa; see _earnings_warning's
docstring.

EX-DIVIDEND vs. PROPOSED DTE: same approach, mirrored for Scout's recorded
ex-dividend date (early-assignment risk on an ITM call near ex-div),
using the same informational-only window approved for options_data.py's
ex_div_before_expiration flag (today <= ex-div date <= implied
expiration, both ends inclusive). See _ex_div_warning's docstring.

Usage:
    python apply_watchlist.py                 # show the diff, change nothing
    python apply_watchlist.py --confirm       # show the diff, then apply it
"""
import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone

PROPOSED_PATH = "pipeline_state/proposed_watchlist.json"
ACTIVE_PATH = "pipeline_state/active_watchlist.json"
SCREENED_PATH = "pipeline_state/screened_symbols.json"

# Flag threshold matches analysis.py's own YIELD_DRIFT_THRESHOLD_PP (same
# "10 percentage points" convention for "this number drifted enough to look
# into"), kept local here rather than in config.py -- same existing pattern
# as this file's own PROPOSED_PATH/ACTIVE_PATH constants.
OOS_GAP_FLAG_THRESHOLD_PP = 10.0

# risk_adjusted_return is a dimensionless ratio, not a bounded percentage
# like yield -- a fixed absolute (percentage-point-style) threshold isn't
# meaningful for it the way OOS_GAP_FLAG_THRESHOLD_PP is for yield. Instead:
# flag when OOS fell below this FRACTION of in-sample (e.g. 0.5 = OOS cut
# to less than half of what in-sample showed), or went non-positive outright
# while in-sample was positive -- both scale-independent, and both are the
# "backtest doesn't generalize" story this is trying to catch.
RISK_ADJUSTED_DEGRADATION_THRESHOLD = 0.5


def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {"symbols": {}}
    with open(path) as f:
        return json.load(f)


def _load_earnings_by_symbol(screened: dict) -> dict[str, date]:
    """
    {symbol: earnings_date} from screener.py's screened_symbols.json --
    only symbols with a KNOWN earnings date (Scout's own fail-soft
    yfinance lookup already returns None/omits it when there isn't one).
    screened.get("results", []) degrades to {} for a missing file (via
    _load's own {"symbols": {}} default, which simply has no "results"
    key either) or a malformed/pre-Task-5 file, same as every other
    loader in this script.
    """
    earnings_by_symbol = {}
    for r in screened.get("results", []):
        earnings_date_str = r.get("earnings_date")
        if not earnings_date_str:
            continue
        try:
            earnings_by_symbol[r["symbol"]] = date.fromisoformat(earnings_date_str)
        except ValueError:
            continue
    return earnings_by_symbol


def _load_ex_div_by_symbol(screened: dict) -> dict[str, date]:
    """
    {symbol: ex_dividend_date} from screener.py's screened_symbols.json --
    mirrors _load_earnings_by_symbol above. Only symbols with a KNOWN
    ex-dividend date (screener._ex_dividend_date already returns None when
    yfinance has nothing to offer).
    """
    ex_div_by_symbol = {}
    for r in screened.get("results", []):
        ex_div_date_str = r.get("ex_dividend_date")
        if not ex_div_date_str:
            continue
        try:
            ex_div_by_symbol[r["symbol"]] = date.fromisoformat(ex_div_date_str)
        except ValueError:
            continue
    return ex_div_by_symbol


def _ex_div_warning(symbol: str, proposed_dte: int, ex_div_by_symbol: dict[str, date]) -> str:
    """
    Informational only -- mirrors _earnings_warning's approach, but uses the
    approved ex-dividend window (today <= ex-div date <= implied
    expiration), the same inclusive-both-ends window options_data.py's
    _build_candidate uses for ex_div_before_expiration. Never excludes a
    symbol or changes order behavior -- a flag only, same as Part A.
    """
    ex_div_date = ex_div_by_symbol.get(symbol)
    if ex_div_date is None:
        return ""

    today = date.today()
    implied_expiration = today + timedelta(days=proposed_dte)
    if today <= ex_div_date <= implied_expiration:
        return (
            f"!! ex-dividend on {ex_div_date.isoformat()} falls within the ~{proposed_dte}-day proposed "
            f"expiration window (today -> ~{implied_expiration.isoformat()}) -- early-assignment risk on "
            f"an ITM call; informational only, does not change strike selection or order behavior"
        )
    return ""


def _earnings_warning(symbol: str, proposed_dte: int, earnings_by_symbol: dict[str, date]) -> str:
    """
    Compares this symbol's known earnings date against the expiration
    IMPLIED by the proposed DTE (today + dte days). "" if there's no known
    earnings date for this symbol, or if it falls outside (today, implied
    expiration].

    NOT the same check as the live bot's actual gate: options_data.
    pick_strike() filters the REAL candidate contracts by their own real
    expiration_date against the real earnings date, at trade time --
    exact, not an approximation. This check only has today + dte (an
    estimate; real expirations land on whatever Friday the chain actually
    offers), and runs at research-review time, not trade time. A clean
    pass here is a useful heads-up, not a guarantee the live gate will
    also see no conflict -- and a warning here doesn't mean the live bot
    will necessarily be affected either, since its own gate independently
    reshapes or skips the cycle if needed.
    """
    earnings_date = earnings_by_symbol.get(symbol)
    if earnings_date is None:
        return ""

    today = date.today()
    implied_expiration = today + timedelta(days=proposed_dte)
    if today < earnings_date <= implied_expiration:
        return (
            f"!! earnings on {earnings_date.isoformat()} falls within the ~{proposed_dte}-day proposed "
            f"expiration window (today -> ~{implied_expiration.isoformat()}) -- IV rank may be inflated; "
            f"the live bot's own gate will independently reshape or skip this cycle's actual strike"
        )
    return ""


def _fmt_pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def _is_oos_summary(stats: dict) -> str:
    """
    One line: in-sample yield, out-of-sample yield, and the gap between
    them, flagged if it's large. stats is a plain dict loaded from JSON
    (not a CycleStats instance), and may predate Task 2 (no
    in_sample_yield/out_of_sample_yield/oos_gap_pp keys at all) -- .get()
    everywhere so an older watchlist file degrades to "n/a" instead of
    crashing the diff.
    """
    is_yield = stats.get("in_sample_yield")
    oos_yield = stats.get("out_of_sample_yield")
    gap_pp = stats.get("oos_gap_pp")

    summary = f"IS yield={_fmt_pct(is_yield)} OOS yield={_fmt_pct(oos_yield)}"
    if gap_pp is None:
        return summary

    summary += f" (gap={gap_pp:+.1f}pp)"
    if abs(gap_pp) > OOS_GAP_FLAG_THRESHOLD_PP:
        direction = "in-sample beat out-of-sample" if gap_pp > 0 else "out-of-sample beat in-sample"
        summary += f"  [!! LARGE GAP — {direction} by {abs(gap_pp):.1f}pp, backtest may not generalize]"
    return summary


def _risk_adjusted_summary(stats: dict) -> str:
    """
    Separate from _is_oos_summary's yield-gap check: flags when OOS
    risk-adjusted return (total_return_pct / drawdown -- what selection
    actually optimized for) degraded hard relative to in-sample, even if
    the yield gap above looked fine. "" (nothing printed) if either side
    is missing -- older watchlist entries, or a symbol whose OOS slice
    produced no cycles at all (oos_risk_adjusted_return stays None).
    """
    is_ra = stats.get("risk_adjusted_return")
    oos_ra = stats.get("oos_risk_adjusted_return")
    if is_ra is None or oos_ra is None:
        return ""

    summary = f"IS risk-adjusted={is_ra:.2f} OOS risk-adjusted={oos_ra:.2f}"
    if is_ra > 0 and oos_ra <= 0:
        summary += "  [!! OOS risk-adjusted return went non-positive — backtest may not generalize]"
    elif is_ra > 0 and oos_ra < is_ra * RISK_ADJUSTED_DEGRADATION_THRESHOLD:
        pct_of_is = oos_ra / is_ra * 100
        summary += (
            f"  [!! OOS risk-adjusted fell to {pct_of_is:.0f}% of in-sample "
            f"(below the {RISK_ADJUSTED_DEGRADATION_THRESHOLD * 100:.0f}% threshold) — "
            f"backtest may not generalize]"
        )
    return summary


def _score_used_summary(stats: dict) -> str:
    """
    Which objective actually picked this symbol's (DTE, delta) -- for
    exit-intent symbols this is turnover_score, NOT risk_adjusted_return,
    even though _risk_adjusted_summary above always shows the latter. A
    reviewer needs to know which number was the deciding criterion, not
    just see both and guess. "" for watchlist entries that predate Task 4.
    """
    score_used = stats.get("score_used")
    if score_used is None:
        return ""
    if score_used == "turnover_score":
        turnover = stats.get("turnover_score")
        turnover_str = f"{turnover:.4f}/day" if turnover is not None else "n/a"
        return f"score used: turnover_score ({turnover_str})"
    return f"score used: {score_used}"


def _intent_summary(stats: dict) -> str:
    """
    intent + whether it came from an explicit pipeline_state/intents.json
    entry or the default (nobody listed this symbol) -- the two cases must
    never be shown identically, per the project's requirement that a
    default is always visibly a default, not indistinguishable from a
    deliberate choice. "n/a" for watchlist entries that predate intent
    tracking entirely (no "intent" key at all).
    """
    intent = stats.get("intent")
    if intent is None:
        return "intent: n/a (predates intent tracking)"
    suffix = " (default — not listed in intents.json)" if stats.get("intent_is_default") else " (explicit)"
    return f"intent={intent}{suffix}"


def _fmt_pct_or_na(value, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _empirical_summary(stats: dict, label: str, prefix: str = "") -> str:
    """
    One line of backtest.py's empirical outcome stats (train or OOS,
    selected via prefix="" / "oos_") -- P(called away), P(small move),
    median move, and the effective sample size, right next to the yield
    columns per the project's requirement. "" if this watchlist entry
    predates empirical stats (older proposed_watchlist.json) or this
    particular window had no valid stats at all (e.g. an OOS slice with no
    windows). If p_called_away IS present but a sibling field is somehow
    missing, each sibling shows "n/a" for ITSELF, never a silently
    substituted 0.0% -- a missing probability is not the same thing as a
    measured zero, and the two must never look identical.
    """
    p_called_away = stats.get(f"{prefix}p_called_away")
    if p_called_away is None:
        return ""

    threshold = stats.get("move_threshold_pct_used")
    threshold_str = f"{threshold:.0%}" if threshold is not None else "?"
    window_count = stats.get(f"{prefix}empirical_window_count")
    effective_n = stats.get(f"{prefix}empirical_effective_n")
    return (
        f"{label}: P(called away)={_fmt_pct_or_na(p_called_away)} "
        f"P(expires below strike)={_fmt_pct_or_na(stats.get(f'{prefix}p_expires_below_strike'))} "
        f"P(|move|<{threshold_str})={_fmt_pct_or_na(stats.get(f'{prefix}p_small_move'))} "
        f"median move={_fmt_pct_or_na(stats.get(f'{prefix}median_move_pct'), signed=True)} "
        f"(n={window_count if window_count is not None else 'n/a'} windows, "
        f"effective_n={effective_n if effective_n is not None else 'n/a'})"
    )


def _print_empirical_summaries(stats: dict) -> None:
    train_line = _empirical_summary(stats, "empirical (train)")
    if train_line:
        print(f"      {train_line}")
    oos_line = _empirical_summary(stats, "empirical (OOS)", prefix="oos_")
    if oos_line:
        print(f"      {oos_line}")


def _diff(
    active: dict, proposed: dict,
    earnings_by_symbol: dict[str, date] | None = None,
    ex_div_by_symbol: dict[str, date] | None = None,
) -> None:
    earnings_by_symbol = earnings_by_symbol or {}
    ex_div_by_symbol = ex_div_by_symbol or {}
    active_symbols = active.get("symbols", {})
    proposed_symbols = proposed.get("symbols", {})

    added = sorted(set(proposed_symbols) - set(active_symbols))
    removed = sorted(set(active_symbols) - set(proposed_symbols))
    changed = sorted(
        s for s in set(active_symbols) & set(proposed_symbols)
        if active_symbols[s].get("dte") != proposed_symbols[s].get("dte")
        or active_symbols[s].get("intent") != proposed_symbols[s].get("intent")
    )

    print(f"Active watchlist generated: {active.get('generated_at', 'never')}")
    print(f"Proposed watchlist generated: {proposed.get('generated_at', 'unknown')}")
    print()

    if added:
        print("ADD:")
        for s in added:
            stats = proposed_symbols[s]
            print(f"  + {s}: DTE={stats['dte']} (avg annualized yield {stats['avg_annualized_yield'] * 100:.1f}%)")
            print(f"      {_intent_summary(stats)}")
            score_summary = _score_used_summary(stats)
            if score_summary:
                print(f"      {score_summary}")
            print(f"      {_is_oos_summary(stats)}")
            ra_summary = _risk_adjusted_summary(stats)
            if ra_summary:
                print(f"      {ra_summary}")
            _print_empirical_summaries(stats)
            earnings_warning = _earnings_warning(s, stats["dte"], earnings_by_symbol)
            if earnings_warning:
                print(f"      {earnings_warning}")
            ex_div_warning = _ex_div_warning(s, stats["dte"], ex_div_by_symbol)
            if ex_div_warning:
                print(f"      {ex_div_warning}")
    if removed:
        print("REMOVE:")
        for s in removed:
            print(f"  - {s}")
            print(f"      {_intent_summary(active_symbols[s])}")
            score_summary = _score_used_summary(active_symbols[s])
            if score_summary:
                print(f"      {score_summary}")
            print(f"      {_is_oos_summary(active_symbols[s])}")
            ra_summary = _risk_adjusted_summary(active_symbols[s])
            if ra_summary:
                print(f"      {ra_summary}")
            _print_empirical_summaries(active_symbols[s])
    if changed:
        print("CHANGE:")
        for s in changed:
            old, new = active_symbols[s], proposed_symbols[s]
            parts = []
            if old.get("dte") != new.get("dte"):
                parts.append(f"DTE {old.get('dte')} -> {new.get('dte')}")
            if old.get("intent") != new.get("intent"):
                parts.append(f"intent {old.get('intent', '?')} -> {new.get('intent', '?')}")
            print(f"  ~ {s}: {', '.join(parts)}")
            print(f"      {_intent_summary(new)}")
            score_summary = _score_used_summary(new)
            if score_summary:
                print(f"      {score_summary}")
            print(f"      {_is_oos_summary(new)}")
            ra_summary = _risk_adjusted_summary(new)
            if ra_summary:
                print(f"      {ra_summary}")
            _print_empirical_summaries(new)
            earnings_warning = _earnings_warning(s, new["dte"], earnings_by_symbol)
            if earnings_warning:
                print(f"      {earnings_warning}")
            ex_div_warning = _ex_div_warning(s, new["dte"], ex_div_by_symbol)
            if ex_div_warning:
                print(f"      {ex_div_warning}")
    if not (added or removed or changed):
        print("No differences — active watchlist already matches the proposed one.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="apply the proposed watchlist as active")
    parser.add_argument("--proposed", default=PROPOSED_PATH)
    parser.add_argument("--active", default=ACTIVE_PATH)
    parser.add_argument("--screened", default=SCREENED_PATH,
                         help="screener.py's output, used only for the earnings-vs-proposed-dte warning")
    args = parser.parse_args()

    if not os.path.exists(args.proposed):
        print(f"No proposed watchlist at {args.proposed} — run backtest.py --from-screener first.")
        raise SystemExit(1)

    active = _load(args.active)
    proposed = _load(args.proposed)
    screened = _load(args.screened)
    earnings_by_symbol = _load_earnings_by_symbol(screened)
    ex_div_by_symbol = _load_ex_div_by_symbol(screened)
    _diff(active, proposed, earnings_by_symbol, ex_div_by_symbol)

    if not args.confirm:
        print("\n(dry run — pass --confirm to apply this to the live trader's watchlist)")
    else:
        os.makedirs(os.path.dirname(args.active), exist_ok=True)
        # confirmed_at is a human-approval timestamp, stamped fresh HERE --
        # deliberately distinct from proposed's own generated_at (when the
        # Oracle produced these numbers, not when a human approved them).
        # main.py's watchlist-expiry check (Phase 3, Task 5) reads this to
        # decide whether its approval is still fresh enough to open NEW
        # positions on; re-running --confirm always resets it to now.
        payload = dict(proposed)
        payload["confirmed_at"] = datetime.now(timezone.utc).isoformat()
        with open(args.active, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nApplied — {args.active} updated (confirmed_at={payload['confirmed_at']}). "
              "main.py will pick this up on its next cycle.")
