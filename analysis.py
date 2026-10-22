"""
Stage 4 of the research pipeline (Auditor): after positions have cycled
through a few expirations, check (1) whether realized premium yield matched
what backtest.py predicted, and (2) full profit/loss — premium collected
AND what happened to the stock, not just the option leg. If the stock leg
lost money, flag data-backed contributing factors (tail event vs. normal
range, earnings coincidence, market-wide vs. stock-specific) so the same
mistake is visible next time a watchlist decision gets made. Never re-runs
screener.py/backtest.py itself and never touches live positions — it's a
report, not a decision-maker.

LIMITATIONS, read before trusting a number out of this:
  - option_trades only logs SELL_TO_OPEN, ROLL, BUY_TO_CLOSE, and (as of the
    pipeline work) ASSIGNED. A call that simply expires worthless and gets
    replaced next cycle has no distinct logged row — it just shows up as
    another SELL_TO_OPEN. So "assignment rate" here is ASSIGNED rows over
    total opened cycles, which undercounts the true cycle count slightly.
  - The contributing-factor flags are data-backed signals (tail event vs.
    this symbol's own history, earnings-date coincidence, SPY comparison
    over the same window) — NOT a definitive causal explanation. They tell
    you what's worth looking into, not "this is definitely why."
  - Stock P&L only covers one lot at a time (this bot never holds more than
    SHARES_PER_LOT per symbol), matched by nearest-prior BUY in share_trades.
  - TASK 4 (per-IVR-tier breakdown): nothing logs the IVR a real trade was
    opened under -- main.py stays unedited in this phase, per the project's
    scoping, so there's no column to read it from directly. Instead,
    _ivr_at_date() reconstructs it AS OF each trade's own open date from
    iv_logger.py's daily history (Task 1) -- which means a trade opened
    before iv_logger.py had 252 days of history for that symbol will show
    up as "insufficient history" here forever; this breakdown gets more
    complete the longer iv_logger.py has been running, not retroactively.
"""
import argparse
import json
import logging
import os
from datetime import date, datetime, timedelta

import backtest
import iv_logger
from calendar_data import get_past_earnings_dates
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, DAILY_IVR_MIN_HISTORY_POINTS
from iv_rank import iv_rank_from_daily_history
from logger import get_option_trades, get_share_trades

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

PROPOSED_PATH = "pipeline_state/proposed_watchlist.json"
ACTIVE_PATH = "pipeline_state/active_watchlist.json"
OUT_PATH = "pipeline_state/analysis_report.json"

YIELD_DRIFT_THRESHOLD_PP = 10.0       # percentage points
ASSIGNMENT_DRIFT_THRESHOLD_PP = 10.0  # percentage points
MARKET_WIDE_DECLINE_THRESHOLD = -0.05  # SPY move over the same window, below which we call it "market-wide"

# Task 4: per-IVR-tier breakdown. Plain quintiles -- this is a reporting
# bucket, unrelated to portfolio_backtest.py's IVR_DTE_TIERS (which encodes
# a DTE *decision*, a different concern entirely).
TIER_LABELS = ["0-20", "20-40", "40-60", "60-80", "80-100"]


def _get_spot_price(symbol: str) -> float | None:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest

    try:
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        trade = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(trade[symbol].price)
    except Exception as e:
        _log.error("_get_spot_price(%s) failed: %s", symbol, e)
        return None


def _load_watchlist_stats(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get("symbols", {})


def _load_predicted(watchlist_path: str) -> dict:
    """active_watchlist.json's stats, falling back to proposed_watchlist.json
    if the pipeline hasn't been approved yet -- same fallback run_analysis()
    already used inline; factored out so Task 4's new functions don't each
    repeat it."""
    predicted = _load_watchlist_stats(watchlist_path)
    if not predicted:
        predicted = _load_watchlist_stats(PROPOSED_PATH)
    return predicted


def _parse_ts(v) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))


def _ivr_at_date(symbol: str, as_of: date) -> float | str | None:
    """
    IVR for `symbol` using only iv_logger.py's daily history up to and
    including `as_of` -- see the module docstring's Task 4 note on why this
    is reconstructed after the fact rather than read off the trade itself.
    Returns iv_rank_from_daily_history's own sentinel ("insufficient
    history") below DAILY_IVR_MIN_HISTORY_POINTS, None if there's enough
    history but no variation in it to rank against.
    """
    history = [h["atm_iv"] for h in iv_logger.get_history(symbol) if h["date"] <= as_of]
    return iv_rank_from_daily_history(history)


def _tier_label(ivr: float) -> str:
    if ivr >= 80:
        return "80-100"
    if ivr >= 60:
        return "60-80"
    if ivr >= 40:
        return "40-60"
    if ivr >= 20:
        return "20-40"
    return "0-20"


def _net_option_premium_dollars(rows: list[dict]) -> float:
    """SELL_TO_OPEN/ROLL premiums are credits, BUY_TO_CLOSE is a debit — all
    *100 for the contract multiplier, consistent with how main.py already
    treats cost_basis/market_value on option Positions."""
    total = 0.0
    for r in rows:
        if r["premium"] is None:
            continue
        qty = r["qty"] or 1
        amount = float(r["premium"]) * 100 * qty
        if r["action"] in ("SELL_TO_OPEN", "ROLL"):
            total += amount
        elif r["action"] == "BUY_TO_CLOSE":
            total -= amount
    return total


def _stock_pnl(symbol: str, rows: list[dict], share_trades: list[dict]) -> dict | None:
    """
    Match the most recent lot's entry (nearest-prior BUY in share_trades)
    against its exit: ASSIGNED row's strike if the lot was called away
    (realized), else the current live spot price if still held
    (unrealized). Returns None if there's no BUY on record to match against
    (shouldn't happen once trader.buy_shares's fill-price fix is live, but
    degrade gracefully for older/incomplete log history).
    """
    buys = [r for r in share_trades if r["side"] == "BUY" and r["price"] is not None]
    if not buys:
        return None

    assigned_rows = [r for r in rows if r["action"] == "ASSIGNED" and r["strike"] is not None]
    if assigned_rows:
        last_assigned = max(assigned_rows, key=lambda r: _parse_ts(r["occurred_at"]))
        exit_ts = _parse_ts(last_assigned["occurred_at"])
        prior_buys = [r for r in buys if _parse_ts(r["occurred_at"]) <= exit_ts]
        if not prior_buys:
            return None
        last_buy = max(prior_buys, key=lambda r: _parse_ts(r["occurred_at"]))
        entry_price = float(last_buy["price"])
        return {
            "status": "realized (assigned)",
            "amount": (float(last_assigned["strike"]) - entry_price) * 100,
            "entry_price": entry_price,
            "exit_price": float(last_assigned["strike"]),
            "entry_ts": _parse_ts(last_buy["occurred_at"]),
            "exit_ts": exit_ts,
        }

    last_buy = max(buys, key=lambda r: _parse_ts(r["occurred_at"]))
    entry_price = float(last_buy["price"])
    spot = _get_spot_price(symbol)
    if spot is None:
        return None
    return {
        "status": "unrealized (still held)",
        "amount": (spot - entry_price) * 100,
        "entry_price": entry_price,
        "exit_price": spot,
        "entry_ts": _parse_ts(last_buy["occurred_at"]),
        "exit_ts": datetime.now(),
    }


def _contributing_factors(symbol: str, entry_ts: datetime, exit_ts: datetime,
                           entry_price: float, exit_price: float) -> list[str]:
    """Data-backed flags only — see module docstring. Each check fails soft:
    a yfinance/data hiccup on one check must not block the others."""
    factors = []
    holding_calendar_days = max((exit_ts - entry_ts).days, 1)
    trading_days = max(1, round(holding_calendar_days * 5 / 7))  # same approximation backtest.py uses
    actual_return = (exit_price - entry_price) / entry_price

    try:
        closes = backtest._price_history(symbol, lookback="10y")
        window_returns = closes.pct_change(trading_days).dropna()
        if not window_returns.empty:
            worst_historical = float(window_returns.min())
            if actual_return < worst_historical:
                factors.append(
                    f"tail event: this lot's {actual_return * 100:.1f}% move over ~{holding_calendar_days}d is "
                    f"worse than any {trading_days}-trading-day move in {symbol}'s available price history "
                    f"({worst_historical * 100:.1f}%)"
                )
    except Exception as e:
        _log.warning("tail-event check failed for %s: %s", symbol, e)

    try:
        earnings = get_past_earnings_dates(symbol, entry_ts.date(), exit_ts.date())
        if earnings:
            factors.append(
                f"earnings date(s) fell inside the holding period: {', '.join(d.isoformat() for d in earnings)}"
            )
    except Exception as e:
        _log.warning("earnings-coincidence check failed for %s: %s", symbol, e)

    try:
        spy_closes = backtest._price_history("SPY", lookback="10y")
        spy_window = spy_closes.loc[str(entry_ts.date()):str(exit_ts.date())]
        if len(spy_window) >= 2:
            spy_return = float(spy_window.iloc[-1] / spy_window.iloc[0] - 1)
            if spy_return <= MARKET_WIDE_DECLINE_THRESHOLD:
                factors.append(f"market-wide decline: SPY also moved {spy_return * 100:.1f}% over the same window")
            else:
                factors.append(f"stock-specific decline: SPY moved only {spy_return * 100:.1f}% over the same window")
    except Exception as e:
        _log.warning("SPY comparison failed for %s: %s", symbol, e)

    return factors


def analyze_symbol(symbol: str, predicted: dict | None) -> dict:
    rows = get_option_trades(symbol)
    share_trades = get_share_trades(symbol)
    opens = [r for r in rows if r["action"] in ("SELL_TO_OPEN", "ROLL") and r["premium"] and r["underlying_price"]]
    assigned_count = sum(1 for r in rows if r["action"] == "ASSIGNED")

    if not opens:
        return {"symbol": symbol, "status": "no trade history yet"}

    premium_pcts = [float(r["premium"]) / float(r["underlying_price"]) for r in opens]
    total_cycles = len(opens)
    assignment_rate = assigned_count / total_cycles if total_cycles else None

    realized_annualized_yield = None
    if len(opens) >= 2:
        timestamps = sorted(_parse_ts(r["occurred_at"]) for r in opens)
        span_days = (timestamps[-1] - timestamps[0]).days
        if span_days > 0:
            avg_cycle_days = span_days / (len(timestamps) - 1)
            realized_annualized_yield = (sum(premium_pcts) / len(premium_pcts)) * (365.0 / avg_cycle_days)

    net_premium_dollars = _net_option_premium_dollars(rows)
    stock_pnl_raw = _stock_pnl(symbol, rows, share_trades)
    total_pnl = net_premium_dollars + stock_pnl_raw["amount"] if stock_pnl_raw else None

    loss_factors = []
    if stock_pnl_raw and stock_pnl_raw["amount"] < 0:
        loss_factors = _contributing_factors(
            symbol, stock_pnl_raw["entry_ts"], stock_pnl_raw["exit_ts"],
            stock_pnl_raw["entry_price"], stock_pnl_raw["exit_price"],
        )

    stock_pnl = None
    if stock_pnl_raw:
        stock_pnl = {
            **stock_pnl_raw,
            "entry_ts": stock_pnl_raw["entry_ts"].isoformat(),
            "exit_ts": stock_pnl_raw["exit_ts"].isoformat(),
        }

    result = {
        "symbol": symbol,
        "status": "ok",
        "total_cycles": total_cycles,
        "assigned_count": assigned_count,
        "assignment_rate": assignment_rate,
        "realized_annualized_yield": realized_annualized_yield,
        "net_premium_dollars": net_premium_dollars,
        "stock_pnl": stock_pnl,
        "total_pnl": total_pnl,
        "loss": total_pnl is not None and total_pnl < 0,
        "flags": [],
        "loss_factors": loss_factors,
    }

    if predicted:
        predicted_yield = predicted.get("avg_annualized_yield")
        predicted_assignment = predicted.get("assignment_rate")

        if realized_annualized_yield is not None and predicted_yield is not None:
            drift_pp = abs(realized_annualized_yield - predicted_yield) * 100
            if drift_pp > YIELD_DRIFT_THRESHOLD_PP:
                result["flags"].append(
                    f"realized annualized yield ({realized_annualized_yield * 100:.1f}%) drifted "
                    f"{drift_pp:.1f}pp from backtest prediction ({predicted_yield * 100:.1f}%)"
                )

        if assignment_rate is not None and predicted_assignment is not None:
            drift_pp = abs(assignment_rate - predicted_assignment) * 100
            if drift_pp > ASSIGNMENT_DRIFT_THRESHOLD_PP:
                result["flags"].append(
                    f"realized assignment rate ({assignment_rate * 100:.1f}%) drifted "
                    f"{drift_pp:.1f}pp from backtest prediction ({predicted_assignment * 100:.1f}%)"
                )

    return result


# ── Wash-sale risk reporting (Phase 3, Task 6) ───────────────────────────────
# REPORT ONLY -- per this project's hard rule, the Auditor never acts on
# this, retriggers anything, or alters a trade. It exists purely so a loss
# sale that might trip the IRS wash-sale rule is visible in the same place
# everything else about that symbol's performance is reviewed.
WASH_SALE_WINDOW_DAYS = 30


def report_wash_sale_risk(symbols: list[str] | None = None, rows: list[dict] | None = None) -> list[dict]:
    """
    For every realized SELL in share_trades (currently only produced by
    trader.sell_shares, via Task 6's recovery-exit) that was a loss
    relative to its matched entry BUY, flags whether any BUY of the same
    symbol falls within WASH_SALE_WINDOW_DAYS before or after the sale --
    the IRS wash-sale window. symbols=None scans every symbol with any
    share_trades history. rows=None reads live via get_share_trades();
    pass rows directly in tests to avoid needing a real database, same
    convention as _ivr_tier_breakdown. Returns [] if there's nothing to
    flag; never raises on a single symbol's bad data (skips it, same
    fail-soft contract as the rest of this module).
    """
    all_trades = get_share_trades() if rows is None else rows
    if symbols is None:
        symbols = sorted({r["symbol"] for r in all_trades})

    flags = []
    for symbol in symbols:
        try:
            symbol_rows = [r for r in all_trades if r["symbol"] == symbol]
            buys = sorted(
                (r for r in symbol_rows if r["side"] == "BUY" and r["price"] is not None),
                key=lambda r: _parse_ts(r["occurred_at"]),
            )
            sells = sorted(
                (r for r in symbol_rows if r["side"] == "SELL" and r["price"] is not None),
                key=lambda r: _parse_ts(r["occurred_at"]),
            )

            for sell in sells:
                sell_ts = _parse_ts(sell["occurred_at"])
                prior_buys = [b for b in buys if _parse_ts(b["occurred_at"]) <= sell_ts]
                if not prior_buys:
                    continue
                entry = max(prior_buys, key=lambda b: _parse_ts(b["occurred_at"]))
                entry_price = float(entry["price"])
                sell_price = float(sell["price"])
                if sell_price >= entry_price:
                    continue  # not a loss sale

                window_start = sell_ts - timedelta(days=WASH_SALE_WINDOW_DAYS)
                window_end = sell_ts + timedelta(days=WASH_SALE_WINDOW_DAYS)
                conflicting_buys = [
                    b for b in buys
                    if b is not entry and window_start <= _parse_ts(b["occurred_at"]) <= window_end
                ]
                if conflicting_buys:
                    flags.append({
                        "symbol": symbol,
                        "sale_date": sell_ts.date().isoformat(),
                        "loss_amount": (sell_price - entry_price) * 100,
                        "conflicting_buy_dates": sorted(
                            _parse_ts(b["occurred_at"]).date().isoformat() for b in conflicting_buys
                        ),
                    })
        except Exception as e:
            _log.warning("report_wash_sale_risk failed for %s: %s", symbol, e)

    return flags


def _print_wash_sale_report(flags: list[dict]) -> None:
    print("\n--- Wash-sale risk (report only -- never auto-acted on) ---")
    if not flags:
        print("  No loss sales with a same-symbol buy inside the 30-day window.")
        return
    for f in flags:
        print(
            f"  !! {f['symbol']}: loss sale on {f['sale_date']} (${f['loss_amount']:.2f}) -- "
            f"buy(s) within {WASH_SALE_WINDOW_DAYS} days: {', '.join(f['conflicting_buy_dates'])}"
        )


def run_analysis(watchlist_path: str = ACTIVE_PATH) -> list[dict]:
    predicted_by_symbol = _load_predicted(watchlist_path)

    rows = get_option_trades()
    symbols = sorted(set(r["symbol"] for r in rows)) if rows else sorted(predicted_by_symbol.keys())

    return [analyze_symbol(symbol, predicted_by_symbol.get(symbol)) for symbol in symbols]


def _ivr_tier_breakdown(rows: list[dict], predicted_by_symbol: dict) -> list[dict]:
    """
    Groups every real SELL_TO_OPEN/ROLL cycle by the IVR (Task 1's daily
    history, as of that cycle's own open date) it was opened under, and
    reports realized vs. predicted annualized yield + assignment rate per
    tier -- the same predicted-vs-realized question analyze_symbol already
    asks per symbol, asked again across the IVR axis instead. Per-cycle
    annualized yield uses that cycle's OWN logged expiration (more precise
    than analyze_symbol's symbol-level avg-cycle-span approximation, and
    available here for free since `expiration` is already a logged column).
    Assignment matching is by occ_symbol, not a global rate, so each ASSIGNED
    row counts against the specific tier its OPENING cycle fell into.
    """
    opens = [
        r for r in rows
        if r["action"] in ("SELL_TO_OPEN", "ROLL") and r["premium"] and r["underlying_price"] and r["expiration"]
    ]
    assigned_occ_symbols = {r["occ_symbol"] for r in rows if r["action"] == "ASSIGNED"}

    tier_cycles: dict[str, list[dict]] = {label: [] for label in TIER_LABELS}
    insufficient_history_count = 0

    for r in opens:
        opened_at = _parse_ts(r["occurred_at"]).date()
        ivr = _ivr_at_date(r["symbol"], opened_at)
        if not isinstance(ivr, float):
            insufficient_history_count += 1
            continue
        tier_cycles[_tier_label(ivr)].append(r)

    breakdown = []
    for label in TIER_LABELS:
        cycles = tier_cycles[label]
        if not cycles:
            continue

        annualized_yields = []
        for c in cycles:
            dte = (c["expiration"] - _parse_ts(c["occurred_at"]).date()).days
            if dte <= 0:
                continue
            annualized_yields.append((float(c["premium"]) / float(c["underlying_price"])) * (365.0 / dte))
        if not annualized_yields:
            continue

        predicted_yields = [
            predicted_by_symbol[c["symbol"]]["avg_annualized_yield"]
            for c in cycles
            if c["symbol"] in predicted_by_symbol and predicted_by_symbol[c["symbol"]].get("avg_annualized_yield") is not None
        ]
        assigned_count = sum(1 for c in cycles if c["occ_symbol"] in assigned_occ_symbols)

        breakdown.append({
            "tier": label,
            "num_cycles": len(cycles),
            "realized_avg_yield": sum(annualized_yields) / len(annualized_yields),
            "predicted_avg_yield": sum(predicted_yields) / len(predicted_yields) if predicted_yields else None,
            "assignment_rate": assigned_count / len(cycles),
        })

    if insufficient_history_count:
        breakdown.append({"tier": "insufficient history", "num_cycles": insufficient_history_count})

    return breakdown


def run_ivr_tier_analysis(watchlist_path: str = ACTIVE_PATH) -> list[dict]:
    predicted_by_symbol = _load_predicted(watchlist_path)
    rows = get_option_trades()
    return _ivr_tier_breakdown(rows, predicted_by_symbol)


def _overall_bias_pp(results: list[dict], predicted_by_symbol: dict) -> float | None:
    """
    Single number: mean(realized - predicted annualized yield) across every
    symbol with both, in percentage points. Positive = live trading is
    running ahead of what Oracle's backtest predicted on average; negative
    = behind. None if no symbol has both numbers yet.
    """
    diffs = []
    for r in results:
        realized = r.get("realized_annualized_yield")
        predicted = predicted_by_symbol.get(r["symbol"], {}).get("avg_annualized_yield")
        if realized is not None and predicted is not None:
            diffs.append(realized - predicted)

    if not diffs:
        return None
    return (sum(diffs) / len(diffs)) * 100


def _print_report(results: list[dict]) -> None:
    if not results:
        print("No trade history yet — nothing to analyze.")
        return

    for r in results:
        if r["status"] == "no trade history yet":
            print(f"{r['symbol']}: no trade history yet")
            continue

        yield_str = f"{r['realized_annualized_yield'] * 100:.1f}%" if r["realized_annualized_yield"] is not None else "n/a"
        assign_str = f"{r['assignment_rate'] * 100:.1f}%" if r["assignment_rate"] is not None else "n/a"
        print(f"{r['symbol']}: {r['total_cycles']} cycles, realized yield {yield_str}, assignment rate {assign_str}")

        stock_pnl = r.get("stock_pnl")
        if stock_pnl:
            print(f"  premium collected: ${r['net_premium_dollars']:.2f}  |  "
                  f"stock P&L ({stock_pnl['status']}): ${stock_pnl['amount']:.2f}  |  "
                  f"total P&L: ${r['total_pnl']:.2f}" + ("  *** LOSS ***" if r["loss"] else ""))
        else:
            print(f"  premium collected: ${r['net_premium_dollars']:.2f}  |  stock P&L: n/a (no BUY price on record)")

        for flag in r["flags"]:
            print(f"  ! prediction drift: {flag}")
        for factor in r["loss_factors"]:
            print(f"  ! loss factor: {factor}")

    flagged = [r["symbol"] for r in results if r.get("flags")]
    if flagged:
        print(f"\nConsider re-running screener.py/backtest.py for: {', '.join(flagged)}")

    lost_money = [r["symbol"] for r in results if r.get("loss")]
    if lost_money:
        print(f"Lost money overall on: {', '.join(lost_money)} — see loss factors above before re-adding/keeping these.")


def _print_tier_and_bias(tier_breakdown: list[dict], bias_pp: float | None) -> None:
    print("\n--- IVR-tier breakdown (realized vs. predicted, by IVR at cycle open) ---")
    numeric_rows = [row for row in tier_breakdown if row["tier"] != "insufficient history"]
    if not numeric_rows:
        print("  No cycles with a resolvable IVR yet.")
    for row in numeric_rows:
        predicted_str = (
            f"{row['predicted_avg_yield'] * 100:.1f}%" if row["predicted_avg_yield"] is not None else "n/a"
        )
        print(
            f"  IVR {row['tier']:>7}: {row['num_cycles']:>3} cycle(s) | "
            f"realized={row['realized_avg_yield'] * 100:>6.1f}%  predicted={predicted_str:>6}  "
            f"assignment_rate={row['assignment_rate'] * 100:.1f}%"
        )

    insufficient = next((row for row in tier_breakdown if row["tier"] == "insufficient history"), None)
    if insufficient:
        print(
            f"  insufficient history: {insufficient['num_cycles']} cycle(s) skipped — iv_logger.py hasn't "
            f"reached {DAILY_IVR_MIN_HISTORY_POINTS} days for the symbol(s) involved as of their open date"
        )

    print("\n--- Overall bias (mean realized - predicted annualized yield, across symbols) ---")
    if bias_pp is None:
        print("  n/a — no symbol has both a realized and a predicted yield yet.")
    else:
        direction = "OUTPERFORMING" if bias_pp > 0 else "UNDERPERFORMING"
        print(f"  {bias_pp:+.1f}pp — live trading is, on average, {direction} Oracle's backtest predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", default=ACTIVE_PATH, help="watchlist JSON to compare realized results against")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    predicted_by_symbol = _load_predicted(args.watchlist)
    results = run_analysis(args.watchlist)
    _print_report(results)

    tier_breakdown = run_ivr_tier_analysis(args.watchlist)
    bias_pp = _overall_bias_pp(results, predicted_by_symbol)
    _print_tier_and_bias(tier_breakdown, bias_pp)

    wash_sale_flags = report_wash_sale_risk()
    _print_wash_sale_report(wash_sale_flags)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "generated_at": date.today().isoformat(),
            "results": results,
            "ivr_tier_breakdown": tier_breakdown,
            "overall_bias_pp": bias_pp,
            "wash_sale_flags": wash_sale_flags,
        }, f, indent=2)
    print(f"\nWrote report to {args.out}")
