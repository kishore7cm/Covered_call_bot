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
"""
import argparse
import json
import logging
import os
from datetime import date, datetime

import backtest
from calendar_data import get_past_earnings_dates
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from logger import get_option_trades, get_share_trades

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

PROPOSED_PATH = "pipeline_state/proposed_watchlist.json"
ACTIVE_PATH = "pipeline_state/active_watchlist.json"
OUT_PATH = "pipeline_state/analysis_report.json"

YIELD_DRIFT_THRESHOLD_PP = 10.0       # percentage points
ASSIGNMENT_DRIFT_THRESHOLD_PP = 10.0  # percentage points
MARKET_WIDE_DECLINE_THRESHOLD = -0.05  # SPY move over the same window, below which we call it "market-wide"


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


def _parse_ts(v) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))


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


def run_analysis(watchlist_path: str = ACTIVE_PATH) -> list[dict]:
    predicted_by_symbol = _load_watchlist_stats(watchlist_path)
    if not predicted_by_symbol:
        predicted_by_symbol = _load_watchlist_stats(PROPOSED_PATH)

    rows = get_option_trades()
    symbols = sorted(set(r["symbol"] for r in rows)) if rows else sorted(predicted_by_symbol.keys())

    return [analyze_symbol(symbol, predicted_by_symbol.get(symbol)) for symbol in symbols]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", default=ACTIVE_PATH, help="watchlist JSON to compare realized results against")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    results = run_analysis(args.watchlist)
    _print_report(results)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"generated_at": date.today().isoformat(), "results": results}, f, indent=2)
    print(f"\nWrote report to {args.out}")
