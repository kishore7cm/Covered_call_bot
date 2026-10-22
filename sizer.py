"""
Sizer — position sizing against Scout's screened universe, before Oracle
ever runs backtest.py. Turns "which symbols are liquid enough to trade"
(screened_symbols.json) into "how many 100-share lots of each, within a
fixed budget" — a purely mechanical price/budget/concentration decision,
with zero knowledge of yield, risk-adjusted return, or any Oracle output.
Deliberately upstream of Oracle: Foreman's research command runs this
right after Scout, before backtest.py, since sizing doesn't need (and
shouldn't wait for) a backtest to know whether a symbol is even
affordable or diversifiable at all.

ALLOCATION RULE: round-robin, cheapest lot cost first, one lot per symbol
per round until either the investable budget runs out or every symbol has
hit its own MAX_WEIGHT_PER_NAME cap. Sizer has no yield/quality ranking to
allocate by this early in the pipeline (that's Oracle's job, and it hasn't
run yet) — "fair, deterministic, maximize breadth per dollar" is the only
defensible rule available here.

Never writes active_watchlist.json, never touches a live position — this
is a planning report, not a decision-maker.

Usage:
    python sizer.py                                 # screened_symbols.json, config defaults
    python sizer.py --budget 30000 --min-names 8     # override config defaults
"""
import argparse
import json
import os
from datetime import datetime

from config import (
    SHARES_PER_LOT, SIZER_BUDGET, SIZER_CASH_RESERVE_PCT,
    SIZER_MAX_PRICE_PER_SHARE, SIZER_MAX_WEIGHT_PER_NAME, SIZER_MIN_NAMES,
)

DEFAULT_SCREENED_PATH = "pipeline_state/screened_symbols.json"
DEFAULT_OUT_PATH = "pipeline_state/sizing_plan.json"


def _load_passed_symbols(screened_path: str) -> tuple[list[dict], str | None]:
    """
    Returns (Scout's passed results as plain dicts, screened_symbols.json's
    own generated_at). Plain dicts rather than screener.ScreenResult so
    sizer.py doesn't need to import screener.py just for its dataclass
    shape — it only ever reads the JSON that script already wrote.
    """
    with open(screened_path) as f:
        payload = json.load(f)
    passed = [r for r in payload.get("results", []) if r.get("passed")]
    return passed, payload.get("generated_at")


def compute_sizing_plan(
    screened_path: str = DEFAULT_SCREENED_PATH,
    budget: float = SIZER_BUDGET,
    max_price_per_share: float = SIZER_MAX_PRICE_PER_SHARE,
    min_names: int = SIZER_MIN_NAMES,
    max_weight_per_name: float = SIZER_MAX_WEIGHT_PER_NAME,
    cash_reserve_pct: float = SIZER_CASH_RESERVE_PCT,
) -> dict:
    passed, spot_prices_as_of = _load_passed_symbols(screened_path)

    investable_budget = budget * (1.0 - cash_reserve_pct)
    # MAX_WEIGHT_PER_NAME is a fraction of the WHOLE budget (including the
    # cash reserve), not just the investable portion — a concentration cap
    # is about exposure relative to total capital, not just what's deployed.
    max_dollars_per_name = budget * max_weight_per_name

    feasible: list[dict] = []
    dropped: dict[str, dict] = {}

    for r in passed:
        symbol = r["symbol"]
        spot_price = r.get("spot_price")

        if spot_price is None:
            dropped[symbol] = {"reason": "no spot price on record — re-run screener.py to capture one"}
            continue

        if spot_price > max_price_per_share:
            dropped[symbol] = {
                "reason": f"price ${spot_price:.2f} exceeds MAX_PRICE_PER_SHARE cap of ${max_price_per_share:.2f}",
                "spot_price": spot_price,
            }
            continue

        lot_cost = SHARES_PER_LOT * spot_price
        if lot_cost > max_dollars_per_name:
            dropped[symbol] = {
                "reason": (
                    f"one lot (${lot_cost:,.2f}) already exceeds the MAX_WEIGHT_PER_NAME cap of "
                    f"${max_dollars_per_name:,.2f} ({max_weight_per_name:.0%} of budget)"
                ),
                "spot_price": spot_price, "lot_cost": lot_cost,
            }
            continue

        feasible.append({
            "symbol": symbol, "spot_price": spot_price, "lot_cost": lot_cost,
            "max_lots_by_weight": int(max_dollars_per_name // lot_cost),
        })

    feasible.sort(key=lambda f: f["lot_cost"])  # cheapest-first round-robin
    lots = {f["symbol"]: 0 for f in feasible}
    remaining = investable_budget
    progress = True
    while progress and remaining > 1e-9:
        progress = False
        for f in feasible:
            if lots[f["symbol"]] >= f["max_lots_by_weight"]:
                continue
            if f["lot_cost"] > remaining:
                continue
            lots[f["symbol"]] += 1
            remaining -= f["lot_cost"]
            progress = True

    symbols_out: dict[str, dict] = {}
    capital_used = 0.0
    for f in feasible:
        n = lots[f["symbol"]]
        capital = n * f["lot_cost"]
        capital_used += capital
        symbols_out[f["symbol"]] = {
            "spot_price": f["spot_price"], "lot_cost": f["lot_cost"], "lots": n,
            "capital": capital, "weight": (capital / budget) if budget else None,
            "max_lots_by_weight": f["max_lots_by_weight"],
        }
        if n == 0:
            # Passed the hard price/weight filters, but the cheapest-first
            # round-robin ran out of budget before this symbol's turn --
            # still "feasible" on its own, just unfunded in THIS plan.
            dropped[f["symbol"]] = {
                "reason": "feasible but budget was exhausted before this symbol's turn in the "
                          "cheapest-first round-robin — raise BUDGET or drop a funded name to free room",
                "spot_price": f["spot_price"], "lot_cost": f["lot_cost"],
            }

    names_funded = sum(1 for n in lots.values() if n > 0)
    cash_left = budget - capital_used

    suggestions: list[str] = []
    if names_funded < min_names:
        # Mode A: hard filters (price cap / weight cap) kept candidates out
        # of `feasible` entirely -- concrete per-symbol fixes.
        for symbol, d in sorted(dropped.items()):
            spot_price = d.get("spot_price")
            if spot_price is not None and spot_price > max_price_per_share:
                suggestions.append(f"raise MAX_PRICE_PER_SHARE to >= ${spot_price:.2f} to make {symbol} feasible")
            lot_cost = d.get("lot_cost")
            if lot_cost is not None and lot_cost > max_dollars_per_name:
                needed_weight = lot_cost / budget
                suggestions.append(
                    f"raise MAX_WEIGHT_PER_NAME to >= {needed_weight:.1%} to make {symbol} feasible "
                    f"(at the current ${budget:,.0f} budget)"
                )

        # Mode B: enough candidates ARE feasible in principle, but BUDGET
        # can't fund min_names of them simultaneously -- the minimum budget
        # that would, independent of how this particular run allocated.
        if len(feasible) >= min_names:
            cheapest_n = sorted(f["lot_cost"] for f in feasible)[:min_names]
            required_budget = sum(cheapest_n) / (1.0 - cash_reserve_pct)
            if required_budget > budget:
                suggestions.append(
                    f"raise BUDGET to >= ${required_budget:,.2f} to fund {min_names} names at 1 lot "
                    f"each (currently ${budget:,.2f})"
                )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "spot_prices_as_of": spot_prices_as_of,
        "budget": budget,
        "cash_reserve_pct": cash_reserve_pct,
        "investable_budget": investable_budget,
        "max_price_per_share": max_price_per_share,
        "max_weight_per_name": max_weight_per_name,
        "min_names": min_names,
        "capital_used": capital_used,
        "cash_left": cash_left,
        "names_funded": names_funded,
        "min_names_met": names_funded >= min_names,
        "symbols": symbols_out,
        "dropped": dropped,
        "suggestions": suggestions,
    }


def _print_plan(plan: dict) -> None:
    print(f"Sizing plan generated {plan['generated_at']} (spot prices as of {plan['spot_prices_as_of'] or 'unknown'})")
    print(
        f"Budget ${plan['budget']:,.2f} | cash reserve {plan['cash_reserve_pct']:.0%} "
        f"(${plan['budget'] - plan['investable_budget']:,.2f}) | investable ${plan['investable_budget']:,.2f}"
    )
    print()

    funded = {s: v for s, v in plan["symbols"].items() if v["lots"] > 0}
    if funded:
        print(f"{'SYMBOL':<8} {'PRICE':>10} {'LOT COST':>12} {'LOTS':>5} {'CAPITAL':>12} {'WEIGHT':>8}")
        for symbol, v in sorted(funded.items(), key=lambda kv: -kv[1]["capital"]):
            print(
                f"{symbol:<8} {v['spot_price']:>10.2f} {v['lot_cost']:>12,.2f} {v['lots']:>5} "
                f"{v['capital']:>12,.2f} {v['weight']:>7.1%}"
            )
    else:
        print("No symbols funded.")

    print(
        f"\nCapital used: ${plan['capital_used']:,.2f}  |  Cash left: ${plan['cash_left']:,.2f}  |  "
        f"Names funded: {plan['names_funded']} (min {plan['min_names']})"
    )

    if plan["dropped"]:
        print("\nDROPPED:")
        for symbol, d in sorted(plan["dropped"].items()):
            print(f"  {symbol}: {d['reason']}")

    if not plan["min_names_met"]:
        print(f"\n!! Only {plan['names_funded']}/{plan['min_names']} names funded.")
        for s in plan["suggestions"]:
            print(f"   - {s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screened", default=DEFAULT_SCREENED_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--budget", type=float, default=SIZER_BUDGET)
    parser.add_argument("--max-price-per-share", type=float, default=SIZER_MAX_PRICE_PER_SHARE)
    parser.add_argument("--min-names", type=int, default=SIZER_MIN_NAMES)
    parser.add_argument("--max-weight-per-name", type=float, default=SIZER_MAX_WEIGHT_PER_NAME)
    parser.add_argument("--cash-reserve-pct", type=float, default=SIZER_CASH_RESERVE_PCT)
    args = parser.parse_args()

    if not os.path.exists(args.screened):
        print(f"No screened symbols at {args.screened} — run screener.py first.")
        raise SystemExit(1)

    plan = compute_sizing_plan(
        args.screened, args.budget, args.max_price_per_share, args.min_names,
        args.max_weight_per_name, args.cash_reserve_pct,
    )
    _print_plan(plan)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\nWrote sizing plan to {args.out}")
