import json

import pytest

import sizer


def _write_screened(tmp_path, results, generated_at="2026-10-21"):
    path = tmp_path / "screened_symbols.json"
    path.write_text(json.dumps({"generated_at": generated_at, "results": results}))
    return str(path)


def _passed(symbol, spot_price):
    return {"symbol": symbol, "passed": True, "reason": "", "spot_price": spot_price}


def test_basic_allocation_round_robin_cheapest_first(tmp_path):
    # AAA lot=$1000, BBB lot=$2000, investable=$4000 -- round 1 gives both
    # a lot ($3000 used, $1000 left); round 2 only AAA's lot cost ($1000)
    # still fits, so cheapest-first round-robin gives AAA 2, BBB 1.
    screened = _write_screened(tmp_path, [_passed("AAA", 10.0), _passed("BBB", 20.0)])
    plan = sizer.compute_sizing_plan(
        screened, budget=4000.0, max_price_per_share=100.0, min_names=1,
        max_weight_per_name=1.0, cash_reserve_pct=0.0,  # investable = 4000
    )

    assert plan["symbols"]["AAA"]["lots"] == 2
    assert plan["symbols"]["BBB"]["lots"] == 1
    assert plan["capital_used"] == pytest.approx(1000 * 2 + 2000 * 1)
    assert plan["cash_left"] == pytest.approx(4000 - 4000)
    assert plan["min_names_met"] is True


def test_price_cap_drops_symbol_with_explicit_reason(tmp_path):
    screened = _write_screened(tmp_path, [_passed("CHEAP", 10.0), _passed("EXPENSIVE", 500.0)])
    plan = sizer.compute_sizing_plan(screened, budget=20000.0, max_price_per_share=35.0, min_names=1)

    assert "EXPENSIVE" not in plan["symbols"]
    assert "EXPENSIVE" in plan["dropped"]
    assert "exceeds MAX_PRICE_PER_SHARE" in plan["dropped"]["EXPENSIVE"]["reason"]
    assert plan["dropped"]["EXPENSIVE"]["spot_price"] == 500.0


def test_weight_cap_drops_symbol_whose_single_lot_is_too_big(tmp_path):
    # lot cost = 100*300 = $30,000; weight cap = 20% of $20,000 = $4,000.
    screened = _write_screened(tmp_path, [_passed("BIGLOT", 300.0)])
    plan = sizer.compute_sizing_plan(
        screened, budget=20000.0, max_price_per_share=1000.0, min_names=1, max_weight_per_name=0.20,
    )

    assert "BIGLOT" not in plan["symbols"]
    assert "exceeds the MAX_WEIGHT_PER_NAME cap" in plan["dropped"]["BIGLOT"]["reason"]


def test_never_silently_drops_budget_exhausted_symbol(tmp_path):
    # Both feasible on their own, but together exceed the investable budget.
    screened = _write_screened(tmp_path, [_passed("AAA", 100.0), _passed("BBB", 100.0)])
    plan = sizer.compute_sizing_plan(
        screened, budget=11000.0, max_price_per_share=1000.0, min_names=1,
        max_weight_per_name=1.0, cash_reserve_pct=0.0,
    )
    # investable = 11000, AAA lot=10000 takes it first (cheapest-first tie ->
    # stable sort keeps AAA before BBB since both cost the same and AAA
    # appears first in the input), leaving nothing for BBB.
    funded = [s for s, v in plan["symbols"].items() if v["lots"] > 0]
    unfunded = [s for s, v in plan["symbols"].items() if v["lots"] == 0]
    assert len(funded) == 1
    assert len(unfunded) == 1
    dropped_symbol = unfunded[0]
    assert dropped_symbol in plan["dropped"]
    assert "budget was exhausted" in plan["dropped"][dropped_symbol]["reason"]


def test_missing_spot_price_is_dropped_not_crashed(tmp_path):
    results = [{"symbol": "NOPRICE", "passed": True, "reason": ""}]  # no spot_price key at all
    screened = _write_screened(tmp_path, results)
    plan = sizer.compute_sizing_plan(screened, min_names=1)

    assert "NOPRICE" not in plan["symbols"]
    assert "no spot price on record" in plan["dropped"]["NOPRICE"]["reason"]


def test_failed_screen_symbols_are_ignored_entirely(tmp_path):
    results = [
        _passed("GOOD", 10.0),
        {"symbol": "BAD", "passed": False, "reason": "illiquid", "spot_price": 10.0},
    ]
    screened = _write_screened(tmp_path, results)
    plan = sizer.compute_sizing_plan(screened, min_names=1)

    assert "GOOD" in plan["symbols"]
    assert "BAD" not in plan["symbols"]
    assert "BAD" not in plan["dropped"]  # out of Sizer's scope entirely -- Scout already rejected it


def test_min_names_shortfall_suggests_price_and_weight_fixes(tmp_path):
    screened = _write_screened(tmp_path, [_passed("CHEAP", 10.0), _passed("EXPENSIVE", 500.0)])
    plan = sizer.compute_sizing_plan(
        screened, budget=20000.0, max_price_per_share=35.0, min_names=2, max_weight_per_name=0.20,
    )

    assert plan["min_names_met"] is False
    assert any("MAX_PRICE_PER_SHARE" in s and "EXPENSIVE" in s for s in plan["suggestions"])


def test_min_names_shortfall_suggests_budget_increase_when_candidates_are_feasible(tmp_path):
    # 3 feasible candidates, each affordable on its own, but budget can't
    # fund all 3 simultaneously (and MIN_NAMES=3) -- pure Mode B case.
    screened = _write_screened(
        tmp_path, [_passed("AAA", 50.0), _passed("BBB", 50.0), _passed("CCC", 50.0)],
    )
    plan = sizer.compute_sizing_plan(
        screened, budget=6000.0, max_price_per_share=1000.0, min_names=3,
        max_weight_per_name=1.0, cash_reserve_pct=0.0,
    )
    # lot cost = $5000 each; only 1 fits in a $6000 budget.
    assert plan["min_names_met"] is False
    assert any("raise BUDGET" in s for s in plan["suggestions"])


def test_min_names_met_has_no_suggestions(tmp_path):
    screened = _write_screened(tmp_path, [_passed("AAA", 10.0)])
    plan = sizer.compute_sizing_plan(screened, budget=5000.0, min_names=1)
    assert plan["min_names_met"] is True
    assert plan["suggestions"] == []


def test_timestamps_present(tmp_path):
    screened = _write_screened(tmp_path, [_passed("AAA", 10.0)], generated_at="2026-10-15")
    plan = sizer.compute_sizing_plan(screened, min_names=1)
    assert plan["spot_prices_as_of"] == "2026-10-15"
    assert plan["generated_at"]  # non-empty, a real timestamp string
