"""
Tests for orchestrator.py's filter_by_capital_caps() -- the gate between
sizer.py and backtest.py that keeps a symbol Oracle can't actually trade
at this bot's own position size (MAX_CAPITAL_PER_SYMBOL / a 100-share
lot, or SIZER_MAX_PRICE_PER_SHARE) from ever reaching the backtest.
"""
import json

import orchestrator
from config import MAX_CAPITAL_PER_SYMBOL, SHARES_PER_LOT, SIZER_MAX_PRICE_PER_SHARE


def _write_screened(tmp_path, results):
    path = tmp_path / "screened.json"
    path.write_text(json.dumps({"generated_at": "2026-10-16", "results": results}))
    return str(path)


def test_cheap_symbol_passes_through_unchanged(tmp_path):
    price = min(30.0, SIZER_MAX_PRICE_PER_SHARE - 1, MAX_CAPITAL_PER_SYMBOL / SHARES_PER_LOT - 1)
    screened = _write_screened(tmp_path, [{"symbol": "CHEAP", "passed": True, "spot_price": price}])
    out_path = str(tmp_path / "filtered.json")

    dropped = orchestrator.filter_by_capital_caps(screened, out_path)

    assert dropped == []
    with open(out_path) as f:
        payload = json.load(f)
    assert payload["results"][0]["passed"] is True


def test_symbol_over_max_capital_per_symbol_is_dropped(monkeypatch, tmp_path):
    # With current defaults, SIZER_MAX_PRICE_PER_SHARE ($35) x 100 shares
    # ($3,500) is already under MAX_CAPITAL_PER_SYMBOL ($4,000), so the
    # price cap always binds first in practice -- raise the price cap
    # here to isolate the MAX_CAPITAL_PER_SYMBOL branch specifically.
    monkeypatch.setattr(orchestrator, "SIZER_MAX_PRICE_PER_SHARE", 1000.0)
    price = (MAX_CAPITAL_PER_SYMBOL / SHARES_PER_LOT) + 1
    screened = _write_screened(tmp_path, [{"symbol": "PRICEY", "passed": True, "spot_price": price}])
    out_path = str(tmp_path / "filtered.json")

    dropped = orchestrator.filter_by_capital_caps(screened, out_path)

    assert len(dropped) == 1
    assert dropped[0][0] == "PRICEY"
    assert "MAX_CAPITAL_PER_SYMBOL" in dropped[0][1]
    with open(out_path) as f:
        payload = json.load(f)
    assert payload["results"][0]["passed"] is False
    assert "MAX_CAPITAL_PER_SYMBOL" in payload["results"][0]["reason"]


def test_symbol_over_sizer_max_price_per_share_is_dropped(tmp_path):
    price = SIZER_MAX_PRICE_PER_SHARE + 1
    screened = _write_screened(tmp_path, [{"symbol": "TOOEXPENSIVE", "passed": True, "spot_price": price}])
    out_path = str(tmp_path / "filtered.json")

    dropped = orchestrator.filter_by_capital_caps(screened, out_path)

    assert len(dropped) == 1
    assert dropped[0][0] == "TOOEXPENSIVE"
    assert "SIZER_MAX_PRICE_PER_SHARE" in dropped[0][1]


def test_symbol_with_no_spot_price_is_dropped(tmp_path):
    screened = _write_screened(tmp_path, [{"symbol": "NOPRICE", "passed": True, "spot_price": None}])
    out_path = str(tmp_path / "filtered.json")

    dropped = orchestrator.filter_by_capital_caps(screened, out_path)

    assert len(dropped) == 1
    assert dropped[0][0] == "NOPRICE"
    assert "no spot price" in dropped[0][1]


def test_symbols_that_already_failed_screening_are_left_alone(tmp_path):
    """A symbol Scout already failed (passed=False) must not show up in
    `dropped` here -- this filter is only about symbols that DID pass
    screening but can't actually be sized."""
    screened = _write_screened(tmp_path, [
        {"symbol": "ILLIQUID", "passed": False, "reason": "thin liquidity", "spot_price": 10.0},
    ])
    out_path = str(tmp_path / "filtered.json")

    dropped = orchestrator.filter_by_capital_caps(screened, out_path)

    assert dropped == []
    with open(out_path) as f:
        payload = json.load(f)
    assert payload["results"][0]["reason"] == "thin liquidity"  # untouched


def test_mixed_batch_keeps_cheap_drops_expensive(tmp_path):
    cheap_price = min(20.0, SIZER_MAX_PRICE_PER_SHARE - 1, MAX_CAPITAL_PER_SYMBOL / SHARES_PER_LOT - 1)
    screened = _write_screened(tmp_path, [
        {"symbol": "CHEAP", "passed": True, "spot_price": cheap_price},
        {"symbol": "AAPL", "passed": True, "spot_price": 348.34},
        {"symbol": "MSFT", "passed": True, "spot_price": 521.60},
    ])
    out_path = str(tmp_path / "filtered.json")

    dropped = orchestrator.filter_by_capital_caps(screened, out_path)

    dropped_symbols = {s for s, _ in dropped}
    assert dropped_symbols == {"AAPL", "MSFT"}
    with open(out_path) as f:
        payload = json.load(f)
    by_symbol = {r["symbol"]: r for r in payload["results"]}
    assert by_symbol["CHEAP"]["passed"] is True
    assert by_symbol["AAPL"]["passed"] is False
    assert by_symbol["MSFT"]["passed"] is False
