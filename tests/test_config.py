"""
Tests for config.validate_live_delta() (Phase 3, Task 8) -- the SOLE place
the live per-symbol delta bounds/reason logic lives, shared by main.py
(resolving a per-symbol watchlist delta before a real order) and
apply_watchlist.py (a review-time warning, display only). Also covers
MIN_LIVE_DELTA/MAX_LIVE_DELTA's own import-time sanity assert.
"""
import os
import subprocess
import sys

import config
from config import INTENT_DELTA_BANDS, MAX_LIVE_DELTA, MIN_LIVE_DELTA, validate_live_delta

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── validate_live_delta ───────────────────────────────────────────────────────

def test_validate_live_delta_valid_value():
    is_valid, reason, value = validate_live_delta(0.30)
    assert is_valid is True
    assert reason is None
    assert value == 0.30


def test_validate_live_delta_missing():
    is_valid, reason, value = validate_live_delta(None)
    assert is_valid is False
    assert reason == "missing"
    assert value is None


def test_validate_live_delta_non_numeric():
    is_valid, reason, value = validate_live_delta("not-a-number")
    assert is_valid is False
    assert "invalid" in reason
    assert value is None


def test_validate_live_delta_out_of_bounds_low():
    is_valid, reason, value = validate_live_delta(MIN_LIVE_DELTA - 0.01)
    assert is_valid is False
    assert "out of bounds" in reason
    assert value is None


def test_validate_live_delta_out_of_bounds_high():
    is_valid, reason, value = validate_live_delta(MAX_LIVE_DELTA + 0.01)
    assert is_valid is False
    assert "out of bounds" in reason
    assert value is None


def test_validate_live_delta_boundary_values_are_valid():
    """Bounds are inclusive on both ends -- a watchlist delta exactly at
    MIN_LIVE_DELTA or MAX_LIVE_DELTA must not be rejected."""
    assert validate_live_delta(MIN_LIVE_DELTA)[0] is True
    assert validate_live_delta(MAX_LIVE_DELTA)[0] is True


def test_validate_live_delta_accepts_int_and_numeric_string():
    """A watchlist entry's JSON-decoded field could plausibly be an int
    (e.g. a hand-edited "0") or already a numeric string -- float() handles
    both; only genuinely non-numeric values should be rejected."""
    is_valid, _reason, value = validate_live_delta("0.25")
    assert is_valid is True
    assert value == 0.25


def test_intent_delta_bands_edges_all_pass_validation():
    """Every real Oracle-selected delta comes from INTENT_DELTA_BANDS
    (backtest._delta_buckets_for_intent sweeps within these bands) -- both
    edges of every band must clear the live bounds with room to spare, or
    a legitimate Oracle pick could get silently rejected at trade time."""
    for intent, (lo, hi) in INTENT_DELTA_BANDS.items():
        for edge in (lo, hi):
            is_valid, reason, _value = validate_live_delta(edge)
            assert is_valid, f"{intent} band edge {edge} failed validation: {reason}"


# ── MIN_LIVE_DELTA / MAX_LIVE_DELTA import-time assert ───────────────────────

def _run_import(module_name: str, env_overrides: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_overrides}
    return subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )


def test_importing_config_with_min_above_max_raises():
    result = _run_import("config", {"MIN_LIVE_DELTA": "0.60", "MAX_LIVE_DELTA": "0.50"})
    assert result.returncode != 0
    assert "MIN_LIVE_DELTA" in result.stderr
    assert "MAX_LIVE_DELTA" in result.stderr


def test_importing_config_with_bounds_outside_zero_one_raises():
    result = _run_import("config", {"MIN_LIVE_DELTA": "0.10", "MAX_LIVE_DELTA": "1.50"})
    assert result.returncode != 0


def test_importing_config_with_sane_bounds_succeeds():
    result = _run_import("config", {"MIN_LIVE_DELTA": "0.15", "MAX_LIVE_DELTA": "0.45"})
    assert result.returncode == 0


def test_default_bounds_satisfy_their_own_assert():
    # The module is already imported successfully in this process (this
    # file's own import of config above), so this just re-confirms the
    # defaults actually in effect make sense on their own terms.
    assert 0 < MIN_LIVE_DELTA < MAX_LIVE_DELTA < 1
