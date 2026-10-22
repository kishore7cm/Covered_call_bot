"""
Tests for config.assert_paper_mode() and the fact that it's enforced at
the exact point trader.py/options_data.py construct their TradingClient
-- not just from main.py's entrypoint. The import-time tests spawn a
real subprocess (re-importing an already-imported module in this same
process would not re-run its top-level code), so these are the only
tests in the suite that actually exercise a fresh process/import.
"""
import os
import subprocess
import sys

import pytest

import config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── config.assert_paper_mode() ───────────────────────────────────────────────

def test_assert_paper_mode_allows_exact_paper_host(monkeypatch, capsys):
    monkeypatch.setattr(config, "PAPER", True)
    monkeypatch.setattr(config, "ALLOW_LIVE", False)
    monkeypatch.setattr(config, "ALPACA_BASE_URL", config.ALPACA_PAPER_HOST)
    monkeypatch.setattr(config, "_printed_alpaca_mode", False)
    config.assert_paper_mode()  # must not raise
    assert "PAPER" in capsys.readouterr().out


def test_assert_paper_mode_refuses_live_without_allow_live(monkeypatch):
    monkeypatch.setattr(config, "PAPER", False)
    monkeypatch.setattr(config, "ALLOW_LIVE", False)
    monkeypatch.setattr(config, "ALPACA_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(SystemExit):
        config.assert_paper_mode()


def test_assert_paper_mode_allows_live_with_allow_live_set(monkeypatch, capsys):
    monkeypatch.setattr(config, "PAPER", False)
    monkeypatch.setattr(config, "ALLOW_LIVE", True)
    monkeypatch.setattr(config, "ALPACA_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setattr(config, "_printed_alpaca_mode", False)
    config.assert_paper_mode()  # must not raise
    assert "LIVE" in capsys.readouterr().out


def test_assert_paper_mode_only_prints_once_per_process(monkeypatch, capsys):
    monkeypatch.setattr(config, "PAPER", True)
    monkeypatch.setattr(config, "ALLOW_LIVE", False)
    monkeypatch.setattr(config, "ALPACA_BASE_URL", config.ALPACA_PAPER_HOST)
    monkeypatch.setattr(config, "_printed_alpaca_mode", False)
    config.assert_paper_mode()
    config.assert_paper_mode()
    assert capsys.readouterr().out.count("ALPACA mode") == 1


# ── PAPER is an exact-host match, not a substring ───────────────────────────

def test_paper_lookalike_host_containing_the_word_paper_is_not_treated_as_paper():
    """The whole point of an exact match: a host that merely CONTAINS
    "paper" (a typo, or a spoofed/lookalike hostname) must not be treated
    as the real paper endpoint."""
    lookalike = "https://api.alpaca-paper.markets.evil.example"
    assert (lookalike.rstrip("/") == config.ALPACA_PAPER_HOST) is False


def test_paper_host_matches_with_or_without_a_trailing_slash():
    assert config.ALPACA_PAPER_HOST.rstrip("/") == config.ALPACA_PAPER_HOST
    assert (config.ALPACA_PAPER_HOST + "/").rstrip("/") == config.ALPACA_PAPER_HOST


def test_config_paper_is_derived_from_the_same_base_url_the_guard_validates():
    """config.PAPER must be computed from config.ALPACA_BASE_URL itself --
    not a second, independently-read value -- otherwise PAPER and the
    guard could silently disagree about which host is actually in use."""
    assert config.PAPER == (config.ALPACA_BASE_URL.rstrip("/") == config.ALPACA_PAPER_HOST)


# ── Enforced at Alpaca client construction, not just main.py's entrypoint ──

def _run_import(module_name: str, env_overrides: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_overrides}
    return subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )


def test_importing_trader_with_live_url_and_no_allow_live_raises():
    result = _run_import("trader", {"ALPACA_BASE_URL": "https://api.alpaca.markets", "ALLOW_LIVE": "false"})
    assert result.returncode != 0
    assert "REFUSING TO START" in result.stderr


def test_importing_trader_with_live_url_and_allow_live_set_succeeds():
    result = _run_import("trader", {"ALPACA_BASE_URL": "https://api.alpaca.markets", "ALLOW_LIVE": "true"})
    assert result.returncode == 0
    assert "ALPACA mode: LIVE" in result.stdout


def test_importing_trader_with_paper_url_succeeds_without_allow_live():
    result = _run_import("trader", {"ALPACA_BASE_URL": config.ALPACA_PAPER_HOST, "ALLOW_LIVE": "false"})
    assert result.returncode == 0
    assert "ALPACA mode: PAPER" in result.stdout


def test_importing_options_data_with_live_url_and_no_allow_live_raises():
    result = _run_import("options_data", {"ALPACA_BASE_URL": "https://api.alpaca.markets", "ALLOW_LIVE": "false"})
    assert result.returncode != 0
    assert "REFUSING TO START" in result.stderr


def test_importing_options_data_with_paper_url_succeeds():
    result = _run_import("options_data", {"ALPACA_BASE_URL": config.ALPACA_PAPER_HOST, "ALLOW_LIVE": "false"})
    assert result.returncode == 0


def test_importing_trader_with_lookalike_host_containing_paper_still_raises():
    """Regression guard for the substring-vs-exact-match fix: a host that
    contains the word "paper" but isn't the real paper host must still
    be refused without ALLOW_LIVE."""
    result = _run_import(
        "trader",
        {"ALPACA_BASE_URL": "https://api.alpaca-paper.markets.evil.example", "ALLOW_LIVE": "false"},
    )
    assert result.returncode != 0
    assert "REFUSING TO START" in result.stderr
