"""
PAPER_AUTOPILOT: paper-only auto-approval + DRY_RUN default flip, and the
MANUAL_SYMBOLS hands-off list. Config import-time behavior is checked in a
real subprocess (same pattern as test_config.py), since config is already
imported in this process.
"""
import json
import os
import subprocess
import sys

import pytest

import orchestrator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = "https://paper-api.alpaca.markets"


def _py(code: str, env_overrides: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("DRY_RUN", "PAPER_AUTOPILOT", "ALLOW_LIVE")}
    env.update(env_overrides)
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=30)


# ── config ───────────────────────────────────────────────────────────────────

def test_autopilot_refused_against_live_host_even_with_allow_live():
    r = _py("import config", {"PAPER_AUTOPILOT": "true", "ALPACA_BASE_URL": "https://api.alpaca.markets",
                              "ALLOW_LIVE": "true"})
    assert r.returncode != 0
    assert "PAPER_AUTOPILOT" in r.stderr


@pytest.mark.parametrize("env, expected", [
    ({}, "True"),                                               # default: dry run
    ({"PAPER_AUTOPILOT": "true"}, "False"),                     # autopilot flips the default
    ({"PAPER_AUTOPILOT": "true", "DRY_RUN": "true"}, "True"),   # explicit DRY_RUN still wins
])
def test_dry_run_default(env, expected):
    r = _py("import config; print(config.DRY_RUN)", {"ALPACA_BASE_URL": PAPER, **env})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == expected


def test_spcx_is_hands_off():
    import config
    assert "SPCX" in config.MANUAL_SYMBOLS
    assert "SPCX" in config.HANDS_OFF_SYMBOLS
    assert set(config.OTHER_BOTS_SYMBOLS) <= config.HANDS_OFF_SYMBOLS


# ── apply_watchlist.py ───────────────────────────────────────────────────────

def _write(path, payload):
    path.write_text(json.dumps(payload))
    return str(path)


def test_apply_watchlist_autopilot_refused_without_flag(tmp_path):
    proposed = _write(tmp_path / "p.json", {"symbols": {"SOFI": {"dte": 21}}})
    active = str(tmp_path / "a.json")
    r = _py(f"import runpy, sys; sys.argv=['apply_watchlist.py','--confirm','--approved-by','autopilot',"
            f"'--proposed',{proposed!r},'--active',{active!r}]; runpy.run_path('apply_watchlist.py', run_name='__main__')",
            {"ALPACA_BASE_URL": PAPER})
    assert r.returncode != 0
    assert not os.path.exists(active)


def test_apply_watchlist_strips_hands_off_and_records_approver(tmp_path):
    stats = {"avg_annualized_yield": 0.2, "in_sample_yield": 0.2, "out_of_sample_yield": 0.18, "oos_gap_pp": 2.0}
    proposed = _write(tmp_path / "p.json", {"generated_at": "2026-10-23", "symbols": {
        "SOFI": {"dte": 21, **stats}, "SPCX": {"dte": 14, **stats}, "AAPL": {"dte": 14, **stats}}})
    active = str(tmp_path / "a.json")
    r = _py(f"import runpy, sys; sys.argv=['apply_watchlist.py','--confirm','--approved-by','autopilot',"
            f"'--proposed',{proposed!r},'--active',{active!r},'--screened',{str(tmp_path / 'none.json')!r}]; "
            f"runpy.run_path('apply_watchlist.py', run_name='__main__')",
            {"ALPACA_BASE_URL": PAPER, "PAPER_AUTOPILOT": "true"})
    assert r.returncode == 0, r.stdout + r.stderr
    payload = json.loads(open(active).read())
    assert list(payload["symbols"]) == ["SOFI"]
    assert payload["approved_by"] == "autopilot"
    assert payload["confirmed_at"]


# ── orchestrator.run_research ────────────────────────────────────────────────

@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    calls = []
    proposed_path = tmp_path / "proposed.json"
    log_path = tmp_path / "approvals.jsonl"

    def fake_run(args):
        calls.append(args)
        return True

    monkeypatch.setattr(orchestrator, "_run", fake_run)
    monkeypatch.setattr(orchestrator, "filter_by_capital_caps", lambda a, b: [])
    monkeypatch.setattr(orchestrator, "_proposed_symbols", lambda path=None: orchestrator_proposed())
    monkeypatch.setattr(orchestrator, "AUTOPILOT_APPROVALS_LOG", str(log_path))
    monkeypatch.setattr(orchestrator, "PAPER_AUTOPILOT", True)

    state = {"proposed": ["SOFI"]}

    def orchestrator_proposed():
        return state["proposed"]

    return {"calls": calls, "state": state, "log": log_path}


def test_auto_confirm_refused_without_autopilot(monkeypatch):
    monkeypatch.setattr(orchestrator, "PAPER_AUTOPILOT", False)
    with pytest.raises(RuntimeError):
        orchestrator.run_research(["SOFI"], None, auto_confirm=True)


def test_hands_off_symbols_never_reach_screener(fake_pipeline):
    orchestrator.run_research(["SOFI", "SPCX", "AAPL"], None, auto_confirm=True)
    screener_call = fake_pipeline["calls"][0]
    assert screener_call[0] == "screener.py"
    assert "SOFI" in screener_call and "SPCX" not in screener_call and "AAPL" not in screener_call


def test_auto_confirm_approves_nonempty_proposal(fake_pipeline):
    assert orchestrator.run_research(["SOFI"], None, auto_confirm=True) is True
    assert fake_pipeline["calls"][-1] == ["apply_watchlist.py", "--confirm", "--approved-by", "autopilot"]
    record = json.loads(fake_pipeline["log"].read_text().splitlines()[-1])
    assert record["approved"] is True and record["symbols"] == ["SOFI"]


def test_auto_confirm_never_approves_empty_proposal(fake_pipeline):
    fake_pipeline["state"]["proposed"] = []
    assert orchestrator.run_research(["SOFI"], None, auto_confirm=True) is False
    assert not any("--confirm" in c for c in fake_pipeline["calls"])
    record = json.loads(fake_pipeline["log"].read_text().splitlines()[-1])
    assert record["approved"] is False


def test_manual_mode_only_prints_diff(fake_pipeline):
    assert orchestrator.run_research(["SOFI"], None) is True
    assert fake_pipeline["calls"][-1] == ["apply_watchlist.py"]
    assert not fake_pipeline["log"].exists()
