from datetime import date

import pytest

import iv_logger


@pytest.fixture(autouse=True)
def sqlite_fallback(tmp_path, monkeypatch):
    """Force the SQLite fallback path at a throwaway location for every test
    in this file -- never touch a real Postgres DB or the real
    pipeline_state/iv_daily.db."""
    monkeypatch.setattr(iv_logger, "DATABASE_URL", "")
    monkeypatch.setattr(iv_logger, "SQLITE_PATH", str(tmp_path / "iv_daily.db"))
    iv_logger.init_schema()


def test_record_and_read_back():
    iv_logger.record_iv_daily("AAPL", 0.28, 230.50, as_of=date(2026, 10, 20))
    history = iv_logger.get_history("AAPL")
    assert history == [{"date": date(2026, 10, 20), "atm_iv": 0.28, "underlying_close": 230.50}]


def test_idempotent_same_symbol_same_day_upserts_not_duplicates():
    iv_logger.record_iv_daily("AAPL", 0.28, 230.50, as_of=date(2026, 10, 20))
    iv_logger.record_iv_daily("AAPL", 0.31, 231.75, as_of=date(2026, 10, 20))  # re-run same day
    history = iv_logger.get_history("AAPL")
    assert len(history) == 1
    assert history[0]["atm_iv"] == 0.31
    assert history[0]["underlying_close"] == 231.75


def test_multiple_days_accumulate_oldest_first():
    iv_logger.record_iv_daily("AAPL", 0.20, 200.0, as_of=date(2026, 10, 18))
    iv_logger.record_iv_daily("AAPL", 0.25, 205.0, as_of=date(2026, 10, 20))
    iv_logger.record_iv_daily("AAPL", 0.22, 202.0, as_of=date(2026, 10, 19))
    history = iv_logger.get_history("AAPL")
    assert [h["date"] for h in history] == [date(2026, 10, 18), date(2026, 10, 19), date(2026, 10, 20)]


def test_symbols_are_isolated():
    iv_logger.record_iv_daily("AAPL", 0.28, 230.0, as_of=date(2026, 10, 20))
    iv_logger.record_iv_daily("MSFT", 0.19, 410.0, as_of=date(2026, 10, 20))
    assert len(iv_logger.get_history("AAPL")) == 1
    assert len(iv_logger.get_history("MSFT")) == 1
    assert iv_logger.get_history("GOOG") == []


def test_log_symbol_records_and_reports_ivr(monkeypatch):
    monkeypatch.setattr(iv_logger, "_get_spot_price", lambda symbol: 230.0)
    monkeypatch.setattr(iv_logger, "get_atm_iv", lambda symbol, spot, target_dte: 0.28)

    result = iv_logger.log_symbol("AAPL")

    assert "AAPL" in result
    assert "insufficient history" in result  # only 1 point logged, far under the 252 floor
    assert iv_logger.get_history("AAPL")[0]["atm_iv"] == 0.28


def test_log_symbol_fails_soft_on_missing_spot(monkeypatch):
    monkeypatch.setattr(iv_logger, "_get_spot_price", lambda symbol: None)
    result = iv_logger.log_symbol("AAPL")
    assert "FAILED" in result
    assert iv_logger.get_history("AAPL") == []


def test_run_continues_past_one_symbols_failure(monkeypatch):
    def fake_spot(symbol):
        if symbol == "BAD":
            raise RuntimeError("boom")
        return 100.0

    monkeypatch.setattr(iv_logger, "_get_spot_price", fake_spot)
    monkeypatch.setattr(iv_logger, "get_atm_iv", lambda symbol, spot, target_dte: 0.3)

    results = iv_logger.run(["AAPL", "BAD", "MSFT"])

    assert len(results) == 3
    assert "FAILED" in results[1]
    assert iv_logger.get_history("AAPL")
    assert iv_logger.get_history("MSFT")
    assert iv_logger.get_history("BAD") == []
