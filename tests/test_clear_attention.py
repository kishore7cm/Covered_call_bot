import clear_attention
import trader


def test_clears_existing_flag_and_logs_reason(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(trader, "_NEEDS_ATTENTION_PATH", str(tmp_path / "needs_attention.json"))
    logged = []
    monkeypatch.setattr(clear_attention, "log_error", lambda module, msg: logged.append((module, msg)))
    trader.mark_needs_attention("AAPL", "order state unknown")

    rc = clear_attention.main(["AAPL", "confirmed broker state by hand"])

    assert rc == 0
    assert trader.check_needs_attention("AAPL") is None
    assert len(logged) == 1
    assert "AAPL" in logged[0][1]
    assert "order state unknown" in logged[0][1]
    assert "confirmed broker state by hand" in logged[0][1]
    out = capsys.readouterr().out
    assert "cleared" in out.lower()


def test_no_flag_set_does_not_clear_or_log(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(trader, "_NEEDS_ATTENTION_PATH", str(tmp_path / "needs_attention.json"))
    logged = []
    monkeypatch.setattr(clear_attention, "log_error", lambda module, msg: logged.append((module, msg)))

    rc = clear_attention.main(["AAPL", "nothing to clear"])

    assert rc == 1
    assert logged == []
    out = capsys.readouterr().out
    assert "nothing to clear" in out.lower()


def test_only_clears_the_named_symbol(monkeypatch, tmp_path):
    monkeypatch.setattr(trader, "_NEEDS_ATTENTION_PATH", str(tmp_path / "needs_attention.json"))
    monkeypatch.setattr(clear_attention, "log_error", lambda module, msg: None)
    trader.mark_needs_attention("AAPL", "reason A")
    trader.mark_needs_attention("MSFT", "reason M")

    clear_attention.main(["AAPL", "resolved"])

    assert trader.check_needs_attention("AAPL") is None
    assert trader.check_needs_attention("MSFT") == "reason M"


def test_reason_argument_is_required(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(trader, "_NEEDS_ATTENTION_PATH", str(tmp_path / "needs_attention.json"))
    try:
        clear_attention.main(["AAPL"])
        raised = False
    except SystemExit:
        raised = True
    assert raised  # argparse exits non-zero when a required positional is missing
