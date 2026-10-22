import subprocess

import alerts


def test_applescript_quote_escapes_backslash_and_double_quote():
    assert alerts._applescript_quote('say "hi"') == '"say \\"hi\\""'
    assert alerts._applescript_quote("back\\slash") == '"back\\\\slash"'


def test_notify_prints_and_appends_to_log_file(monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "alerts.log"
    monkeypatch.setattr(alerts, "ALERTS_LOG_PATH", str(log_path))
    monkeypatch.setattr(alerts, "_macos_notify", lambda title, message: None)

    alerts.notify("needs_attention", "AAPL: something is wrong")

    out = capsys.readouterr().out
    assert "[NEEDS_ATTENTION] AAPL: something is wrong" in out
    assert log_path.read_text() == "[NEEDS_ATTENTION] AAPL: something is wrong\n"


def test_notify_appends_multiple_lines(monkeypatch, tmp_path):
    log_path = tmp_path / "alerts.log"
    monkeypatch.setattr(alerts, "ALERTS_LOG_PATH", str(log_path))
    monkeypatch.setattr(alerts, "_macos_notify", lambda title, message: None)

    alerts.notify("partial_fill", "AAPL: filled 60/100")
    alerts.notify("kill_switch", "AAPL: blocked")

    lines = log_path.read_text().splitlines()
    assert lines == ["[PARTIAL_FILL] AAPL: filled 60/100", "[KILL_SWITCH] AAPL: blocked"]


def test_notify_calls_macos_notify_with_category_message(monkeypatch, tmp_path):
    log_path = tmp_path / "alerts.log"
    monkeypatch.setattr(alerts, "ALERTS_LOG_PATH", str(log_path))
    calls = []
    monkeypatch.setattr(alerts, "_macos_notify", lambda title, message: calls.append((title, message)))

    alerts.notify("kill_switch", "AAPL: blocked by kill switch")

    assert len(calls) == 1
    assert calls[0][1] == "AAPL: blocked by kill switch"


def test_macos_notify_failure_does_not_raise(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no GUI session")
    monkeypatch.setattr(subprocess, "run", boom)
    alerts._macos_notify("title", "message")  # must not raise


def test_macos_notify_invokes_osascript_with_display_notification(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)))
    alerts._macos_notify("Covered Call Bot", "AAPL: something happened")
    assert len(calls) == 1
    args = calls[0][0][0]
    assert args[0] == "osascript"
    assert "display notification" in args[2]
    assert "AAPL: something happened" in args[2]


def test_notify_failure_to_write_log_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(alerts, "ALERTS_LOG_PATH", str(tmp_path / "nonexistent_dir" / "alerts.log"))
    monkeypatch.setattr(alerts, "_macos_notify", lambda title, message: None)
    alerts.notify("needs_attention", "AAPL: test")  # must not raise even though the dir doesn't exist
