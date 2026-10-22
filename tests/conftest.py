"""
Blanket test-safety net. Autouse means every test in this suite gets this
fixture whether it asks for it or not — a test that forgets to monkeypatch
trader._client (or options_data's clients) still cannot reach the real
Alpaca API, it just gets a RuntimeError instead of a silent live call.
Individual tests are still free to monkeypatch._client themselves (e.g. to
a FakeBroker) to exercise the real code paths; this fixture only closes the
gap for whatever they DON'T patch.
"""
import pytest

import options_data
import trader


class _ForbiddenBrokerClient:
    """Raises on any attribute access — standing in for a client no test
    should ever actually call."""

    def __getattr__(self, name):
        raise RuntimeError(
            f"Test attempted a real broker/data call via .{name}() — "
            "monkeypatch the client with a fake before exercising this path."
        )


@pytest.fixture(autouse=True)
def _block_real_broker(monkeypatch):
    monkeypatch.setattr(trader, "_client", _ForbiddenBrokerClient())
    monkeypatch.setattr(options_data, "_trading_client", _ForbiddenBrokerClient())
    monkeypatch.setattr(options_data, "_data_client", _ForbiddenBrokerClient())
