"""
greeks.py's Black-Scholes functions must return plain Python floats, not
numpy scalar types -- scipy.stats.norm.cdf/.pdf return numpy.float64 even
for plain-float inputs, and psycopg2 can't adapt that type as a SQL
parameter. This is exactly the bug that silently dropped a real CSCO
option sale from option_trades (logger.log_option_trade's INSERT failed
with `schema "np" does not exist`, swallowed by its own except-and-log).
"""
from greeks import bs_call_delta, bs_call_price, bs_call_theta_per_day


def test_bs_call_delta_returns_plain_float():
    result = bs_call_delta(spot=100.0, strike=105.0, t_years=30 / 365.0, rate=0.04, iv=0.25)
    assert type(result) is float


def test_bs_call_price_returns_plain_float():
    result = bs_call_price(spot=100.0, strike=105.0, t_years=30 / 365.0, rate=0.04, iv=0.25)
    assert type(result) is float


def test_bs_call_theta_per_day_returns_plain_float():
    result = bs_call_theta_per_day(spot=100.0, strike=105.0, t_years=30 / 365.0, rate=0.04, iv=0.25)
    assert type(result) is float
