"""
Black-Scholes pricing and delta calculation for European-style calls.

Alpaca's free "indicative" options feed (what this paper account gets by
default) returns NBBO quotes and trades but no Greeks or implied volatility.
The real-time `opra` feed has Greeks but requires signing Alpaca's OPRA
market data agreement, which this account hasn't done. Rather than add that
dependency, we back out implied vol from the indicative feed's bid/ask mid
and derive delta ourselves.

Pure functions, no I/O — safe to unit test without touching the network.
American-style early exercise is ignored (standard simplification for
short-dated, cash-settled retail covered-call screening; the error vs. true
American pricing is small for calls on non-dividend-paying windows, and this
bot explicitly should account for dividend risk around ex-div dates in
trader.py, not here).
"""
import math

from scipy.optimize import brentq
from scipy.stats import norm


def bs_call_price(spot: float, strike: float, t_years: float, rate: float, iv: float) -> float:
    """Black-Scholes price of a European call."""
    if t_years <= 0 or iv <= 0:
        return max(spot - strike, 0.0)

    d1 = (math.log(spot / strike) + (rate + 0.5 * iv ** 2) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)
    return spot * norm.cdf(d1) - strike * math.exp(-rate * t_years) * norm.cdf(d2)


def bs_call_delta(spot: float, strike: float, t_years: float, rate: float, iv: float) -> float:
    """Black-Scholes delta of a European call, in [0, 1]."""
    if t_years <= 0 or iv <= 0:
        return 1.0 if spot > strike else 0.0

    d1 = (math.log(spot / strike) + (rate + 0.5 * iv ** 2) * t_years) / (iv * math.sqrt(t_years))
    return norm.cdf(d1)


def bs_call_theta_per_day(spot: float, strike: float, t_years: float, rate: float, iv: float) -> float:
    """
    Black-Scholes theta of a European call, converted to $/day (the raw
    formula gives $/year). Negative for a long call — i.e. a long call loses
    this much value per day from time decay alone. For a *short* call (what
    this bot actually holds), the seller's daily P&L from theta is the
    negation of this value — positive, since time decay favors the seller.
    Caller is responsible for sign convention; this returns the standard
    long-call theta.
    """
    if t_years <= 0 or iv <= 0:
        return 0.0

    d1 = (math.log(spot / strike) + (rate + 0.5 * iv ** 2) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)

    term1 = -(spot * norm.pdf(d1) * iv) / (2 * math.sqrt(t_years))
    term2 = -rate * strike * math.exp(-rate * t_years) * norm.cdf(d2)
    theta_per_year = term1 + term2
    return theta_per_year / 365.0


def implied_vol(
    mid_price: float,
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    lo: float = 0.01,
    hi: float = 5.0,
) -> float | None:
    """
    Solve for implied vol such that bs_call_price(...) == mid_price, via
    Brent's method. Returns None if no root exists in [lo, hi] (e.g. the
    quoted mid is below intrinsic value — bad/stale quote, skip this strike).
    """
    if mid_price <= 0 or t_years <= 0:
        return None

    intrinsic = max(spot - strike, 0.0)
    if mid_price < intrinsic:
        return None  # stale/crossed quote — not solvable, caller should skip

    def f(iv):
        return bs_call_price(spot, strike, t_years, rate, iv) - mid_price

    try:
        if f(lo) > 0 or f(hi) < 0:
            return None  # price outside the range spannable by [lo, hi] vol
        return brentq(f, lo, hi, xtol=1e-6)
    except ValueError:
        return None


def strike_from_delta(
    spot: float,
    t_years: float,
    rate: float,
    iv: float,
    target_delta: float,
    lo_mult: float = 0.5,
    hi_mult: float = 3.0,
) -> float | None:
    """
    Solve for the strike whose BS call delta equals target_delta, via Brent's
    method. The inverse of the usual quote -> IV -> delta direction above —
    needed when simulating a strike pick against an assumed IV rather than a
    real market quote (e.g. backtesting). Returns None if no root exists in
    [spot*lo_mult, spot*hi_mult] (shouldn't happen for 0 < target_delta < 1
    with sane multipliers, but mirrors implied_vol's defensive None return).
    """
    if t_years <= 0 or iv <= 0 or not (0.0 < target_delta < 1.0):
        return None

    lo, hi = spot * lo_mult, spot * hi_mult

    def f(strike):
        return bs_call_delta(spot, strike, t_years, rate, iv) - target_delta

    try:
        if f(lo) < 0 or f(hi) > 0:
            return None  # target delta outside what's spannable by [lo, hi] strikes
        return brentq(f, lo, hi, xtol=1e-6)
    except ValueError:
        return None


def delta_from_quote(
    mid_price: float,
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
) -> float | None:
    """Convenience: mid price -> IV -> delta in one call. None if unsolvable."""
    iv = implied_vol(mid_price, spot, strike, t_years, rate)
    if iv is None:
        return None
    return bs_call_delta(spot, strike, t_years, rate, iv)
