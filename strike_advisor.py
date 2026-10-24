"""
Strike advisor: ranks covered-call strikes across every expiration in
[ADVISOR_DTE_MIN, ADVISOR_DTE_MAX] days by VOLATILITY EDGE -- how much the
market is paying for a call above what it's worth under our own forecast
of the stock's volatility -- and handles earnings explicitly instead of
skipping every expiration that spans one.

Never imports trader.py and never places an order itself. As a CLI it's
read-only research; main.py uses pick_strike_by_edge() as its strike
selector when config.STRIKE_SELECTION == "edge" (the default since
2026-10-23), and the order still goes through trader.py's full guard stack.

Usage:
    python strike_advisor.py SOFI INTC WFC
    python strike_advisor.py SOFI --top 10 --basis SOFI=24.10 --json out.json

THE MODEL
---------
Forecast vol (sigma_ex): blend of trailing realized vols (ADVISOR_VOL_BLEND)
on daily log returns with earnings-reaction days removed, so a past
earnings gap doesn't inflate the "normal day" estimate.

Earnings jump: the extra one-off variance from the earnings event, as a
1-standard-deviation log move, estimated two ways:
  - historical: RMS of the stock's own 2-day log move around each past
    earnings date (close before -> close after, which covers both
    before-open and after-close reports), minus the 2 normal days of
    diffusion that window also contains.
  - implied: from the ATM IV term structure -- total variance of the first
    expiration AFTER earnings minus what the pre-earnings vol would
    produce over the same time. (sigma_post^2 - sigma_pre^2) * T_post.

Fair value of a call = Black-Scholes with total variance
    sigma_ex^2 * T  (+ historical jump^2 if the expiration spans earnings)
Edge = expected fill (mid minus FILL_HAIRCUT_FRACTION of the half-spread)
       minus fair value. Ranked by edge annualized as % of spot.

Hard filters (a candidate must pass all):
  - Filter 2 quote/liquidity gates (same as pick_strike: fresh, uncrossed,
    spread <= MAX_SPREAD_PCT, OI >= MIN_OPEN_INTEREST)
  - ADVISOR_MIN_DELTA <= market delta <= ADVISOR_MAX_DELTA
  - spans earnings -> strike >= spot * exp(EARNINGS_MOVE_BUFFER * max(implied, historical))
  - strike >= cost basis, when --basis is given
  - edge > 0

Caveats: the forecast ignores drift and uses realized vol as the view, so
"edge" is the variance risk premium as seen by a backward-looking
estimate -- not a guarantee. Alpaca's free indicative feed can show wide
spreads that fail the liquidity gate on names that are tight in reality.
"""
import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import (
    ADVISOR_DTE_MAX, ADVISOR_DTE_MIN, ADVISOR_MAX_DELTA, ADVISOR_MIN_DELTA,
    ADVISOR_VOL_BLEND, EARNINGS_HISTORY_YEARS, EARNINGS_MIN_HISTORY,
    EARNINGS_MOVE_BUFFER, FILL_HAIRCUT_FRACTION, RISK_FREE_RATE,
)
from greeks import bs_call_price

_log = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ── Pure math (no I/O) ───────────────────────────────────────────────────────

def earnings_positions(closes: pd.Series, earnings_dates: list[date]) -> list[int]:
    """Index positions in `closes` of each past earnings date that has a
    trading day on both sides (needed for the close-before -> close-after move)."""
    idx_dates = [ts.date() if hasattr(ts, "date") else ts for ts in closes.index]
    pos_by_date = {d: i for i, d in enumerate(idx_dates)}
    out = []
    for d in sorted(set(earnings_dates)):
        i = pos_by_date.get(d)
        if i is not None and 1 <= i < len(closes) - 1:
            out.append(i)
    return out


def clean_log_returns(closes: pd.Series, earn_pos: list[int]) -> np.ndarray:
    """Daily log returns with each earnings reaction (returns at pos and pos+1) removed."""
    c = closes.to_numpy(dtype=float)
    r = np.log(c[1:] / c[:-1])  # r[k] is the return INTO day k+1
    drop = set()
    for p in earn_pos:
        drop.update((p - 1, p))  # returns into day p and into day p+1
    keep = [k for k in range(len(r)) if k not in drop]
    return r[keep]


def forecast_vol(returns: np.ndarray, blend=ADVISOR_VOL_BLEND) -> float | None:
    """Annualized blend of trailing realized vols. None if too little data."""
    if len(returns) < 20:
        return None
    total = 0.0
    for window, weight in blend:
        tail = returns[-window:]
        total += weight * float(np.std(tail, ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    return total


def historical_jump(closes: pd.Series, earn_pos: list[int], sigma_ex: float) -> tuple[float | None, list[float]]:
    """
    (1-sd earnings jump as a log move, list of raw 2-day abs moves).
    Subtracts the 2 normal days of diffusion the 2-day window contains.
    """
    c = closes.to_numpy(dtype=float)
    moves = [math.log(c[p + 1] / c[p - 1]) for p in earn_pos]
    if not moves:
        return None, []
    rms_sq = sum(m * m for m in moves) / len(moves)
    jump_sq = max(rms_sq - 2.0 * sigma_ex ** 2 / TRADING_DAYS_PER_YEAR, 0.0)
    return math.sqrt(jump_sq), [abs(m) for m in moves]


def implied_jump(sigma_pre: float, sigma_post: float, t_post_years: float) -> float:
    """1-sd implied earnings log move from the ATM IV term structure."""
    return math.sqrt(max((sigma_post ** 2 - sigma_pre ** 2) * t_post_years, 0.0))


def fair_call(spot: float, strike: float, t_years: float, sigma_ex: float, jump: float) -> tuple[float, float]:
    """(fair price, probability of finishing ITM) under total variance sigma_ex^2*T + jump^2."""
    total_var = sigma_ex ** 2 * t_years + jump ** 2
    sigma_eff = math.sqrt(total_var / t_years)
    price = bs_call_price(spot, strike, t_years, RISK_FREE_RATE, sigma_eff)
    d2 = (math.log(spot / strike) + (RISK_FREE_RATE - 0.5 * sigma_eff ** 2) * t_years) / (sigma_eff * math.sqrt(t_years))
    return price, float(norm.cdf(d2))


def expected_fill(mid: float, spread_pct: float | None) -> float:
    half_spread = mid * (spread_pct or 0.0) / 2.0
    return mid - FILL_HAIRCUT_FRACTION * half_spread


def earnings_strike_floor(spot: float, implied: float | None, historical: float | None,
                          buffer: float = EARNINGS_MOVE_BUFFER) -> float | None:
    moves = [m for m in (implied, historical) if m is not None]
    if not moves:
        return None
    return spot * math.exp(buffer * max(moves))


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class ScoredStrike:
    occ_symbol: str
    expiration: str
    dte: int
    strike: float
    mid: float
    fill: float
    fair: float
    edge: float               # $/share
    edge_ann_pct: float       # edge / spot, annualized, in %
    premium_ann_pct: float    # fill / spot, annualized, in %
    market_delta: float
    market_iv: float
    p_assign: float           # under our forecast, not the market's
    open_interest: int | None
    spread_pct: float | None
    spans_earnings: bool


@dataclass
class SymbolAdvice:
    symbol: str
    status: str               # "ok" | "skipped"
    reason: str | None = None
    spot: float | None = None
    sigma_ex: float | None = None
    earnings_date: str | None = None
    implied_move: float | None = None
    historical_move: float | None = None
    past_earnings_used: int = 0
    earnings_floor: float | None = None
    rejected: dict = field(default_factory=dict)
    ranked: list[ScoredStrike] = field(default_factory=list)
    # occ_symbol -> options_data.CallCandidate for every ranked strike, so
    # main.py can hand the chosen one straight to trader/logger. Not
    # serialized (see main()'s --json).
    candidates: dict = field(default_factory=dict, repr=False)


# ── I/O-backed advisor ───────────────────────────────────────────────────────

def _atm_iv_by_expiration(candidates, spot: float) -> dict[date, tuple[float, int]]:
    """{expiration: (ATM iv, dte)} from the strike closest to spot per expiration."""
    best = {}
    for c in candidates:
        cur = best.get(c.expiration)
        if cur is None or abs(c.strike - spot) < abs(cur.strike - spot):
            best[c.expiration] = c
    return {exp: (c.iv, c.dte) for exp, c in best.items()}


def pick_strike_by_edge(symbol: str, spot: float, min_strike: float | None = None):
    """
    Live-bot entry point (main.py, STRIKE_SELECTION="edge"): the top-ranked
    strike as an options_data.CallCandidate, or None if nothing qualifies.
    Same contract as options_data.pick_strike()'s return value, so every
    downstream order/log/notify path is unchanged.
    """
    adv = advise(symbol, basis=min_strike, spot=spot)
    if adv.status != "ok":
        _log.warning("%s: edge strike selection found nothing — %s | rejected: %s", symbol, adv.reason, adv.rejected)
        return None
    best = adv.ranked[0]
    _log.info(
        "%s: edge pick %s strike %.2f exp %s | fill ~%.2f fair %.2f edge %.2f (%.1f%%/yr) | delta %.2f%s",
        symbol, best.occ_symbol, best.strike, best.expiration, best.fill, best.fair, best.edge,
        best.edge_ann_pct, best.market_delta,
        f" | THROUGH EARNINGS {adv.earnings_date}, floor {adv.earnings_floor:.2f}" if best.spans_earnings else "",
    )
    return adv.candidates[best.occ_symbol]


def advise(symbol: str, basis: float | None = None,
           dte_min: int = ADVISOR_DTE_MIN, dte_max: int = ADVISOR_DTE_MAX,
           spot: float | None = None) -> SymbolAdvice:
    # Imported here so the pure math above stays importable/testable without
    # constructing Alpaca clients (options_data builds them at import time).
    from backtest import _price_history
    from calendar_data import get_earnings_status, get_past_earnings_dates
    from options_data import _build_candidate, _passes_liquidity_gates, get_chain_snapshot, list_call_contracts
    from screener import _get_spot_price

    adv = SymbolAdvice(symbol=symbol, status="skipped")

    status, earnings_date, ex_div_date = get_earnings_status(symbol)
    if status == "unknown":
        adv.reason = "earnings date could not be determined (fail closed)"
        return adv
    adv.earnings_date = earnings_date.isoformat() if earnings_date else None

    if spot is None:
        spot = _get_spot_price(symbol)
    if spot is None:
        adv.reason = "no live spot price"
        return adv
    adv.spot = spot

    closes = _price_history(symbol, lookback=f"{EARNINGS_HISTORY_YEARS + 1}y")
    today = date.today()
    past = get_past_earnings_dates(symbol, today - timedelta(days=365 * EARNINGS_HISTORY_YEARS), today - timedelta(days=2))
    earn_pos = earnings_positions(closes, past)
    sigma_ex = forecast_vol(clean_log_returns(closes, earn_pos))
    if sigma_ex is None:
        adv.reason = "not enough price history for a vol forecast"
        return adv
    adv.sigma_ex = sigma_ex

    hist_jump, _ = historical_jump(closes, earn_pos, sigma_ex)
    adv.past_earnings_used = len(earn_pos)
    if len(earn_pos) >= EARNINGS_MIN_HISTORY:
        adv.historical_move = hist_jump

    contracts = list_call_contracts(symbol, spot_price=spot, dte_min=dte_min, dte_max=dte_max)
    if not contracts:
        adv.reason = f"no call contracts in {dte_min}-{dte_max} DTE"
        return adv
    snapshots = get_chain_snapshot(symbol)
    built = []
    for c in contracts:
        snap = snapshots.get(c.symbol)
        cand = _build_candidate(c, snap, spot, today, ex_div_date) if snap is not None else None
        if cand is not None:
            built.append(cand)
    adv.rejected["bad_or_wide_quote"] = len(contracts) - len(built)
    if not built:
        adv.reason = "no usable quotes (stale, crossed, or spread too wide on the indicative feed)"
        return adv

    def spans(exp: date) -> bool:
        return earnings_date is not None and exp >= earnings_date

    if earnings_date is not None and any(spans(c.expiration) for c in built):
        atm = _atm_iv_by_expiration(built, spot)
        pre = [(e, v) for e, v in atm.items() if not spans(e)]
        post = [(e, v) for e, v in atm.items() if spans(e)]
        sigma_post, dte_post = min(post)[1]
        sigma_pre = max(pre)[1][0] if pre else sigma_ex
        adv.implied_move = implied_jump(sigma_pre, sigma_post, dte_post / 365.0)
        adv.earnings_floor = earnings_strike_floor(spot, adv.implied_move, adv.historical_move)

    counts = {"illiquid": 0, "delta_out_of_band": 0, "below_basis": 0,
              "below_earnings_floor": 0, "earnings_no_history": 0, "no_edge": 0}
    ranked = []
    for c in built:
        if not _passes_liquidity_gates(c):
            counts["illiquid"] += 1
            continue
        if not (ADVISOR_MIN_DELTA <= c.delta <= ADVISOR_MAX_DELTA):
            counts["delta_out_of_band"] += 1
            continue
        if basis is not None and c.strike < basis:
            counts["below_basis"] += 1
            continue
        through_earnings = spans(c.expiration)
        if through_earnings:
            if adv.historical_move is None:
                counts["earnings_no_history"] += 1
                continue
            if c.strike < adv.earnings_floor:
                counts["below_earnings_floor"] += 1
                continue
        t = c.dte / 365.0
        fair, p_assign = fair_call(spot, c.strike, t, sigma_ex, adv.historical_move if through_earnings else 0.0)
        fill = expected_fill(c.mid_price, c.bid_ask_spread_pct)
        edge = fill - fair
        if edge <= 0:
            counts["no_edge"] += 1
            continue
        ranked.append(ScoredStrike(
            occ_symbol=c.occ_symbol, expiration=c.expiration.isoformat(), dte=c.dte,
            strike=c.strike, mid=c.mid_price, fill=fill, fair=fair, edge=edge,
            edge_ann_pct=edge / spot * 365.0 / c.dte * 100.0,
            premium_ann_pct=fill / spot * 365.0 / c.dte * 100.0,
            market_delta=c.delta, market_iv=c.iv, p_assign=p_assign,
            open_interest=c.open_interest, spread_pct=c.bid_ask_spread_pct,
            spans_earnings=through_earnings,
        ))
        adv.candidates[c.occ_symbol] = c
    adv.rejected.update(counts)
    ranked.sort(key=lambda s: s.edge_ann_pct, reverse=True)
    adv.ranked = ranked
    if not ranked:
        adv.reason = "no strike passed every filter with positive edge"
        return adv
    adv.status = "ok"
    return adv


# ── CLI ──────────────────────────────────────────────────────────────────────

def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def _print(adv: SymbolAdvice, top: int) -> None:
    print(f"\n=== {adv.symbol} ===")
    if adv.spot is not None:
        print(f"spot ${adv.spot:.2f} | forecast vol (ex-earnings) {_pct(adv.sigma_ex)} | "
              f"next earnings {adv.earnings_date or 'none in range'}")
    if adv.earnings_date:
        floor = f"${adv.earnings_floor:.2f}" if adv.earnings_floor else "n/a (no expiration spans it)"
        print(f"earnings move (1 sd): implied {_pct(adv.implied_move)} | historical {_pct(adv.historical_move)} "
              f"from {adv.past_earnings_used} reports | strike floor through earnings: {floor}")
    if adv.rejected:
        print("rejected: " + ", ".join(f"{k}={v}" for k, v in adv.rejected.items() if v))
    if adv.status != "ok":
        print(f"SKIPPED: {adv.reason}")
        return
    print(f"{'expiry':<11}{'dte':>4}{'strike':>9}{'mid':>7}{'fair':>7}{'edge':>7}{'edge/yr':>9}"
          f"{'prem/yr':>9}{'delta':>7}{'IV':>7}{'P(asgn)':>8}{'OI':>7}  earn")
    for s in adv.ranked[:top]:
        print(f"{s.expiration:<11}{s.dte:>4}{s.strike:>9.2f}{s.mid:>7.2f}{s.fair:>7.2f}{s.edge:>7.2f}"
              f"{s.edge_ann_pct:>8.1f}%{s.premium_ann_pct:>8.1f}%{s.market_delta:>7.2f}{s.market_iv * 100:>6.0f}%"
              f"{s.p_assign * 100:>7.0f}%{(s.open_interest or 0):>7}  {'YES' if s.spans_earnings else ''}")
    b = adv.ranked[0]
    print(f"PICK: sell {b.occ_symbol} (${b.strike:.2f} exp {b.expiration}) ~${b.fill:.2f}/sh, "
          f"edge ${b.edge:.2f}/sh vs. fair ${b.fair:.2f}")


def _parse_basis(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        sym, _, price = item.partition("=")
        out[sym.upper()] = float(price)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Rank covered-call strikes by volatility edge (read-only).")
    p.add_argument("symbols", nargs="+")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--dte-min", type=int, default=ADVISOR_DTE_MIN)
    p.add_argument("--dte-max", type=int, default=ADVISOR_DTE_MAX)
    p.add_argument("--basis", action="append", default=[], metavar="SYM=PRICE",
                   help="cost basis floor per symbol (repeatable)")
    p.add_argument("--json", metavar="PATH", help="also write full results as JSON")
    p.add_argument("--allow-stale", action="store_true",
                   help="research only: accept quotes older than QUOTE_MAX_AGE_MINUTES (e.g. outside market hours)")
    args = p.parse_args(argv)

    if args.allow_stale:
        # Only this process's copy of the module constant -- main.py/live
        # trading never runs with this, and this tool never places orders.
        import options_data
        options_data.QUOTE_MAX_AGE_MINUTES = 60 * 24 * 7
        print("WARNING: --allow-stale -- quotes may be from the last session, not live prices")

    basis = _parse_basis(args.basis)
    results = []
    for sym in (s.upper() for s in args.symbols):
        try:
            adv = advise(sym, basis=basis.get(sym), dte_min=args.dte_min, dte_max=args.dte_max)
        except Exception as e:  # one bad symbol never kills the run
            _log.exception("advise(%s) failed", sym)
            adv = SymbolAdvice(symbol=sym, status="skipped", reason=f"error: {e}")
        results.append(adv)
        _print(adv, args.top)

    if args.json:
        with open(args.json, "w") as f:
            json.dump([{k: v for k, v in asdict(a).items() if k != "candidates"} for a in results], f, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
