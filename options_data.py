"""
Alpaca option contract listing + chain snapshots, and strike selection by
computed delta (see greeks.py for why delta is computed rather than pulled
from Alpaca directly).
"""
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest

from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER,
    DTE_MIN, DTE_MAX, RISK_FREE_RATE, DELTA_TOLERANCE,
    MIN_OPEN_INTEREST, MAX_SPREAD_PCT, REQUIRE_EXPIRY_BEFORE_EARNINGS,
    QUOTE_MAX_AGE_MINUTES,
)
from calendar_data import get_earnings_status
from greeks import implied_vol, bs_call_delta, bs_call_theta_per_day

_log = logging.getLogger(__name__)

_trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)
_data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


@dataclass
class CallCandidate:
    occ_symbol: str
    strike: float
    expiration: date
    mid_price: float
    delta: float
    dte: int
    iv: float

    # Filter 3 contract math — all derived from the fields above + spot_price
    # at construction time. Formulas per the standard covered-call (buy-write)
    # convention: breakeven assumes you bought the stock at spot_price and
    # received this premium against it.
    premium_yield: float        # premium / stock_price
    annualized_yield: float     # premium_yield * (365 / DTE)
    breakeven: float            # stock_price - premium
    # NOTE: with breakeven defined as stock_price - premium, downside_cushion
    # reduces algebraically to exactly premium_yield ((stock_price - breakeven)
    # / stock_price == premium / stock_price). Kept as a separate field anyway
    # since that's how the metric was specified — but don't expect it to ever
    # read differently from premium_yield for a plain buy-write.
    downside_cushion: float     # (stock_price - breakeven) / stock_price
    if_called_return: float     # (strike - breakeven) / breakeven
    theta_per_day: float        # SHORT-call convention: positive = time decay earns the seller money/day

    # Filter 2 liquidity/timing fields.
    open_interest: int | None = None
    bid_ask_spread_pct: float | None = None
    # True if a known ex-dividend date falls within [today, expiration] --
    # early-assignment risk on an ITM short call. This is a FLAG, not an
    # exclusion: pick_strike does not drop candidates for this alone, the
    # caller (main.py) surfaces it (Telegram/logs) and lets the human
    # decide. ex_dividend_date is carried alongside it (not just the bool)
    # so a caller can report the actual date/days-until without having to
    # re-derive it from calendar_data a second time.
    ex_div_before_expiration: bool = False
    ex_dividend_date: date | None = None


@dataclass
class ParsedOcc:
    root: str
    expiration: date
    option_type: str  # "C" or "P"
    strike: float


def parse_occ_symbol(occ_symbol: str) -> ParsedOcc:
    """
    Parse an OCC option symbol: <ROOT><YYMMDD><C|P><strike*1000, 8 digits>,
    e.g. "AAPL261113C00360000" -> root=AAPL, exp=2026-11-13, type=C, strike=360.0.
    Alpaca positions/orders carry this as the only identifier for an option
    leg — there's no separate strike/expiration field on a Position object,
    so anything needing those for an already-open position must parse it
    out of the symbol rather than guess at other fields.
    """
    # last 15 chars are fixed-width: YYMMDD(6) + C/P(1) + strike(8)
    date_type_strike = occ_symbol[-15:]
    root = occ_symbol[:-15]

    yymmdd = date_type_strike[:6]
    option_type = date_type_strike[6]
    strike_raw = date_type_strike[7:]

    year = 2000 + int(yymmdd[0:2])
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    expiration = date(year, month, day)
    strike = int(strike_raw) / 1000.0

    return ParsedOcc(root=root, expiration=expiration, option_type=option_type, strike=strike)


def list_call_contracts(
    underlying: str,
    spot_price: float | None = None,
    dte_min: int = DTE_MIN,
    dte_max: int = DTE_MAX,
):
    """
    Fetch tradeable call contracts for `underlying` expiring in [dte_min, dte_max] days.

    Alpaca caps a single page at 100 contracts by default. For a wide chain
    (e.g. SPY, which has hundreds of strikes per expiration) that silently
    returns only the lowest ~100 strikes — all deep ITM, nowhere near a
    useful OTM covered-call strike. Confirmed live: an unbounded SPY query
    returned strikes 500-771 while spot was ~778, missing every strike a
    0.20-0.35 delta call would actually use. Two mitigations:
      1. `limit=1000` — well above what one expiration's chain needs in
         practice (confirmed up to 539 contracts return in a single page).
      2. `strike_price_gte` when spot_price is known, to skip deep-ITM
         strikes a covered call would never target — also cuts the amount
         of data pulled/scored for no benefit.
    Pagination via next_page_token is still handled defensively in case a
    chain ever exceeds the limit, capped at a few pages to avoid a runaway
    loop against a misbehaving response.
    """
    today = date.today()
    kwargs = dict(
        underlying_symbols=[underlying],
        status=AssetStatus.ACTIVE,
        type=ContractType.CALL,
        expiration_date_gte=today + timedelta(days=dte_min),
        expiration_date_lte=today + timedelta(days=dte_max),
        limit=1000,
    )
    if spot_price is not None:
        # Covered calls are sold OTM/near-the-money; allow some headroom
        # below spot for higher-delta configs, but skip strikes far enough
        # ITM that no reasonable target_delta would ever select them.
        kwargs["strike_price_gte"] = str(round(spot_price * 0.85, 2))

    all_contracts = []
    page_token = None
    for _ in range(5):  # defensive cap on pagination loops
        req = GetOptionContractsRequest(**kwargs, page_token=page_token)
        resp = _trading_client.get_option_contracts(req)
        all_contracts.extend(resp.option_contracts)
        if not resp.next_page_token:
            break
        page_token = resp.next_page_token

    return all_contracts


def get_chain_snapshot(underlying: str) -> dict:
    """OCC symbol -> snapshot (quote/trade/bars), via the free indicative feed."""
    req = OptionChainRequest(underlying_symbol=underlying)
    return _data_client.get_option_chain(req)


def _quote(snapshot) -> tuple[float, float] | None:
    """
    Return (bid, ask) if the quote is usable, else None. Rejects:
      - missing/zero sides (thin/no market)
      - crossed quotes (bid > ask) -- an inverted market is never a valid
        trade price, no matter how it got that way. This used to slip
        through into _build_candidate, where spread_pct = (ask-bid)/mid
        came out NEGATIVE and therefore always passed the MAX_SPREAD_PCT
        gate (a negative number is never greater than a positive
        threshold) -- rejecting it here, before mid_price is ever
        computed from it, closes that gap at the source.
      - stale quotes (older than QUOTE_MAX_AGE_MINUTES)
      - too-wide spreads (over MAX_SPREAD_PCT of the mid) -- checked here,
        not just in the post-hoc liquidity gate, so get_atm_iv() (which
        never runs that gate) gets the same protection as pick_strike().
    """
    q = getattr(snapshot, "latest_quote", None)
    if q is None or q.bid_price is None or q.ask_price is None:
        return None
    if q.bid_price <= 0 or q.ask_price <= 0:
        return None
    if q.bid_price > q.ask_price:
        _log.warning("Crossed quote (bid %.2f > ask %.2f) for %s — rejecting", q.bid_price, q.ask_price, q.symbol)
        return None

    timestamp = getattr(q, "timestamp", None)
    if timestamp is not None:
        age = datetime.now(timezone.utc) - timestamp
        if age > timedelta(minutes=QUOTE_MAX_AGE_MINUTES):
            _log.warning("Stale quote (%s old) for %s — rejecting", age, q.symbol)
            return None

    mid = (q.bid_price + q.ask_price) / 2.0
    spread_pct = (q.ask_price - q.bid_price) / mid if mid > 0 else None
    if spread_pct is not None and spread_pct > MAX_SPREAD_PCT:
        _log.warning("Spread %.1f%% exceeds MAX_SPREAD_PCT for %s — rejecting", spread_pct * 100, q.symbol)
        return None

    return (q.bid_price, q.ask_price)


def _build_candidate(
    contract, snapshot, spot_price: float, today: date,
    ex_dividend_date: date | None = None,
) -> CallCandidate | None:
    """Build a fully-populated CallCandidate from a contract + its snapshot, or
    None if the quote/IV isn't solvable (thin/stale market)."""
    quote = _quote(snapshot)
    if quote is None:
        return None
    bid, ask = quote
    mid = (bid + ask) / 2.0

    exp = contract.expiration_date
    dte = (exp - today).days
    if dte <= 0:
        return None
    t_years = dte / 365.0

    strike = float(contract.strike_price)
    iv = implied_vol(mid, spot_price, strike, t_years, RISK_FREE_RATE)
    if iv is None:
        return None
    delta = bs_call_delta(spot_price, strike, t_years, RISK_FREE_RATE, iv)
    # bs_call_theta_per_day returns the standard LONG-call theta (negative —
    # a long call loses value to time decay). This bot only ever holds the
    # SHORT side of the call, so flip the sign here, once, at the source —
    # every downstream consumer of CallCandidate.theta_per_day can then
    # assume "positive = good for us" like every other field on this class.
    short_theta_per_day = -bs_call_theta_per_day(spot_price, strike, t_years, RISK_FREE_RATE, iv)

    premium_yield = mid / spot_price
    annualized_yield = premium_yield * (365.0 / dte)
    breakeven = spot_price - mid
    downside_cushion = (spot_price - breakeven) / spot_price
    if_called_return = (strike - breakeven) / breakeven if breakeven > 0 else float("nan")

    spread_pct = (ask - bid) / mid if mid > 0 else None

    # Alpaca types open_interest as Optional[str] (confirmed against the live
    # SDK model), not int, despite being a count — cast explicitly rather
    # than storing a string in a field declared `int | None`.
    oi = int(contract.open_interest) if contract.open_interest is not None else None

    # A past ex-div date (yfinance returns the MOST RECENT one when the
    # next hasn't been officially scheduled yet -- live-confirmed for
    # AAPL, whose Ex-Dividend Date comes back months in the past) is not
    # upcoming risk at all; comparing it against a future expiration with
    # no lower bound used to flag every single candidate as a false
    # positive. Informational only either way -- never excludes a
    # candidate, never affects which strike gets picked.
    ex_div_flag = ex_dividend_date is not None and today <= ex_dividend_date <= exp

    return CallCandidate(
        occ_symbol=contract.symbol,
        strike=strike,
        expiration=exp,
        mid_price=mid,
        delta=delta,
        dte=dte,
        iv=iv,
        premium_yield=premium_yield,
        annualized_yield=annualized_yield,
        breakeven=breakeven,
        downside_cushion=downside_cushion,
        if_called_return=if_called_return,
        theta_per_day=short_theta_per_day,
        open_interest=oi,
        bid_ask_spread_pct=spread_pct,
        ex_div_before_expiration=ex_div_flag,
        ex_dividend_date=ex_dividend_date,
    )


def _passes_liquidity_gates(c: CallCandidate) -> bool:
    if c.open_interest is None or c.open_interest < MIN_OPEN_INTEREST:
        return False
    if c.bid_ask_spread_pct is None or c.bid_ask_spread_pct > MAX_SPREAD_PCT:
        return False
    return True


def pick_strike(
    underlying: str,
    spot_price: float,
    target_delta: float,
    tolerance: float = DELTA_TOLERANCE,
    dte_min: int = DTE_MIN,
    dte_max: int = DTE_MAX,
    min_strike: float | None = None,
) -> CallCandidate | None:
    """
    Screen call contracts for `underlying`, apply Filter 2 (liquidity +
    earnings/dividend timing) and Filter 3 (contract math), then return the
    surviving candidate closest to `target_delta`. None if nothing in the
    chain survives (thin liquidity, no quote, earnings inside the window,
    or nothing near the target delta).

    dte_min/dte_max default to the global config window but can be
    overridden per call — main.py passes a per-symbol window derived from
    backtest.py's recommended DTE once the research pipeline has run.

    min_strike (Phase 3, Task 6): when given, drops any candidate with
    strike < min_strike before delta-matching -- main.py passes the open
    lot's own cost basis here, so a covered call is never sold below what
    the shares actually cost (selling below basis locks in a loss on
    assignment even if the premium looked attractive).

    Earnings risk is resolved via calendar_data.get_earnings_status(),
    which fails CLOSED: "no earnings expected" (a known ETF) and "a
    confirmed date was found" are the only two statuses that let this
    continue. "unknown" (lookup failed, the ETF cross-check disagreed, or
    no usable date could be resolved) skips the symbol entirely for this
    cycle -- it is NOT treated as "no earnings risk," which is what this
    function used to do whenever the old get_calendar_dates() lookup
    simply came back empty for a stock.
    """
    earnings_status, earnings_date, ex_div_date = get_earnings_status(underlying)
    if earnings_status == "unknown":
        _log.warning("%s: could not determine earnings risk (lookup failed or signals disagree) — skipping", underlying)
        return None

    contracts = list_call_contracts(underlying, spot_price=spot_price, dte_min=dte_min, dte_max=dte_max)
    if not contracts:
        _log.warning("No call contracts found for %s in DTE window", underlying)
        return None

    if earnings_date is not None and REQUIRE_EXPIRY_BEFORE_EARNINGS:
        before = len(contracts)
        contracts = [c for c in contracts if c.expiration_date < earnings_date]
        _log.info(
            "%s: earnings on %s — dropped %d/%d contracts expiring on/after it",
            underlying, earnings_date, before - len(contracts), before,
        )
        if not contracts:
            _log.warning("%s: no contracts expire before earnings (%s) — skipping", underlying, earnings_date)
            return None

    snapshots = get_chain_snapshot(underlying)
    today = date.today()

    candidates: list[CallCandidate] = []
    for c in contracts:
        snap = snapshots.get(c.symbol)
        if snap is None:
            continue
        candidate = _build_candidate(c, snap, spot_price, today, ex_div_date)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        _log.warning("No solvable-delta candidates for %s — thin chain or stale quotes", underlying)
        return None

    liquid = [c for c in candidates if _passes_liquidity_gates(c)]
    if not liquid:
        _log.warning(
            "%s: %d candidates found but none pass liquidity gates (OI>=%d, spread<=%.0f%%)",
            underlying, len(candidates), MIN_OPEN_INTEREST, MAX_SPREAD_PCT * 100,
        )
        return None

    if min_strike is not None:
        before = len(liquid)
        liquid = [c for c in liquid if c.strike >= min_strike]
        if not liquid:
            _log.warning(
                "%s: no qualifying call at or above basis $%.2f (%d candidate(s) were all below it)",
                underlying, min_strike, before,
            )
            return None

    best = min(liquid, key=lambda x: abs(x.delta - target_delta))

    if abs(best.delta - target_delta) > tolerance:
        _log.warning(
            "%s: closest delta %.2f is outside tolerance of target %.2f (+/-%.2f) — skipping",
            underlying, best.delta, target_delta, tolerance,
        )
        return None

    if best.ex_div_before_expiration:
        _log.warning(
            "%s: selected contract %s has ex-dividend before expiration — early assignment risk",
            underlying, best.occ_symbol,
        )

    return best


# How far either side of target_dte get_atm_iv() searches when target_dte is
# given — wide enough to usually contain a real expiration near the target,
# narrow enough that "nearest expiration" doesn't drift to something only
# loosely related to the requested tenor.
ATM_IV_DTE_SEARCH_PAD_DAYS = 10


def get_atm_iv(underlying: str, spot_price: float, target_dte: int | None = None) -> float | None:
    """
    Solve IV for a call contract with strike closest to spot_price. Used for
    IV-rank/percentile tracking (see iv_rank.py) — intentionally independent
    of pick_strike()'s delta/liquidity gates, since IV history should reflect
    the market's general level, not just whatever happened to pass this
    cycle's filters.

    target_dte=None (default — what main.py's weekly cycle already uses):
    picks nearest-strike-to-spot across every expiration in the standard
    DTE_MIN..DTE_MAX window, which can mix expirations from one call to the
    next.

    target_dte=N: first narrows to the single expiration closest to N days
    out (searching +/- ATM_IV_DTE_SEARCH_PAD_DAYS), then picks nearest strike
    within just that expiration — gives a consistent, comparable "ATM IV at
    ~N DTE" reading day over day, which iv_logger.py needs for a clean daily
    series. Existing callers that don't pass target_dte are unaffected.
    """
    today = date.today()

    if target_dte is not None:
        dte_min = max(1, target_dte - ATM_IV_DTE_SEARCH_PAD_DAYS)
        dte_max = target_dte + ATM_IV_DTE_SEARCH_PAD_DAYS
        contracts = list_call_contracts(underlying, spot_price=spot_price, dte_min=dte_min, dte_max=dte_max)
    else:
        contracts = list_call_contracts(underlying, spot_price=spot_price)

    if not contracts:
        return None

    if target_dte is not None:
        nearest_expiration = min(
            contracts, key=lambda c: abs((c.expiration_date - today).days - target_dte)
        ).expiration_date
        contracts = [c for c in contracts if c.expiration_date == nearest_expiration]

    snapshots = get_chain_snapshot(underlying)

    nearest = min(contracts, key=lambda c: abs(float(c.strike_price) - spot_price))
    snap = snapshots.get(nearest.symbol)
    if snap is None:
        return None

    candidate = _build_candidate(nearest, snap, spot_price, today)
    return candidate.iv if candidate else None
