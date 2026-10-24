"""
Human-operated remediation for a lot that landed short of SHARES_PER_LOT
after a partial fill (trader._walk_limit_order reports the real filled
quantity rather than retrying indefinitely -- see its own docstring).
Buys exactly enough additional shares to reach the target, through the
SAME buy_shares()/_guard() path every other order takes (kill switch,
market hours, hard caps all still apply), then logs the trade and blends
the top-up into the EXISTING lot's cost basis (logger.add_to_lot) rather
than opening a second lot row for the same symbol -- which would violate
the "one open lot per symbol" invariant get_open_lot()'s own callers rely
on.

Refuses outright if there's no open lot on record for the symbol (this
tool is for completing an existing partial lot, not opening a new one),
or if the requested top-up would push the lot over SHARES_PER_LOT.

Usage:
    python top_up_partial_lot.py AMZN 4
"""
import argparse
import sys

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

import trader
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, SHARES_PER_LOT
from logger import DBUnavailable, add_to_lot, get_open_lot, log_error, log_share_trade

_stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _get_spot_price(symbol: str) -> float | None:
    try:
        trade = _stock_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(trade[symbol].price)
    except Exception as e:
        print(f"{symbol}: failed to get a live spot price ({e})")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol")
    parser.add_argument("additional_qty", type=int)
    args = parser.parse_args(argv)

    try:
        lot = get_open_lot(args.symbol)
    except DBUnavailable as e:
        print(f"{args.symbol}: lot lookup unavailable ({e}) -- refusing to top up blind")
        return 1
    if lot is None:
        print(f"{args.symbol}: no open lot on record -- this tool is for topping up an EXISTING partial lot only")
        return 1

    current_qty = int(lot["qty"])
    target = current_qty + args.additional_qty
    if target > SHARES_PER_LOT:
        print(
            f"{args.symbol}: {current_qty} + {args.additional_qty} = {target}, over SHARES_PER_LOT "
            f"({SHARES_PER_LOT}) -- refusing"
        )
        return 1

    # estimated_price is required for buy_shares's hard-cap check to be
    # enforceable at all -- omitting it fails the cap closed, not open
    # (see trader.buy_shares's own docstring).
    spot = _get_spot_price(args.symbol)
    if spot is None:
        return 1

    # client_order_id_suffix="topup": today's date-based ID was already used
    # by the original (partially-filled) buy -- live-confirmed Alpaca
    # rejects a reused client_order_id even once that order reached a
    # terminal CANCELED state, so this needs a distinct one.
    result = trader.buy_shares(
        args.symbol, qty=args.additional_qty, estimated_price=spot, client_order_id_suffix="topup",
    )
    if result is None:
        print(f"{args.symbol}: buy_shares failed -- see logs")
        return 1
    if not result.get("submitted", True):
        print(f"{args.symbol}: not submitted -- {result.get('block_reason')}")
        return 0

    filled_qty = result["qty"]
    fill_price = result["price"]
    if filled_qty != args.additional_qty:
        print(
            f"{args.symbol}: requested {args.additional_qty} more but only {filled_qty} filled -- "
            f"lot is now {current_qty + filled_qty}/{SHARES_PER_LOT}, may need another top-up"
        )

    try:
        log_share_trade(args.symbol, "BUY", filled_qty, price=fill_price, order_id=result.get("order_id"))
        add_to_lot(args.symbol, filled_qty, fill_price)
    except DBUnavailable as e:
        msg = (
            f"{args.symbol}: bought {filled_qty} more shares for real (order {result.get('order_id')}) "
            f"but failed to log/blend into the lot: {e} -- MUST reconcile the lots table by hand"
        )
        log_error("top_up_partial_lot", msg)
        print(msg)
        return 1

    print(
        f"{args.symbol}: topped up by {filled_qty} shares @ ${fill_price:.2f} -- "
        f"lot is now {current_qty + filled_qty}/{SHARES_PER_LOT}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
