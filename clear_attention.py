"""
Human-operated clear for pipeline_state/needs_attention.json -- the only
sanctioned way to remove a symbol's flag (trader.mark_needs_attention is
deliberately one-directional, see its docstring) without hand-editing the
JSON file directly. Every clear is logged (symbol, the flag that was
cleared, and the reason given) via logger.log_error, so there's a durable
record of who cleared what and why, rather than an entry silently
disappearing from the file with no trace it ever happened.

Usage:
    python clear_attention.py AAPL "confirmed broker position matches, topped up manually"
"""
import argparse
import sys

import trader
from logger import log_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol")
    parser.add_argument("reason", help="why this is safe to clear -- logged, not optional")
    args = parser.parse_args(argv)

    existing = trader.check_needs_attention(args.symbol)
    if existing is None:
        print(f"{args.symbol} has no needs-attention flag set -- nothing to clear.")
        return 1

    trader.clear_needs_attention(args.symbol, args.reason)
    log_error(
        "clear_attention",
        f"{args.symbol}: cleared needs-attention (was: {existing!r}) -- reason: {args.reason}",
    )
    print(f"{args.symbol}: cleared (was: {existing!r})")
    print(f"Reason logged: {args.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
