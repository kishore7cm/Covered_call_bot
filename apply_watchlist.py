"""
Human approval gate between research (screener.py + backtest.py) and the
live trader (main.py). The live trader never reads proposed_watchlist.json
directly — it reads active_watchlist.json, and this script is the only thing
that writes that file. Deliberately a separate, trivial script rather than a
flag on backtest.py, so "backtest produced new numbers" and "I'm approving
these for live trading" are always two distinct, explicit actions.

Usage:
    python apply_watchlist.py                 # show the diff, change nothing
    python apply_watchlist.py --confirm       # show the diff, then apply it
"""
import argparse
import json
import os

PROPOSED_PATH = "pipeline_state/proposed_watchlist.json"
ACTIVE_PATH = "pipeline_state/active_watchlist.json"


def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {"symbols": {}}
    with open(path) as f:
        return json.load(f)


def _diff(active: dict, proposed: dict) -> None:
    active_symbols = active.get("symbols", {})
    proposed_symbols = proposed.get("symbols", {})

    added = sorted(set(proposed_symbols) - set(active_symbols))
    removed = sorted(set(active_symbols) - set(proposed_symbols))
    changed = sorted(
        s for s in set(active_symbols) & set(proposed_symbols)
        if active_symbols[s].get("dte") != proposed_symbols[s].get("dte")
    )

    print(f"Active watchlist generated: {active.get('generated_at', 'never')}")
    print(f"Proposed watchlist generated: {proposed.get('generated_at', 'unknown')}")
    print()

    if added:
        print("ADD:")
        for s in added:
            stats = proposed_symbols[s]
            print(f"  + {s}: DTE={stats['dte']} (avg annualized yield {stats['avg_annualized_yield'] * 100:.1f}%)")
    if removed:
        print("REMOVE:")
        for s in removed:
            print(f"  - {s}")
    if changed:
        print("CHANGE DTE:")
        for s in changed:
            print(f"  ~ {s}: {active_symbols[s]['dte']} -> {proposed_symbols[s]['dte']}")
    if not (added or removed or changed):
        print("No differences — active watchlist already matches the proposed one.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="apply the proposed watchlist as active")
    parser.add_argument("--proposed", default=PROPOSED_PATH)
    parser.add_argument("--active", default=ACTIVE_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.proposed):
        print(f"No proposed watchlist at {args.proposed} — run backtest.py --from-screener first.")
        raise SystemExit(1)

    active = _load(args.active)
    proposed = _load(args.proposed)
    _diff(active, proposed)

    if not args.confirm:
        print("\n(dry run — pass --confirm to apply this to the live trader's watchlist)")
    else:
        os.makedirs(os.path.dirname(args.active), exist_ok=True)
        with open(args.active, "w") as f:
            json.dump(proposed, f, indent=2)
        print(f"\nApplied — {args.active} updated. main.py will pick this up on its next cycle.")
