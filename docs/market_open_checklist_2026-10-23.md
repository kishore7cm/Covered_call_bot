# Market-open checklist — Fri 2026-10-23

Start at **7:00 AM PT / 10:00 ET or later**. The market opens at 6:30 PT, and the
bot blocks trading for the first 30 minutes.
`DRY_RUN` is not set in `.env`, so it defaults to **true**: nothing below places an
order, not even a paper one.

All commands run from `~/Documents/covered_call_bot`.

## Before the open (any time now)

- [ ] Reply to the other Claude session (the one prepping the Stage 3 tests):
  - Delete the 3 test error rows it left in `ccbot_dev`? → **yes**
  - Delete the hand-written `pipeline_state/active_watchlist.json` (KO)? → **yes**
  - Commit the test-only changes? → your call

## After 7:00 AM PT

### 1. Research run (Scout → Sizer → Foreman → Oracle, read-only)

```
python3 orchestrator.py research SOFI F T PFE BAC
```

- [ ] Check which symbols passed the screen. Many get dropped because Alpaca's free
      data feed shows wide spreads.
- [ ] Read the proposed diff it prints: DTE, warnings, lot cost against the cap.

### 2. Approve the watchlist (the only step that changes what the bot trades)

```
python3 apply_watchlist.py --confirm
```

- [ ] Confirm `pipeline_state/active_watchlist.json` now lists the approved symbols,
      with today's `confirmed_at`.

### 3. Dry-run one bot cycle

```
python3 main.py --once
```

What to expect for each symbol:
- [ ] Skipped if it's on the other-bots list or flagged for attention
- [ ] No shares held → it decides to buy 100 shares → every safety check runs → a
      "DRY_RUN — would have…" line is logged, and no order is sent
- [ ] There is no real "sell call" step, because no shares were actually bought
- [ ] No tracebacks or errors in the output

- [ ] Tell the other Claude session the step 2 approval and this run are done,
      so it can rerun its A1/A3 checks (A4 needs market hours too).

### 4. Strike advisor on live quotes (read-only)

```
python3 strike_advisor.py SOFI F T PFE BAC --top 5
```

- [ ] Pick 2–3 contracts from its output and check them in Robinhood:
  - mid price ≈ the midpoint of Robinhood's bid/ask?
  - delta in the right range? (Robinhood: chance of profit ≈ 1 − delta, roughly)
- [ ] For any symbol with earnings in the window (SOFI reports Oct 27): does the
      "strike floor through earnings" look sensible next to the strikes Robinhood
      shows?
- [ ] Note anything that looks off, and paste the output back to Claude

## Not today

- Connecting the strike advisor to the live bot (6–34 DTE, volatility edge,
  earnings-aware strikes) and replacing the "roll at 7 days left" rule with the
  danger-only rule. This comes after step 4 checks out.
- `DRY_RUN=false` (real paper orders): only after the dry runs look right, and
  only when you decide.
