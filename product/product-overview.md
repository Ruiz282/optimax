# Options Strategy Recommender (OptiMax)

## The Problem

A trader who understands market dynamics (regime shifts, concentration changes, creative destruction) still faces a gap when translating that understanding into options trades. The questions:

- "I'm bullish on NVDA — do I buy calls, sell puts, or run a spread?"
- "IV is elevated after earnings — how do I profit from that?"
- "The market is concentrating into AI stocks — what options strategies capture that?"
- "How much should I risk on this trade?"

Existing tools either assume advanced knowledge (thinkorswim, tastytrade) or lack the analytical depth to connect market signals to strategy selection.

## Success Looks Like

1. Enter a ticker → see live options chain enriched with computed Greeks
2. Input your market view (direction + conviction + timeframe) → get a ranked list of recommended strategies with payoff diagrams
3. See IV analysis: is vol high or low vs. history? → strategies adapt accordingly
4. Connect entropy/regime signals from the parent repo → "regime shift detected, here's how to position"
5. Position sizing guidance → "risk no more than X on this trade"
6. Beginner-friendly explanations → every recommendation explains the "why"

## Building On (Existing Foundations)

- **Parent repo (Case 2: ChatGPT Launch)** — Entropy calculations, concentration ratios, sample space expansion detection, creative destruction metrics. These become input signals.
- **yfinance** — Free options chain data (strikes, expiries, IV, bid/ask, volume, open interest). 15-min delayed but sufficient for strategy selection.
- **scipy.stats.norm + numpy** — Self-contained Black-Scholes pricing and Greeks computation. No extra dependencies. Vectorized for speed.
- **Streamlit** — Interactive Python UI. See results immediately. Iterate fast.
- **matplotlib** — Payoff diagrams, IV surface visualization, historical vol charts.

## The Unique Part

1. **Entropy-to-Strategy Bridge** — No tool connects information-theoretic market signals to options strategy selection. When entropy drops (concentration increases), recommend strategies that benefit from momentum/trend. When entropy rises (uncertainty), recommend strategies that benefit from volatility expansion.

2. **Beginner Decision Engine** — A simple input (bullish/bearish/neutral × high IV/low IV × timeframe) produces a ranked strategy recommendation with clear risk/reward, max loss, and break-even — not just "buy a call" but "buy the $150 call expiring March 21 at $5.20, max loss $520, break-even $155.20."

3. **IV Percentile Context** — Shows where current IV sits vs. 30/60/90-day history. High IV = sell premium strategies. Low IV = buy premium strategies. This single insight drives most profitable options decisions for beginners.

## Tech Stack

- **UI:** Streamlit (interactive, Python-native, zero frontend complexity)
- **Data:** yfinance (free — options chains, historical prices, IV)
- **Greeks:** Self-contained Black-Scholes (scipy.stats.norm + numpy)
- **Visualization:** matplotlib (payoff diagrams, IV charts, surface plots)
- **Parent Repo Integration:** Import entropy/concentration functions from src/

## Key Strategies Covered

| Strategy | When Recommended | Risk Profile |
|----------|-----------------|--------------|
| Long Call | Bullish + Low IV | Defined risk, unlimited reward |
| Long Put | Bearish + Low IV | Defined risk, large reward |
| Bull Call Spread | Bullish + High IV | Defined risk, defined reward |
| Bear Put Spread | Bearish + High IV | Defined risk, defined reward |
| Cash-Secured Put | Bullish + High IV | Obligation to buy shares |
| Covered Call | Neutral/Mildly Bullish | Reduces cost basis |
| Iron Condor | Neutral + High IV | Defined risk, profits from time decay |
| Straddle | Uncertain direction + Low IV | Profits from big move either way |
| Strangle | Uncertain direction + Low IV | Cheaper than straddle, wider break-evens |
| Calendar Spread | Neutral + IV term structure play | Profits from near-term decay |

## Open Questions

- Should we add paper-trading simulation (track P&L of recommended trades over time)?
- How deep should the entropy signal integration go in v1 vs. later iterations?
- Add support for scanning multiple tickers for opportunities, or single-ticker focus first?
