# Roadmap

Building on: yfinance (options data), scipy/numpy (BS Greeks), Streamlit (UI), parent repo entropy functions

## Sections

### 1. Options Data Engine
Fetch live options chains via yfinance, compute all Greeks with self-contained Black-Scholes, display an enriched interactive chain explorer in Streamlit.

### 2. IV Analyzer & Strategy Recommender
Calculate IV percentile vs. 30/60/90-day history, implement the decision engine (outlook x IV environment x timeframe → ranked strategy list), render payoff diagrams for each recommendation.

### 3. Entropy Signal Integration
Import sector_entropy and concentration functions from parent repo, compute current market entropy vs. historical baseline, overlay regime signal onto strategy recommendations (rising entropy → long gamma, falling entropy → sell premium).

### 4. Position Sizer & Trade Card
Apply 2% portfolio risk rule and simplified Kelly criterion, generate a complete trade card for each recommendation (exact contracts, entry price, max loss, max profit, break-evens, profit target, stop-loss exit rules).
