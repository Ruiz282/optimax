"""
Case 2: ChatGPT Launch 2022
MGMT 69000: Mastering AI for Finance

Theme: Sample Space Expansion — When New Asset Classes Enter
Question: When does a new game begin?

DRIVER Framework Applied: This template follows the six stages
with explicit checkpoints, loops, and validation criteria.
"""

# ============================================================
# DISCOVER & DEFINE
# ============================================================
"""
═══════════════════════════════════════════════════════════════
DEFINE FIRST (Provisional)
═══════════════════════════════════════════════════════════════

OBJECTIVE (One Sentence):
    Map how ChatGPT launch (Nov 30, 2022) caused sample space
    expansion — when the investment universe itself changed.

WHAT DOES "DONE" LOOK LIKE?
    □ Disruption cascade mapped (trigger → direct → 2nd → 3rd order)
    □ Returns calculated for winners (NVDA) and losers (CHGG)
    □ Sector entropy quantified before/after ChatGPT
    □ Can explain WHY this is "sample space expansion" vs "regime shift"

SUCCESS CRITERIA:
    1. Divergence between winners/losers exceeds 500%
    2. Entropy metrics show concentration increased
    3. Cascade logic is sound and defensible
    4. Can answer: "What changed — P or X?"

HOW WILL I KNOW IF I'M WRONG?
    - If winners/losers don't diverge significantly → not creative destruction
    - If entropy increases → market diversified, not concentrated
    - If no new allocation category emerged → just repricing, not expansion

═══════════════════════════════════════════════════════════════
DISCOVER (After Provisional Definition)
═══════════════════════════════════════════════════════════════

RESOURCES AVAILABLE:
    □ Stock price data: yfinance (free, no API key)
    □ Tickers: NVDA, CHGG, MSFT, SPY, QQQ, sector ETFs
    □ Timeline: Nov 30, 2022 (launch), May 2, 2023 (Chegg), May 24, 2023 (Nvidia)
    □ Sector weights: S&P 500 public data

CONSTRAINTS:
    □ Historical data only (Nov 2022 - Dec 2024)
    □ Publicly traded companies only
    □ No real-time trading — analysis only
    □ yfinance rate limits (cache data locally)

FOUNDATION I ALREADY HAVE:
    □ Week 1: Regime shift detection (P changed)
    □ Shannon entropy: H = -Σ p(x) * log2(p(x))
    □ Understanding that lower entropy = more concentrated

KNOWLEDGE GAPS TO FILL:
    □ How is "sample space expansion" different from "regime shift"?
    □ What makes an event become a "new asset class"?
    □ How do we detect expansion in real-time?

═══════════════════════════════════════════════════════════════
DEFINE (Refined After Discovery)
═══════════════════════════════════════════════════════════════

REFINED OBJECTIVE:
    Prove that ChatGPT launch caused "sample space expansion" —
    the investment UNIVERSE changed (X₁ → X₂), not just probabilities
    within the existing universe (P₁ → P₂).

    Week 1: P changed (tariff shock — same assets, new dynamics)
    Week 3: X changed (ChatGPT — new asset class entered)

IN SCOPE:
    ✓ Nvidia vs Chegg divergence
    ✓ Sector entropy analysis
    ✓ Cascade mapping
    ✓ The paradox: expansion + concentration

OUT OF SCOPE:
    ✗ Real-time trading signals
    ✗ Full Mag 7 individual analysis
    ✗ Sentiment analysis of earnings calls
    ✗ Prediction of next paradigm shift
"""

# ============================================================
# REPRESENT
# ============================================================
"""
═══════════════════════════════════════════════════════════════
WORKFLOW DIAGRAM (If you can't draw it, you don't understand it)
═══════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────┐
    │                    INPUT DATA                           │
    │  • Stock prices: NVDA, CHGG, SPY (yfinance)            │
    │  • Sector weights: Nov 2022 vs Nov 2024                │
    │  • Timeline: Key dates                                  │
    └──────────────────────┬──────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────────┐
    │              STEP 1: MAP DISRUPTION CASCADE             │
    │  • Trigger: ChatGPT launch (Nov 30, 2022)              │
    │  • Direct effects: Winners (NVDA) + Losers (CHGG)      │
    │  • Second-order: Data centers, energy, education        │
    │  • Third-order: REITs, utilities, curriculum            │
    └──────────────────────┬──────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────────┐
    │              STEP 2: CALCULATE RETURNS                  │
    │  • Normalize prices to 100 at ChatGPT launch           │
    │  • Calculate total return, annualized, volatility       │
    │  • Compare: NVDA (+700%?) vs CHGG (-99%?)              │
    │                                                         │
    │  ⚠️ CHECKPOINT: Divergence > 500%?                      │
    │     If NO → May not be creative destruction             │
    └──────────────────────┬──────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────────┐
    │              STEP 3: ENTROPY ANALYSIS                   │
    │  • Sector weights Nov 2022 → Nov 2024                  │
    │  • Calculate Shannon entropy both periods              │
    │  • Calculate Mag 7 concentration change                 │
    │                                                         │
    │  ⚠️ CHECKPOINT: Entropy decreased?                      │
    │     If NO → Market diversified, not concentrated        │
    └──────────────────────┬──────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────────┐
    │              STEP 4: VISUALIZE & INTERPRET              │
    │  • Winners vs losers chart (log scale)                 │
    │  • Cascade network diagram                              │
    │  • Entropy before/after comparison                      │
    └──────────────────────┬──────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    OUTPUT                               │
    │  • Disruption cascade visualization                     │
    │  • Returns comparison table                             │
    │  • Entropy analysis results                             │
    │  • Answer: "Sample space expanded (X changed)"          │
    └─────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════
CHECKPOINTS (Where to validate before continuing)
═══════════════════════════════════════════════════════════════

    [1] After data fetch: Do we have data for all tickers?
    [2] After returns calc: Is divergence significant (>500%)?
    [3] After entropy calc: Did entropy decrease (concentration)?
    [4] After cascade map: Does the logic make sense?

═══════════════════════════════════════════════════════════════
THE R-I LOOP: Expect to loop back
═══════════════════════════════════════════════════════════════

    Your plan WILL be wrong in some way. That's fine.

    Common loops for this case:
    • Data issue → Loop back: add data cleaning step
    • Divergence not significant → Loop back: check date range
    • Entropy calculation unclear → Loop back: refine weights
    • Cascade logic gaps → Loop back: research second-order effects

    THIS IS THE PROCESS WORKING CORRECTLY.
"""

# ============================================================
# IMPLEMENT
# ============================================================
"""
═══════════════════════════════════════════════════════════════
OWNERSHIP TEST: Before using ANY code (yours or AI-generated)
═══════════════════════════════════════════════════════════════

    For EVERY function below, ask yourself:
    □ Can I explain what this does? (Line by line if needed)
    □ Can I explain WHY this approach over alternatives?
    □ Could I modify this if requirements changed?
    □ Would I catch it if this were wrong?

    If ANY answer is "no" → you don't own the output yet.

═══════════════════════════════════════════════════════════════
START SMALL, BUILD UP
═══════════════════════════════════════════════════════════════

    Order of implementation:
    1. First: Get ONE ticker working (NVDA)
    2. Then: Add the comparison ticker (CHGG)
    3. Then: Add entropy calculation
    4. Then: Add visualization
    5. Finally: Add cascade structure

    Don't try to build everything at once.

═══════════════════════════════════════════════════════════════
AI PROMPTING GUIDANCE (For this stage)
═══════════════════════════════════════════════════════════════

    GOOD prompts for IMPLEMENT:
    • "Write a function that calculates Shannon entropy given sector weights"
    • "This code produces [error]. I expected [behavior]. Here's the code..."
    • "Explain why this return calculation uses (end/start - 1) * 100"

    DANGEROUS prompts (avoid):
    • "Build me a complete analysis of ChatGPT's market impact"
    • "Analyze this data and tell me what's interesting"
    • "Write code to prove ChatGPT caused sample space expansion"

    Remember: The more specific and bounded, the more useful and verifiable.

═══════════════════════════════════════════════════════════════
KEY DECISIONS LOG (Document as you go)
═══════════════════════════════════════════════════════════════

    Decision 1: Using log2 for entropy (not ln)
        Why: Gives results in "bits" — more interpretable

    Decision 2: Using adjusted close prices
        Why: Accounts for splits/dividends automatically

    Decision 3: Normalizing to 100 at launch date
        Why: Makes visual comparison of divergence clear

    Decision 4: Using sector weights (not individual stock weights)
        Why: Sample space expansion is about categories, not single stocks

    [Add your decisions as you implement]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

# --- Configuration ---
CHATGPT_LAUNCH = "2022-11-30"
ANALYSIS_END = "2025-01-01"

# Key tickers for analysis
WINNERS = {
    "NVDA": "Nvidia (AI Infrastructure)",
    "MSFT": "Microsoft (Azure AI)",
    "GOOGL": "Google (AI Search)",
    "META": "Meta (AI Investment)",
}

LOSERS = {
    "CHGG": "Chegg (Education Disruption)",
}

BENCHMARK = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
}

# Sector ETFs for entropy analysis
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLY": "Consumer Discretionary",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch stock data using yfinance.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def fetch_multiple_stocks(tickers: Dict[str, str],
                          start_date: str,
                          end_date: str) -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple tickers."""
    results = {}
    for ticker, name in tickers.items():
        try:
            results[ticker] = fetch_stock_data(ticker, start_date, end_date)
            print(f"  Loaded {ticker}: {name}")
        except Exception as e:
            print(f"  Error loading {ticker}: {e}")
    return results


# ============================================================
# RETURNS ANALYSIS
# ============================================================

def calculate_returns(data: pd.DataFrame,
                      start_date: str = None) -> Dict[str, float]:
    """
    Calculate various return metrics.

    Returns:
        Dictionary with total return, annualized return, volatility
    """
    if start_date:
        data = data[data.index >= start_date]

    if len(data) < 2:
        return {"total_return": 0, "annualized": 0, "volatility": 0}

    start_price = data["Close"].iloc[0]
    end_price = data["Close"].iloc[-1]

    total_return = (end_price / start_price - 1) * 100

    # Annualized return
    days = (data.index[-1] - data.index[0]).days
    years = days / 365.25
    if years > 0:
        annualized = ((end_price / start_price) ** (1 / years) - 1) * 100
    else:
        annualized = 0

    # Volatility
    daily_returns = data["Close"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100

    return {
        "total_return": total_return,
        "annualized": annualized,
        "volatility": volatility,
        "start_price": start_price,
        "end_price": end_price,
    }


def compare_returns(stock_data: Dict[str, pd.DataFrame],
                    labels: Dict[str, str],
                    start_date: str) -> pd.DataFrame:
    """Create comparison table of returns."""
    results = []

    for ticker, data in stock_data.items():
        metrics = calculate_returns(data, start_date)
        results.append({
            "Ticker": ticker,
            "Name": labels.get(ticker, ticker),
            "Total Return (%)": f"{metrics['total_return']:.1f}",
            "Annualized (%)": f"{metrics['annualized']:.1f}",
            "Volatility (%)": f"{metrics['volatility']:.1f}",
        })

    return pd.DataFrame(results)


# ============================================================
# ENTROPY ANALYSIS
# ============================================================

def sector_entropy(weights: np.ndarray) -> float:
    """
    Calculate Shannon entropy of sector weights.

    Lower entropy = more concentrated market
    Higher entropy = more diversified market

    Args:
        weights: Array of sector weights (should sum to 1)

    Returns:
        Shannon entropy in bits
    """
    weights = np.array(weights)
    weights = weights[weights > 0]  # Remove zeros

    if len(weights) == 0:
        return 0.0

    # Normalize if not already
    weights = weights / weights.sum()

    return -np.sum(weights * np.log2(weights))


def calculate_concentration_ratio(weights: np.ndarray, top_n: int = 7) -> float:
    """
    Calculate concentration ratio (CR_n).

    CR_7 = sum of top 7 weights / total
    """
    weights = np.array(weights)
    weights = np.sort(weights)[::-1]  # Sort descending

    return weights[:top_n].sum() / weights.sum()


def entropy_change_analysis(weights_before: List[float],
                           weights_after: List[float]) -> Dict:
    """
    Analyze how entropy changed between two periods.
    """
    entropy_before = sector_entropy(weights_before)
    entropy_after = sector_entropy(weights_after)

    cr7_before = calculate_concentration_ratio(np.array(weights_before))
    cr7_after = calculate_concentration_ratio(np.array(weights_after))

    return {
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "entropy_change": entropy_after - entropy_before,
        "cr7_before": cr7_before,
        "cr7_after": cr7_after,
        "cr7_change": cr7_after - cr7_before,
    }


# ============================================================
# DISRUPTION CASCADE
# ============================================================

# The cascade structure
DISRUPTION_CASCADE = {
    "trigger": {
        "event": "ChatGPT Launch",
        "date": "2022-11-30",
        "description": "100M users in 2 months, fastest consumer app adoption",
    },
    "direct_effects": {
        "winners": [
            {"name": "Nvidia", "ticker": "NVDA", "reason": "GPU demand for AI training"},
            {"name": "Microsoft", "ticker": "MSFT", "reason": "Azure AI, OpenAI partnership"},
            {"name": "Google", "ticker": "GOOGL", "reason": "AI search, Gemini"},
        ],
        "losers": [
            {"name": "Chegg", "ticker": "CHGG", "reason": "Homework help disrupted"},
            {"name": "Pearson", "ticker": "PSO", "reason": "Education content disrupted"},
        ],
    },
    "second_order": {
        "winners": [
            {"name": "Data Center REITs", "reason": "Increased compute demand"},
            {"name": "Energy/Utilities", "reason": "Power demand from AI"},
            {"name": "Semiconductor Equipment", "reason": "Chip manufacturing"},
        ],
        "losers": [
            {"name": "Traditional Tutoring", "reason": "AI tutors compete"},
            {"name": "Content Farms", "reason": "AI-generated content"},
        ],
    },
    "third_order": {
        "trends": [
            "Curriculum restructuring in education",
            "New job categories (prompt engineering)",
            "Regulatory discussions (AI safety)",
            "Energy infrastructure investment",
        ],
    },
}


def print_cascade():
    """Print the disruption cascade in readable format."""
    cascade = DISRUPTION_CASCADE

    print("\n" + "=" * 60)
    print("CHATGPT DISRUPTION CASCADE")
    print("=" * 60)

    print(f"\n[TRIGGER] {cascade['trigger']['event']}")
    print(f"  Date: {cascade['trigger']['date']}")
    print(f"  {cascade['trigger']['description']}")

    print("\n[DIRECT EFFECTS]")
    print("  Winners:")
    for w in cascade["direct_effects"]["winners"]:
        print(f"    + {w['name']} ({w['ticker']}): {w['reason']}")
    print("  Losers:")
    for l in cascade["direct_effects"]["losers"]:
        print(f"    - {l['name']} ({l['ticker']}): {l['reason']}")

    print("\n[SECOND-ORDER EFFECTS]")
    print("  Winners:")
    for w in cascade["second_order"]["winners"]:
        print(f"    + {w['name']}: {w['reason']}")
    print("  Losers:")
    for l in cascade["second_order"]["losers"]:
        print(f"    - {l['name']}: {l['reason']}")

    print("\n[THIRD-ORDER EFFECTS]")
    for trend in cascade["third_order"]["trends"]:
        print(f"    → {trend}")


# ============================================================
# VISUALIZATION
# ============================================================

def plot_creative_destruction(stock_data: Dict[str, pd.DataFrame],
                              labels: Dict[str, str],
                              start_date: str):
    """
    Create visualization showing winners vs losers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Normalize all prices to 100 at start
    def normalize(data, start):
        data = data[data.index >= start]
        return (data["Close"] / data["Close"].iloc[0]) * 100

    # Plot 1: Winners
    ax1 = axes[0, 0]
    for ticker in ["NVDA", "MSFT"]:
        if ticker in stock_data:
            norm = normalize(stock_data[ticker], start_date)
            ax1.plot(norm.index, norm.values, label=labels.get(ticker, ticker))
    ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(x=pd.to_datetime(start_date), color="red", linestyle=":", alpha=0.7)
    ax1.set_title("Winners: AI Infrastructure", fontsize=12)
    ax1.set_ylabel("Normalized Price (100 = ChatGPT Launch)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Losers
    ax2 = axes[0, 1]
    for ticker in ["CHGG"]:
        if ticker in stock_data:
            norm = normalize(stock_data[ticker], start_date)
            ax2.plot(norm.index, norm.values, label=labels.get(ticker, ticker), color="red")
    if "SPY" in stock_data:
        norm = normalize(stock_data["SPY"], start_date)
        ax2.plot(norm.index, norm.values, label="S&P 500", color="gray", linestyle="--")
    ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=pd.to_datetime(start_date), color="red", linestyle=":", alpha=0.7)
    ax2.set_title("Losers: Knowledge Work Disruption", fontsize=12)
    ax2.set_ylabel("Normalized Price (100 = ChatGPT Launch)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: All together
    ax3 = axes[1, 0]
    colors = {"NVDA": "green", "MSFT": "blue", "CHGG": "red", "SPY": "gray"}
    for ticker, color in colors.items():
        if ticker in stock_data:
            norm = normalize(stock_data[ticker], start_date)
            linestyle = "--" if ticker == "SPY" else "-"
            ax3.plot(norm.index, norm.values, label=labels.get(ticker, ticker),
                    color=color, linestyle=linestyle)
    ax3.axvline(x=pd.to_datetime(start_date), color="red", linestyle=":", alpha=0.7,
               label="ChatGPT Launch")
    ax3.set_title("Creative Destruction: Winners vs Losers", fontsize=12)
    ax3.set_ylabel("Normalized Price")
    ax3.set_xlabel("Date")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")  # Log scale to show both +700% and -99%

    # Plot 4: Returns bar chart
    ax4 = axes[1, 1]
    all_labels = {**labels, **BENCHMARK}
    returns_data = []
    for ticker, data in stock_data.items():
        metrics = calculate_returns(data, start_date)
        returns_data.append({
            "ticker": ticker,
            "name": all_labels.get(ticker, ticker),
            "return": metrics["total_return"],
        })

    df_returns = pd.DataFrame(returns_data).sort_values("return", ascending=True)
    colors = ["green" if r > 0 else "red" for r in df_returns["return"]]
    ax4.barh(df_returns["name"], df_returns["return"], color=colors, alpha=0.7)
    ax4.axvline(x=0, color="black", linewidth=0.5)
    ax4.set_title("Total Returns Since ChatGPT Launch", fontsize=12)
    ax4.set_xlabel("Return (%)")

    plt.tight_layout()
    plt.savefig("creative_destruction.png", dpi=150, bbox_inches="tight")
    plt.show()

    return fig


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    """
    Main analysis workflow following DRIVER methodology.
    """
    print("=" * 60)
    print("Case 2: ChatGPT Launch 2022")
    print("Sample Space Expansion Analysis")
    print("=" * 60)

    # Step 1: Print the disruption cascade
    print_cascade()

    # Step 2: Fetch stock data
    print("\n[1] Fetching stock data...")
    all_tickers = {**WINNERS, **LOSERS, **BENCHMARK}
    stock_data = fetch_multiple_stocks(all_tickers, CHATGPT_LAUNCH, ANALYSIS_END)

    # Step 3: Calculate returns
    print("\n[2] Returns Since ChatGPT Launch:")
    returns_df = compare_returns(stock_data, all_tickers, CHATGPT_LAUNCH)
    print(returns_df.to_string(index=False))

    # Step 4: Entropy analysis
    print("\n[3] Sector Entropy Analysis:")
    # Approximate S&P 500 sector weights (illustrative)
    weights_nov2022 = [0.20, 0.10, 0.12, 0.05, 0.15, 0.08, 0.03, 0.03, 0.03, 0.08]  # ~87% + other
    weights_nov2024 = [0.32, 0.08, 0.10, 0.04, 0.12, 0.08, 0.02, 0.03, 0.02, 0.07]  # Mag 7 up

    analysis = entropy_change_analysis(weights_nov2022, weights_nov2024)
    print(f"  Entropy (Nov 2022): {analysis['entropy_before']:.3f} bits")
    print(f"  Entropy (Nov 2024): {analysis['entropy_after']:.3f} bits")
    print(f"  Entropy Change: {analysis['entropy_change']:.3f} bits")
    print(f"  CR7 (Nov 2022): {analysis['cr7_before']:.1%}")
    print(f"  CR7 (Nov 2024): {analysis['cr7_after']:.1%}")
    print(f"  CR7 Change: {analysis['cr7_change']:.1%}")

    # Step 5: Key insight
    print("\n[4] Key Insight:")
    print("  Paradox: Sample space EXPANDED (new AI asset class)")
    print("           but entropy DECREASED (Mag 7 concentration)")
    print("  This means: New category entered, but dominated by few players")

    # Step 6: Create visualizations
    print("\n[5] Creating visualizations...")
    try:
        plot_creative_destruction(stock_data, all_tickers, CHATGPT_LAUNCH)
        print("  Saved: creative_destruction.png")
    except Exception as e:
        print(f"  Visualization error: {e}")

    return stock_data, returns_df


# ============================================================
# VALIDATE
# ============================================================
"""
═══════════════════════════════════════════════════════════════
VALIDATION CHECKLIST (Complete ALL before claiming "done")
═══════════════════════════════════════════════════════════════

DATA VALIDATION:
    □ Stock data loads for all tickers (NVDA, CHGG, SPY, MSFT)
    □ Date range is correct (Nov 30, 2022 - Dec 2024)
    □ No unexpected gaps or missing data
    □ Prices are adjusted (splits/dividends handled)

CALCULATION VALIDATION:
    □ Return calculation: (end/start - 1) * 100 gives %
    □ Entropy calculation: Uniform [0.25, 0.25, 0.25, 0.25] gives 2.0 bits
    □ Single weight [1.0] gives entropy = 0 (monopoly)
    □ Weights that don't sum to 1 are normalized

LOGIC VALIDATION:
    □ Cascade makes causal sense (ChatGPT → AI demand → Nvidia)
    □ Winners/losers are correctly categorized
    □ Entropy interpretation is correct (lower = more concentrated)

═══════════════════════════════════════════════════════════════
THE "BET MONEY" TEST
═══════════════════════════════════════════════════════════════

    Ask yourself: Would I bet my own money on these results?

    Specifically:
    □ Would I bet $100 that NVDA return is > 500%? (Check with yfinance)
    □ Would I bet $100 that CHGG return is < -80%? (Check)
    □ Would I bet $100 that divergence > 800%? (Calculate)
    □ Would I bet $100 that entropy decreased? (Verify with real weights)

    If you hesitate on any → investigate before proceeding.

═══════════════════════════════════════════════════════════════
EDGE CASES TO CHECK
═══════════════════════════════════════════════════════════════

    □ What if a ticker has missing days? (Should handle gracefully)
    □ What if weights don't sum to 1? (Should normalize)
    □ What if all weight on one sector? (Entropy should be 0)
    □ What if yfinance returns empty data? (Should error clearly)

═══════════════════════════════════════════════════════════════
AI-SPECIFIC VALIDATION (Trust but verify)
═══════════════════════════════════════════════════════════════

    If AI generated any code or facts, verify:
    □ Are the dates correct? (ChatGPT launch was Nov 30, 2022)
    □ Are the tickers correct? (NVDA not NVA, CHGG not CHG)
    □ Is the entropy formula correct? (H = -Σ p * log2(p))
    □ Does the return calculation actually match Yahoo Finance?

    NEVER use AI alone to validate AI output.

═══════════════════════════════════════════════════════════════
RUN AUTOMATED TESTS
═══════════════════════════════════════════════════════════════

    Command: pytest tests/test_sample_space.py -v

    Expected: 32 passed, 2 skipped (live data tests)

═══════════════════════════════════════════════════════════════
V-D LOOP: When validation reveals wrong problem
═══════════════════════════════════════════════════════════════

    If validation shows your results don't support the thesis:
    • DON'T force the conclusion
    • DO loop back to DISCOVER & DEFINE
    • Ask: "Did I define the right problem?"
    • Ask: "Is sample space expansion the right frame?"

    The V-D loop is uncomfortable but essential.
"""
# Run tests: pytest tests/test_sample_space.py -v

# ============================================================
# EVOLVE
# ============================================================
"""
═══════════════════════════════════════════════════════════════
EVOLUTION IS NOT JUST POLISH
═══════════════════════════════════════════════════════════════

    Evolution can mean:
    • Refactoring: Making the same thing cleaner
    • Optimizing: Making it faster or more efficient
    • Generalizing: Making it work for more cases
    • Documenting: Making it understandable to others
    • Rebuilding: Fixing fundamental issues revealed by usage

═══════════════════════════════════════════════════════════════
E-I LOOP: Sometimes EVOLVE means rebuild
═══════════════════════════════════════════════════════════════

    While improving, you might realize fundamental pieces need rebuilding.

    Signs you need to loop back to IMPLEMENT:
    • The core logic is flawed (not just messy)
    • Edge cases break the fundamentals
    • The architecture can't support needed extensions

═══════════════════════════════════════════════════════════════
POTENTIAL EXTENSIONS (Prioritized)
═══════════════════════════════════════════════════════════════

    IMMEDIATE VALUE:
    □ Add networkx visualization of cascade graph
    □ Export cascade as JSON for other tools
    □ Add more tickers to winners/losers

    MEDIUM EFFORT:
    □ Calculate rolling entropy over time (animate the concentration)
    □ Add second/third-order companies (AMD, TSMC, DELL, SMCI)
    □ Compare to other technology shifts (iPhone 2007, COVID 2020)

    STRETCH GOALS:
    □ Sentiment analysis of earnings call transcripts
    □ Real-time monitoring of new "sample space" signals
    □ Predictive model for next paradigm shift

═══════════════════════════════════════════════════════════════
BUILDING YOUR PATTERN LIBRARY
═══════════════════════════════════════════════════════════════

    Every significant project teaches you something reusable.
    Capture these patterns from this case:

    PATTERN 1: Shannon Entropy for Concentration
    ─────────────────────────────────────────────
        Use case: Measuring market concentration
        Function: sector_entropy(weights)
        Key insight: Lower = more concentrated

    PATTERN 2: Normalized Price Comparison
    ─────────────────────────────────────────────
        Use case: Comparing assets with different price scales
        Method: Normalize to 100 at reference date
        Key insight: Shows relative performance regardless of price

    PATTERN 3: Cascade Mapping Structure
    ─────────────────────────────────────────────
        Use case: Tracing second/third-order effects
        Structure: trigger → direct → second_order → third_order
        Key insight: Effects propagate in predictable patterns

    PATTERN 4: yfinance Data Fetching
    ─────────────────────────────────────────────
        Use case: Historical stock data without API key
        Gotchas: MultiIndex columns, timezone handling
        Template: fetch_stock_data() function

    [Add your own patterns as you discover them]

═══════════════════════════════════════════════════════════════
CODE CLEANUP CHECKLIST
═══════════════════════════════════════════════════════════════

    □ Functions have docstrings explaining inputs/outputs
    □ Magic numbers are replaced with named constants
    □ Error handling is appropriate (not excessive)
    □ Code is DRY (Don't Repeat Yourself)
    □ Variable names are descriptive
    □ Comments explain "why" not "what"
"""

# ============================================================
# REFLECT
# ============================================================
"""
═══════════════════════════════════════════════════════════════
THE FIVE REFLECTION QUESTIONS (Answer after completing analysis)
═══════════════════════════════════════════════════════════════

Take 15 minutes after completing your analysis to answer:

1. WHAT WORKED WELL that I should keep doing?
   ─────────────────────────────────────────────
   [Your answer here]

   Example: "The cascade mapping structure helped me think
   systematically about second-order effects I would have missed."

2. WHAT DIDN'T WORK that I should avoid?
   ─────────────────────────────────────────────
   [Your answer here]

   Example: "I spent too long on visualization before validating
   the underlying calculations were correct."

3. WHAT SURPRISED ME that I should remember?
   ─────────────────────────────────────────────
   [Your answer here]

   Example: "The paradox that sample space expanded but entropy
   decreased — this wasn't intuitive until I worked through it."

4. WHAT WOULD I DO DIFFERENTLY knowing what I know now?
   ─────────────────────────────────────────────
   [Your answer here]

   Example: "Start with the simplest validation (just NVDA vs CHGG)
   before adding complexity."

5. WHAT PATTERNS HERE might apply elsewhere?
   ─────────────────────────────────────────────
   [Your answer here]

   Example: "The cascade mapping structure could apply to any
   disruption event — regulatory changes, technology shifts, etc."

═══════════════════════════════════════════════════════════════
THE TEACHING TEST
═══════════════════════════════════════════════════════════════

    One of the best tests of understanding: Could you teach this?

    Imagine explaining to a colleague:
    □ What is "sample space expansion" vs "regime shift"?
    □ Why did Nvidia go up 700% while Chegg fell 99%?
    □ What does the entropy paradox tell us?
    □ How would you detect the NEXT sample space expansion?

    If you can't explain it clearly, you might not understand it.

═══════════════════════════════════════════════════════════════
CONNECTING TO BROADER PRINCIPLES
═══════════════════════════════════════════════════════════════

    This case is an instance of general patterns:

    PROBLEM TYPE: Structural market change detection
    SOLUTION TYPE: Information-theoretic analysis (entropy)
    TRANSFERABLE TO:
        • Any technology disruption (autonomous vehicles, quantum computing)
        • Regulatory regime changes (GENIUS Act — Week 4)
        • Geopolitical shifts (energy decoupling — Week 5)

═══════════════════════════════════════════════════════════════
WHAT I LEARNED ABOUT DRIVER
═══════════════════════════════════════════════════════════════

    After this case, I understand better:
    □ DISCOVER & DEFINE: How to set success criteria upfront
    □ REPRESENT: Why visual workflows matter
    □ IMPLEMENT: The importance of ownership test
    □ VALIDATE: "Bet money" as a mental check
    □ EVOLVE: Building pattern libraries for reuse
    □ REFLECT: Capturing transferable insights

    Loops I experienced:
    □ R-I loop: [What gap in my plan did building reveal?]
    □ V-D loop: [Did validation change my understanding of the problem?]

═══════════════════════════════════════════════════════════════
KEY TAKEAWAYS FROM CASE 2
═══════════════════════════════════════════════════════════════

    1. SAMPLE SPACE EXPANSION ≠ REGIME SHIFT
       Week 1 (Tariff) = P changed (probabilities within existing universe)
       Week 3 (ChatGPT) = X changed (the universe itself expanded)

    2. CREATIVE DESTRUCTION IS MEASURABLE
       Not just narrative — quantifiable divergence (800%+)

    3. THE PARADOX IS THE INSIGHT
       Sample space expanded BUT concentration increased
       New categories can be dominated by few players

    4. ENTROPY CAPTURES STRUCTURAL CHANGE
       VIX measures volatility; entropy measures information structure

    5. CASCADE THINKING REVEALS HIDDEN CONNECTIONS
       Second and third-order effects are often where the opportunity lives
"""

if __name__ == "__main__":
    stock_data, returns_df = main()
