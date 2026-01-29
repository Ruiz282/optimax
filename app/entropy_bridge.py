"""
Entropy Bridge — Connects parent repo's information theory framework
to options strategy signals.

Maps entropy changes and concentration shifts to actionable options
strategy regime signals.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# Core Entropy Functions (from parent repo)
# ─────────────────────────────────────────────

def sector_entropy(weights: np.ndarray) -> float:
    """Calculate Shannon entropy of sector weights (bits)."""
    weights = np.array(weights, dtype=float)
    weights = weights[weights > 0]
    if len(weights) == 0:
        return 0.0
    weights = weights / weights.sum()
    return -np.sum(weights * np.log2(weights))


def max_entropy(n: int) -> float:
    """Maximum entropy for n equally-weighted items."""
    if n <= 0:
        return 0.0
    return np.log2(n)


def normalized_entropy(weights: np.ndarray) -> float:
    """Entropy normalized to 0-1 scale (1 = perfectly diversified)."""
    n = len([w for w in weights if w > 0])
    if n <= 1:
        return 0.0
    return sector_entropy(weights) / max_entropy(n)


def herfindahl_index(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman Index. Higher = more concentrated."""
    weights = np.array(weights, dtype=float)
    weights = weights[weights > 0]
    weights = weights / weights.sum()
    return float(np.sum(weights ** 2))


def concentration_ratio(weights: np.ndarray, top_n: int = 7) -> float:
    """CR_n: sum of top-n weights."""
    weights = np.array(weights, dtype=float)
    weights = weights[weights > 0]
    weights = weights / weights.sum()
    sorted_w = np.sort(weights)[::-1]
    return float(np.sum(sorted_w[:top_n]))


# ─────────────────────────────────────────────
# Sector ETFs for Real-Time Entropy
# ─────────────────────────────────────────────

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLY": "Consumer Disc.",
    "XLC": "Communication",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
}

# Historical baseline (Nov 2022, pre-AI boom)
BASELINE_WEIGHTS = np.array([
    0.20, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.05, 0.03, 0.03, 0.03
])
BASELINE_ENTROPY = sector_entropy(BASELINE_WEIGHTS)
BASELINE_HHI = herfindahl_index(BASELINE_WEIGHTS)


# ─────────────────────────────────────────────
# Entropy Signal
# ─────────────────────────────────────────────

@dataclass
class EntropySignal:
    """Market entropy analysis result with options strategy implications."""
    # Current state
    current_entropy: float
    current_hhi: float
    current_normalized: float
    current_cr7: float

    # Baseline comparison
    baseline_entropy: float
    entropy_change: float
    entropy_change_pct: float

    # Rolling analysis
    entropy_30d_ago: Optional[float]
    entropy_trend: str  # "rising", "falling", "stable"

    # Strategy signal
    regime: str          # "UNCERTAINTY", "CONCENTRATION", "STABLE"
    strategy_bias: str   # "BUY_PREMIUM", "SELL_PREMIUM", "NEUTRAL"
    signal_strength: str # "STRONG", "MODERATE", "WEAK"

    # Sector breakdown
    sector_weights: Dict[str, float]
    top_sectors: List[Tuple[str, float]]

    # Explanation
    explanation: str
    strategy_implications: List[str]


def compute_market_entropy(lookback_days: int = 90) -> Optional[EntropySignal]:
    """
    Compute current market entropy from sector ETF market caps and compare
    to historical baseline.

    Uses relative performance of sector ETFs as a proxy for weight changes.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 10)

    # Fetch sector ETF data
    sector_prices = {}
    for etf, name in SECTOR_ETFS.items():
        try:
            ticker = yf.Ticker(etf)
            hist = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            if not hist.empty:
                sector_prices[etf] = hist["Close"]
        except Exception:
            continue

    if len(sector_prices) < 8:
        return None

    prices_df = pd.DataFrame(sector_prices).dropna()
    if len(prices_df) < 30:
        return None

    # ── Current weights (proxy via relative market cap / price performance) ──
    # Use latest prices normalized to a base period
    latest_prices = prices_df.iloc[-1]
    base_prices = prices_df.iloc[0]
    relative_performance = latest_prices / base_prices

    # Weight by relative performance (stronger sectors get higher weight)
    # Start from baseline and adjust by performance
    adjusted_weights = BASELINE_WEIGHTS.copy()
    etf_list = list(SECTOR_ETFS.keys())
    for i, etf in enumerate(etf_list):
        if etf in relative_performance.index and i < len(adjusted_weights):
            adjusted_weights[i] *= relative_performance[etf]

    # Normalize
    adjusted_weights = adjusted_weights / adjusted_weights.sum()

    # ── Compute entropy metrics ──
    current_entropy = sector_entropy(adjusted_weights)
    current_hhi = herfindahl_index(adjusted_weights)
    current_normalized = normalized_entropy(adjusted_weights)
    current_cr7 = concentration_ratio(adjusted_weights, 7)

    # ── 30-day ago entropy ──
    if len(prices_df) >= 30:
        prices_30d_ago = prices_df.iloc[-30]
        perf_30d_ago = prices_30d_ago / base_prices
        weights_30d_ago = BASELINE_WEIGHTS.copy()
        for i, etf in enumerate(etf_list):
            if etf in perf_30d_ago.index and i < len(weights_30d_ago):
                weights_30d_ago[i] *= perf_30d_ago[etf]
        weights_30d_ago = weights_30d_ago / weights_30d_ago.sum()
        entropy_30d_ago = sector_entropy(weights_30d_ago)
    else:
        entropy_30d_ago = None

    # ── Entropy change from baseline ──
    entropy_change = current_entropy - BASELINE_ENTROPY
    entropy_change_pct = (entropy_change / BASELINE_ENTROPY) * 100

    # ── Trend ──
    if entropy_30d_ago is not None:
        recent_change = current_entropy - entropy_30d_ago
        if recent_change > 0.03:
            entropy_trend = "rising"
        elif recent_change < -0.03:
            entropy_trend = "falling"
        else:
            entropy_trend = "stable"
    else:
        entropy_trend = "stable"

    # ── Regime classification ──
    if entropy_change < -0.10:
        regime = "CONCENTRATION"
        strategy_bias = "SELL_PREMIUM"
        if abs(entropy_change) > 0.20:
            signal_strength = "STRONG"
        else:
            signal_strength = "MODERATE"
    elif entropy_change > 0.10:
        regime = "UNCERTAINTY"
        strategy_bias = "BUY_PREMIUM"
        if abs(entropy_change) > 0.20:
            signal_strength = "STRONG"
        else:
            signal_strength = "MODERATE"
    else:
        regime = "STABLE"
        strategy_bias = "NEUTRAL"
        signal_strength = "WEAK"

    # Override with trend if trend disagrees with level
    if entropy_trend == "rising" and regime != "UNCERTAINTY":
        signal_strength = "MODERATE" if signal_strength == "STRONG" else "WEAK"
    elif entropy_trend == "falling" and regime != "CONCENTRATION":
        signal_strength = "MODERATE" if signal_strength == "STRONG" else "WEAK"

    # ── Sector breakdown ──
    sector_weights = {}
    for i, etf in enumerate(etf_list):
        if i < len(adjusted_weights):
            sector_weights[SECTOR_ETFS[etf]] = float(adjusted_weights[i])

    top_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)[:5]

    # ── Build explanations ──
    explanation = _build_explanation(
        regime, entropy_change, entropy_change_pct, entropy_trend,
        current_entropy, current_hhi, top_sectors,
    )
    strategy_implications = _build_implications(regime, strategy_bias, entropy_trend)

    return EntropySignal(
        current_entropy=current_entropy,
        current_hhi=current_hhi,
        current_normalized=current_normalized,
        current_cr7=current_cr7,
        baseline_entropy=BASELINE_ENTROPY,
        entropy_change=entropy_change,
        entropy_change_pct=entropy_change_pct,
        entropy_30d_ago=entropy_30d_ago,
        entropy_trend=entropy_trend,
        regime=regime,
        strategy_bias=strategy_bias,
        signal_strength=signal_strength,
        sector_weights=sector_weights,
        top_sectors=top_sectors,
        explanation=explanation,
        strategy_implications=strategy_implications,
    )


def _build_explanation(regime, entropy_change, entropy_change_pct, trend,
                       current_entropy, current_hhi, top_sectors):
    """Build human-readable entropy explanation."""
    direction = "decreased" if entropy_change < 0 else "increased"
    top3 = ", ".join(f"{name} ({w:.1%})" for name, w in top_sectors[:3])

    if regime == "CONCENTRATION":
        return (
            f"Market entropy has {direction} by {abs(entropy_change):.3f} bits "
            f"({abs(entropy_change_pct):.1f}%) from the Nov 2022 baseline. "
            f"This signals increasing concentration — fewer sectors are driving returns. "
            f"Top sectors: {top3}. "
            f"HHI at {current_hhi:.4f} confirms concentration. "
            f"30-day trend: {trend}."
        )
    elif regime == "UNCERTAINTY":
        return (
            f"Market entropy has {direction} by {abs(entropy_change):.3f} bits "
            f"({abs(entropy_change_pct):.1f}%) from the Nov 2022 baseline. "
            f"This signals increasing dispersion — more sectors are participating in returns. "
            f"The investment universe is expanding. "
            f"Top sectors: {top3}. "
            f"30-day trend: {trend}."
        )
    else:
        return (
            f"Market entropy is near baseline (change: {entropy_change:+.3f} bits, "
            f"{entropy_change_pct:+.1f}%). No strong regime signal. "
            f"Top sectors: {top3}. "
            f"30-day trend: {trend}."
        )


def _build_implications(regime, strategy_bias, trend):
    """Build strategy implications list."""
    implications = []

    if regime == "CONCENTRATION":
        implications.extend([
            "Market is concentrating into fewer sectors — momentum is strong in leaders",
            "Index-level volatility tends to compress in concentrated markets",
            "FAVOR: Iron condors on indices (range-bound behavior expected)",
            "FAVOR: Credit spreads in the direction of concentration (e.g., bull put spreads on tech leaders)",
            "FAVOR: Covered calls on winners (upside decelerating)",
            "AVOID: Long straddles on indices (unlikely to break range)",
        ])
        if trend == "falling":
            implications.append("TREND CONFIRMS: Entropy still falling — concentration deepening")
        elif trend == "rising":
            implications.append("CAUTION: Entropy trend is reversing — concentration may be peaking")
    elif regime == "UNCERTAINTY":
        implications.extend([
            "Market dispersion is increasing — the game may be changing",
            "Higher entropy means more uncertainty about which sectors will lead",
            "FAVOR: Long straddles/strangles (positioning for larger moves)",
            "FAVOR: Long puts as tail hedges (regime changes often involve drawdowns)",
            "FAVOR: Calendar spreads (buy cheap long-term vol)",
            "AVOID: Selling naked premium (tail risk is elevated)",
        ])
        if trend == "rising":
            implications.append("TREND CONFIRMS: Entropy still rising — uncertainty deepening")
        elif trend == "falling":
            implications.append("CAUTION: Entropy trend stabilizing — new regime may be settling")
    else:
        implications.extend([
            "No strong entropy signal — market structure is near historical norms",
            "Standard IV-based strategy selection applies",
            "Focus on individual stock IV percentile rather than market regime",
        ])

    return implications


# ─────────────────────────────────────────────
# Score Adjustment for Strategy Engine
# ─────────────────────────────────────────────

def entropy_score_adjustment(strategy_name: str, signal: EntropySignal) -> float:
    """
    Return a score adjustment (-15 to +15) for a strategy based on
    the current entropy regime.
    """
    if signal is None:
        return 0.0

    # Strategy affinities by regime
    concentration_favored = {
        "Iron Condor": 12,
        "Bull Put Spread": 10,
        "Bear Call Spread": 10,
        "Covered Call": 8,
        "Cash-Secured Put": 8,
        "Calendar Spread": 5,
        "Bull Call Spread": 3,
        "Bear Put Spread": 3,
    }
    concentration_penalized = {
        "Long Straddle": -10,
        "Long Strangle": -10,
        "Long Call": -3,
        "Long Put": -3,
    }

    uncertainty_favored = {
        "Long Straddle": 12,
        "Long Strangle": 12,
        "Long Put": 8,
        "Long Call": 5,
        "Calendar Spread": 5,
    }
    uncertainty_penalized = {
        "Iron Condor": -10,
        "Bull Put Spread": -5,
        "Bear Call Spread": -5,
        "Covered Call": -5,
        "Cash-Secured Put": -5,
    }

    adjustment = 0.0

    if signal.regime == "CONCENTRATION":
        adjustment += concentration_favored.get(strategy_name, 0)
        adjustment += concentration_penalized.get(strategy_name, 0)
    elif signal.regime == "UNCERTAINTY":
        adjustment += uncertainty_favored.get(strategy_name, 0)
        adjustment += uncertainty_penalized.get(strategy_name, 0)

    # Scale by signal strength
    if signal.signal_strength == "STRONG":
        adjustment *= 1.0
    elif signal.signal_strength == "MODERATE":
        adjustment *= 0.6
    else:
        adjustment *= 0.3

    return adjustment
