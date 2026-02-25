"""
Entropy Bridge — Connects information theory framework to options strategy signals.

Maps entropy changes and concentration shifts to actionable options
strategy regime signals. Uses live market caps for sector weights
and continuous functions for smooth regime transitions.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# Core Entropy Functions
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

# Static fallback weights (used only when live data is unavailable)
FALLBACK_WEIGHTS = np.array([
    0.20, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.05, 0.03, 0.03, 0.03
])


# ─────────────────────────────────────────────
# Live Market Cap Weights
# ─────────────────────────────────────────────

def fetch_live_sector_weights() -> Optional[np.ndarray]:
    """
    Fetch actual market caps for sector ETFs and compute weights.
    Returns normalized weight array in SECTOR_ETFS order, or None on failure.
    """
    caps = {}
    etf_list = list(SECTOR_ETFS.keys())
    # Batch download to minimize API calls
    try:
        tickers_str = " ".join(etf_list)
        data = yf.download(tickers_str, period="1d", progress=False)
        # Use latest closing prices × shares as proxy, or try info
    except Exception:
        pass
    for i, etf in enumerate(etf_list):
        try:
            ticker = yf.Ticker(etf)
            cap = None
            # Try fast_info first
            try:
                cap = ticker.fast_info.get('marketCap')
            except (KeyError, AttributeError):
                pass
            # For ETFs, marketCap is often None — use totalAssets instead
            if not cap or cap <= 0:
                try:
                    info = ticker.info
                    cap = info.get('totalAssets') or info.get('netAssets') or info.get('marketCap')
                except Exception:
                    pass
            if cap and cap > 0:
                caps[etf] = float(cap)
            if i % 4 == 3:
                time.sleep(0.3)  # gentle rate limiting
        except Exception:
            continue

    if len(caps) < 8:
        return None

    weights = np.array([caps.get(etf, 0) for etf in etf_list], dtype=float)
    # Replace any zeros with a small value to avoid division issues
    weights = np.maximum(weights, 1.0)
    return weights / weights.sum()


# ─────────────────────────────────────────────
# Continuous Entropy Signal
# ─────────────────────────────────────────────

def continuous_entropy_signal(entropy_change: float, scale: float = 0.15) -> float:
    """
    Continuous regime signal from entropy change.

    Returns a value from -1.0 (full concentration) to +1.0 (full uncertainty).
    Uses tanh for smooth, bounded scaling.

    Args:
        entropy_change: Current entropy minus baseline entropy (bits)
        scale: Controls sensitivity — smaller = more sensitive to small changes
    """
    return float(np.tanh(entropy_change / scale))


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

    # Strategy signal (derived from continuous function)
    regime: str          # "UNCERTAINTY", "CONCENTRATION", "STABLE"
    strategy_bias: str   # "BUY_PREMIUM", "SELL_PREMIUM", "NEUTRAL"
    signal_strength: str # "STRONG", "MODERATE", "WEAK"
    regime_score: float  # continuous: -1.0 to +1.0

    # Sector breakdown
    sector_weights: Dict[str, float]
    top_sectors: List[Tuple[str, float]]
    using_live_caps: bool  # whether live market caps were used

    # Explanation
    explanation: str
    strategy_implications: List[str]


def compute_market_entropy(lookback_days: int = 90) -> Optional[EntropySignal]:
    """
    Compute current market entropy from sector ETF data.

    Uses live market caps for current weights when available,
    falling back to price-performance-adjusted baseline weights.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 10)

    # ── Try live market cap weights first ──
    live_weights = fetch_live_sector_weights()
    using_live_caps = live_weights is not None

    # Fetch sector ETF price history in a single batch call
    etf_symbols = list(SECTOR_ETFS.keys())
    try:
        batch_data = yf.download(
            etf_symbols,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
        )
        # yf.download returns MultiIndex columns: (Price, Ticker)
        if "Close" in batch_data.columns or hasattr(batch_data.columns, 'levels'):
            try:
                prices_df = batch_data["Close"]
            except KeyError:
                prices_df = batch_data
        else:
            prices_df = batch_data
    except Exception:
        prices_df = pd.DataFrame()

    if prices_df.empty or len(prices_df.columns) < 8:
        return None

    prices_df = prices_df.dropna()
    if len(prices_df) < 30:
        return None

    etf_list = list(SECTOR_ETFS.keys())

    # ── Current weights ──
    if using_live_caps:
        adjusted_weights = live_weights
    else:
        # Fallback: adjust static weights by relative price performance
        latest_prices = prices_df.iloc[-1]
        base_prices = prices_df.iloc[0]
        relative_performance = latest_prices / base_prices
        adjusted_weights = FALLBACK_WEIGHTS.copy()
        for i, etf in enumerate(etf_list):
            if etf in relative_performance.index and i < len(adjusted_weights):
                adjusted_weights[i] *= relative_performance[etf]
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

    # ── 30-day ago weights (for trend) ──
    if len(prices_df) >= 30:
        prices_30d_ago = prices_df.iloc[-30]
        latest_prices = prices_df.iloc[-1]
        # Compute relative change over 30 days
        perf_30d = latest_prices / prices_30d_ago
        # Estimate 30-day-ago weights by reversing recent performance
        weights_30d_ago = adjusted_weights.copy()
        for i, etf in enumerate(etf_list):
            if etf in perf_30d.index and i < len(weights_30d_ago):
                if perf_30d[etf] != 0:
                    weights_30d_ago[i] /= perf_30d[etf]
        weights_30d_ago = weights_30d_ago / weights_30d_ago.sum()
        entropy_30d_ago = sector_entropy(weights_30d_ago)
    else:
        entropy_30d_ago = None

    # ── Compute entropy metrics ──
    current_entropy = sector_entropy(adjusted_weights)
    current_hhi = herfindahl_index(adjusted_weights)
    current_normalized = normalized_entropy(adjusted_weights)
    current_cr7 = concentration_ratio(adjusted_weights, 7)

    # ── Baseline entropy (equal-weight = maximum entropy for 11 sectors) ──
    baseline_entropy = max_entropy(len(SECTOR_ETFS))

    # ── Entropy change from equal-weight baseline ──
    entropy_change = current_entropy - baseline_entropy
    entropy_change_pct = (entropy_change / baseline_entropy) * 100 if baseline_entropy > 0 else 0

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

    # ── Continuous regime classification ──
    regime_score = continuous_entropy_signal(entropy_change)

    if regime_score < -0.3:
        regime = "CONCENTRATION"
        strategy_bias = "SELL_PREMIUM"
    elif regime_score > 0.3:
        regime = "UNCERTAINTY"
        strategy_bias = "BUY_PREMIUM"
    else:
        regime = "STABLE"
        strategy_bias = "NEUTRAL"

    abs_score = abs(regime_score)
    if abs_score > 0.7:
        signal_strength = "STRONG"
    elif abs_score > 0.3:
        signal_strength = "MODERATE"
    else:
        signal_strength = "WEAK"

    # ── Sector breakdown ──
    sector_weights = {}
    for i, etf in enumerate(etf_list):
        if i < len(adjusted_weights):
            sector_weights[SECTOR_ETFS[etf]] = float(adjusted_weights[i])

    top_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)[:5]

    # ── Build explanations ──
    explanation = _build_explanation(
        regime, entropy_change, entropy_change_pct, entropy_trend,
        current_entropy, current_hhi, top_sectors, using_live_caps, regime_score,
    )
    strategy_implications = _build_implications(regime, strategy_bias, entropy_trend)

    return EntropySignal(
        current_entropy=current_entropy,
        current_hhi=current_hhi,
        current_normalized=current_normalized,
        current_cr7=current_cr7,
        baseline_entropy=baseline_entropy,
        entropy_change=entropy_change,
        entropy_change_pct=entropy_change_pct,
        entropy_30d_ago=entropy_30d_ago,
        entropy_trend=entropy_trend,
        regime=regime,
        strategy_bias=strategy_bias,
        signal_strength=signal_strength,
        regime_score=regime_score,
        sector_weights=sector_weights,
        top_sectors=top_sectors,
        using_live_caps=using_live_caps,
        explanation=explanation,
        strategy_implications=strategy_implications,
    )


def _build_explanation(regime, entropy_change, entropy_change_pct, trend,
                       current_entropy, current_hhi, top_sectors,
                       using_live_caps, regime_score):
    """Build human-readable entropy explanation."""
    direction = "decreased" if entropy_change < 0 else "increased"
    top3 = ", ".join(f"{name} ({w:.1%})" for name, w in top_sectors[:3])
    source = "live market caps" if using_live_caps else "price-adjusted estimates"

    if regime == "CONCENTRATION":
        return (
            f"Market entropy has {direction} by {abs(entropy_change):.3f} bits "
            f"({abs(entropy_change_pct):.1f}%) from equal-weight baseline. "
            f"Regime score: {regime_score:.2f} (continuous). "
            f"This signals increasing concentration — fewer sectors are driving returns. "
            f"Top sectors: {top3}. "
            f"HHI at {current_hhi:.4f} confirms concentration. "
            f"Weights source: {source}. "
            f"30-day trend: {trend}."
        )
    elif regime == "UNCERTAINTY":
        return (
            f"Market entropy has {direction} by {abs(entropy_change):.3f} bits "
            f"({abs(entropy_change_pct):.1f}%) from equal-weight baseline. "
            f"Regime score: {regime_score:.2f} (continuous). "
            f"This signals increasing dispersion — more sectors are participating in returns. "
            f"Top sectors: {top3}. "
            f"Weights source: {source}. "
            f"30-day trend: {trend}."
        )
    else:
        return (
            f"Market entropy is near baseline (change: {entropy_change:+.3f} bits, "
            f"{entropy_change_pct:+.1f}%). Regime score: {regime_score:.2f} (continuous). "
            f"No strong regime signal. "
            f"Top sectors: {top3}. "
            f"Weights source: {source}. "
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
            "FAVOR: Credit spreads in the direction of concentration",
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
# Score Adjustment for Strategy Engine (Continuous)
# ─────────────────────────────────────────────

# Strategy affinities: negative = favors concentration, positive = favors uncertainty
STRATEGY_AFFINITIES = {
    "Iron Condor": -12,
    "Bull Put Spread": -10,
    "Bear Call Spread": -10,
    "Covered Call": -8,
    "Cash-Secured Put": -8,
    "Calendar Spread": -3,
    "Bull Call Spread": 3,
    "Bear Put Spread": 3,
    "Long Call": 5,
    "Long Put": 8,
    "Long Straddle": 12,
    "Long Strangle": 12,
}


def entropy_score_adjustment(strategy_name: str, signal: EntropySignal) -> float:
    """
    Return a continuous score adjustment for a strategy based on
    the current entropy regime.

    Uses tanh-based regime_score for smooth transitions instead of
    stepped thresholds. Returns values in the range [-12, +12].
    """
    if signal is None:
        return 0.0

    affinity = STRATEGY_AFFINITIES.get(strategy_name, 0)
    # regime_score is -1 to +1 (concentration to uncertainty)
    # affinity is negative for concentration-favored strategies
    # So: concentration regime (score < 0) × negative affinity = positive adjustment
    return affinity * signal.regime_score
