"""
Strategy Engine — Decision engine for OptiMax.

Maps (outlook × IV environment × timeframe × risk tolerance) to ranked
options strategy recommendations with concrete trade construction.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# Strategy Definitions
# ─────────────────────────────────────────────

@dataclass
class StrategyLeg:
    """A single leg of an options strategy."""
    option_type: str  # "call" or "put"
    action: str       # "buy" or "sell"
    strike_offset: float  # relative to spot: 0 = ATM, +5 = $5 OTM call / ITM put
    quantity: int = 1


@dataclass
class StrategyDefinition:
    """Complete definition of an options strategy."""
    name: str
    description: str
    legs: List[StrategyLeg]
    outlook: List[str]        # ["bullish", "neutral", "bearish", "volatile"]
    iv_preference: str        # "high", "low", "any"
    risk_type: str            # "defined" or "undefined"
    ideal_dte_min: int        # minimum ideal DTE
    ideal_dte_max: int        # maximum ideal DTE
    complexity: int           # 1=beginner, 2=intermediate, 3=advanced
    max_loss_desc: str
    max_gain_desc: str
    greeks_profile: str       # human-readable Greeks summary
    why_it_works: str         # explanation for beginners
    score_boost: float = 0.0  # extra score for special conditions


# All strategy definitions
STRATEGIES = [
    StrategyDefinition(
        name="Long Call",
        description="Buy a call option to profit from upside movement.",
        legs=[StrategyLeg("call", "buy", 0)],
        outlook=["bullish"],
        iv_preference="low",
        risk_type="defined",
        ideal_dte_min=30,
        ideal_dte_max=90,
        complexity=1,
        max_loss_desc="Premium paid",
        max_gain_desc="Unlimited",
        greeks_profile="Long delta, long gamma, long vega, short theta",
        why_it_works="You own the right to buy shares at a fixed price. If the stock rises above your strike + premium, you profit. Best when IV is low because you're buying cheap optionality.",
    ),
    StrategyDefinition(
        name="Long Put",
        description="Buy a put option to profit from downside movement.",
        legs=[StrategyLeg("put", "buy", 0)],
        outlook=["bearish"],
        iv_preference="low",
        risk_type="defined",
        ideal_dte_min=30,
        ideal_dte_max=60,
        complexity=1,
        max_loss_desc="Premium paid",
        max_gain_desc="Strike - premium (stock to zero)",
        greeks_profile="Short delta, long gamma, long vega, short theta",
        why_it_works="You own the right to sell shares at a fixed price. If the stock drops below your strike - premium, you profit. Best when IV is low so puts are cheap.",
    ),
    StrategyDefinition(
        name="Bull Call Spread",
        description="Buy a lower-strike call and sell a higher-strike call. Reduces cost vs. a long call by capping upside.",
        legs=[
            StrategyLeg("call", "buy", 0),
            StrategyLeg("call", "sell", 10),
        ],
        outlook=["bullish"],
        iv_preference="high",
        risk_type="defined",
        ideal_dte_min=30,
        ideal_dte_max=60,
        complexity=1,
        max_loss_desc="Net debit paid",
        max_gain_desc="Strike width - net debit",
        greeks_profile="Long delta (reduced), modest vega exposure",
        why_it_works="When IV is high, outright calls are expensive. The short call offsets much of the cost. You give up unlimited upside but pay far less. Works well when you expect a moderate move, not a moonshot.",
    ),
    StrategyDefinition(
        name="Bear Put Spread",
        description="Buy a higher-strike put and sell a lower-strike put. Cheaper than a long put, but caps the downside profit.",
        legs=[
            StrategyLeg("put", "buy", 0),
            StrategyLeg("put", "sell", -10),
        ],
        outlook=["bearish"],
        iv_preference="high",
        risk_type="defined",
        ideal_dte_min=30,
        ideal_dte_max=60,
        complexity=1,
        max_loss_desc="Net debit paid",
        max_gain_desc="Strike width - net debit",
        greeks_profile="Short delta (reduced), modest vega exposure",
        why_it_works="When IV is high, outright puts are expensive. The short put reduces cost. You profit from a moderate decline without paying full put premium.",
    ),
    StrategyDefinition(
        name="Bull Put Spread",
        description="Sell a higher-strike put and buy a lower-strike put. Collect credit, profit if stock stays above short strike.",
        legs=[
            StrategyLeg("put", "sell", -5),
            StrategyLeg("put", "buy", -15),
        ],
        outlook=["bullish"],
        iv_preference="high",
        risk_type="defined",
        ideal_dte_min=20,
        ideal_dte_max=45,
        complexity=2,
        max_loss_desc="Strike width - credit received",
        max_gain_desc="Credit received",
        greeks_profile="Mildly long delta, short vega, positive theta",
        why_it_works="You collect premium upfront. If the stock stays above your short put strike by expiration, you keep the full credit. High IV means you collect MORE premium. Time decay (theta) works in your favor every day.",
        score_boost=5.0,  # very popular strategy, good risk/reward
    ),
    StrategyDefinition(
        name="Bear Call Spread",
        description="Sell a lower-strike call and buy a higher-strike call. Collect credit, profit if stock stays below short strike.",
        legs=[
            StrategyLeg("call", "sell", 5),
            StrategyLeg("call", "buy", 15),
        ],
        outlook=["bearish"],
        iv_preference="high",
        risk_type="defined",
        ideal_dte_min=20,
        ideal_dte_max=45,
        complexity=2,
        max_loss_desc="Strike width - credit received",
        max_gain_desc="Credit received",
        greeks_profile="Mildly short delta, short vega, positive theta",
        why_it_works="Mirror of the bull put spread for bearish views. You collect premium and profit if the stock stays below your short call. High IV means more premium collected.",
    ),
    StrategyDefinition(
        name="Cash-Secured Put",
        description="Sell a put and hold cash to cover assignment. Collect premium, willing to buy shares at strike.",
        legs=[StrategyLeg("put", "sell", -5)],
        outlook=["bullish"],
        iv_preference="high",
        risk_type="undefined",
        ideal_dte_min=20,
        ideal_dte_max=45,
        complexity=2,
        max_loss_desc="Strike price x 100 - premium (if stock goes to zero)",
        max_gain_desc="Premium received",
        greeks_profile="Long delta, short vega, positive theta",
        why_it_works="You get paid to agree to buy a stock you like at a lower price. If the stock stays above your strike, you keep the premium. If assigned, you own shares at an effective price below market. Best when IV is high so premium is rich.",
    ),
    StrategyDefinition(
        name="Covered Call",
        description="Own 100 shares and sell a call against them. Generate income, cap upside.",
        legs=[StrategyLeg("call", "sell", 5)],
        outlook=["bullish", "neutral"],
        iv_preference="high",
        risk_type="undefined",
        ideal_dte_min=20,
        ideal_dte_max=45,
        complexity=1,
        max_loss_desc="Stock drops to zero (minus premium received)",
        max_gain_desc="(Strike - current price) + premium",
        greeks_profile="Mildly long delta (from stock), short vega, positive theta",
        why_it_works="If you already own shares, selling calls generates income. The premium lowers your cost basis. You give up upside above the strike, but collect cash every month. Best when IV is high so you collect more.",
    ),
    StrategyDefinition(
        name="Iron Condor",
        description="Sell a bull put spread + bear call spread. Profit if stock stays in a range.",
        legs=[
            StrategyLeg("put", "sell", -5),
            StrategyLeg("put", "buy", -15),
            StrategyLeg("call", "sell", 5),
            StrategyLeg("call", "buy", 15),
        ],
        outlook=["neutral"],
        iv_preference="high",
        risk_type="defined",
        ideal_dte_min=30,
        ideal_dte_max=45,
        complexity=2,
        max_loss_desc="Width of wider wing - total credit",
        max_gain_desc="Total credit received",
        greeks_profile="Near-zero delta, short vega, positive theta",
        why_it_works="You collect premium on both sides. As long as the stock stays between your short strikes, you keep the credit. High IV is crucial because (1) you collect more premium and (2) IV contraction increases your profit. Time decay helps every day.",
        score_boost=5.0,
    ),
    StrategyDefinition(
        name="Long Straddle",
        description="Buy a call and a put at the same strike. Profit from a big move in either direction.",
        legs=[
            StrategyLeg("call", "buy", 0),
            StrategyLeg("put", "buy", 0),
        ],
        outlook=["volatile"],
        iv_preference="low",
        risk_type="defined",
        ideal_dte_min=14,
        ideal_dte_max=45,
        complexity=2,
        max_loss_desc="Total premium paid (both options)",
        max_gain_desc="Unlimited (either direction)",
        greeks_profile="Near-zero delta, very long gamma, very long vega, very short theta",
        why_it_works="You profit from a BIG move in either direction. The stock must move more than the total premium paid. Best when IV is LOW because you're buying pure volatility — any IV expansion also helps. Time decay is aggressive, so you need the move to happen before expiration.",
    ),
    StrategyDefinition(
        name="Long Strangle",
        description="Buy an OTM call and an OTM put. Cheaper than a straddle, needs a bigger move.",
        legs=[
            StrategyLeg("call", "buy", 5),
            StrategyLeg("put", "buy", -5),
        ],
        outlook=["volatile"],
        iv_preference="low",
        risk_type="defined",
        ideal_dte_min=14,
        ideal_dte_max=45,
        complexity=2,
        max_loss_desc="Total premium paid",
        max_gain_desc="Unlimited (either direction)",
        greeks_profile="Near-zero delta, long gamma, long vega, short theta",
        why_it_works="Like a straddle but cheaper because both options are OTM. The tradeoff: you need a bigger move to profit. Best when IV is low and you expect a catalyst (earnings, FDA, macro event) to cause a large move.",
    ),
    StrategyDefinition(
        name="Calendar Spread",
        description="Sell a near-term option and buy a longer-term option at the same strike. Profit from time decay differential.",
        legs=[
            StrategyLeg("call", "sell", 0),  # near-term
            StrategyLeg("call", "buy", 0),   # far-term (conceptual)
        ],
        outlook=["neutral"],
        iv_preference="low",
        risk_type="defined",
        ideal_dte_min=45,
        ideal_dte_max=90,
        complexity=3,
        max_loss_desc="Net debit paid",
        max_gain_desc="Occurs when stock is at strike at front-month expiration",
        greeks_profile="Near-zero delta, long vega (net), positive theta",
        why_it_works="The near-term option decays faster than the far-term option you own. If the stock stays near the strike, the front option expires worthless while your back option retains value. Benefits from IV expansion.",
    ),
]


# ─────────────────────────────────────────────
# Decision Engine
# ─────────────────────────────────────────────

@dataclass
class StrategyRecommendation:
    """A scored strategy recommendation with concrete trade details."""
    strategy: StrategyDefinition
    score: float
    rank: int
    concrete_legs: List[dict]  # [{type, action, strike, estimated_price}, ...]
    net_cost: float            # positive = debit, negative = credit
    max_loss_dollars: float
    max_gain_dollars: Optional[float]  # None = unlimited
    breakeven: List[float]
    explanation: str
    entropy_adjustment: float = 0.0


def recommend_strategies(
    outlook: str,
    iv_percentile: float,
    dte: int,
    risk_tolerance: str = "conservative",
    spot: float = 100.0,
    chain_df: pd.DataFrame = None,
    entropy_adjustment_fn=None,
) -> List[StrategyRecommendation]:
    """
    Core decision engine.

    Args:
        outlook: "bullish", "bearish", "neutral", "volatile"
        iv_percentile: 0-100 (from compute_iv_percentile)
        dte: days to expiration for the selected chain
        risk_tolerance: "conservative", "moderate", "aggressive"
        spot: current stock price
        chain_df: enriched options chain DataFrame (for concrete pricing)
        entropy_adjustment_fn: optional callable(strategy_name) -> float score adjustment

    Returns:
        List of StrategyRecommendation sorted by score (best first)
    """
    iv_label = "high" if iv_percentile >= 50 else "low"

    scored = []

    for strat in STRATEGIES:
        score = 0.0
        reasons = []

        # ── Outlook match (most important) ──
        if outlook in strat.outlook:
            score += 40
            reasons.append(f"Matches your {outlook} outlook")
        else:
            continue  # hard filter: skip strategies that don't match outlook

        # ── IV alignment ──
        if strat.iv_preference == iv_label:
            score += 25
            if iv_label == "high":
                reasons.append(f"IV at {iv_percentile:.0f}th percentile — this strategy SELLS expensive premium")
            else:
                reasons.append(f"IV at {iv_percentile:.0f}th percentile — this strategy BUYS cheap optionality")
        elif strat.iv_preference == "any":
            score += 15
            reasons.append("Works in any IV environment")
        else:
            score += 5
            if iv_label == "high" and strat.iv_preference == "low":
                reasons.append(f"IV is elevated ({iv_percentile:.0f}th %ile) — this strategy prefers low IV, consider alternatives")
            else:
                reasons.append(f"IV is low ({iv_percentile:.0f}th %ile) — this strategy prefers high IV, may collect less premium")

        # ── DTE alignment ──
        if strat.ideal_dte_min <= dte <= strat.ideal_dte_max:
            score += 15
            reasons.append(f"{dte} DTE is in the ideal range ({strat.ideal_dte_min}-{strat.ideal_dte_max}d)")
        elif abs(dte - strat.ideal_dte_min) <= 10 or abs(dte - strat.ideal_dte_max) <= 10:
            score += 8
            reasons.append(f"{dte} DTE is close to ideal range ({strat.ideal_dte_min}-{strat.ideal_dte_max}d)")
        else:
            score += 2
            reasons.append(f"{dte} DTE is outside ideal range ({strat.ideal_dte_min}-{strat.ideal_dte_max}d)")

        # ── Risk tolerance ──
        if risk_tolerance == "conservative" and strat.risk_type == "undefined":
            score -= 15
            reasons.append("Undefined risk — not ideal for conservative sizing")
        elif risk_tolerance == "aggressive" and strat.risk_type == "defined":
            score += 5  # defined risk is always fine
        if strat.risk_type == "defined":
            score += 5
            reasons.append("Defined risk — you know your max loss upfront")

        # ── Complexity (beginners prefer simpler) ──
        if strat.complexity == 1:
            score += 10
            reasons.append("Simple to understand and manage")
        elif strat.complexity == 2:
            score += 5
        elif strat.complexity == 3:
            score += 0
            reasons.append("More complex — requires understanding of term structure")

        # ── Bonus ──
        score += strat.score_boost

        # ── Entropy adjustment ──
        entropy_adj = 0.0
        if entropy_adjustment_fn is not None:
            entropy_adj = entropy_adjustment_fn(strat.name)
            score += entropy_adj
            if abs(entropy_adj) >= 3:
                direction = "boosted" if entropy_adj > 0 else "penalized"
                reasons.append(f"Entropy regime {direction} this strategy ({entropy_adj:+.0f} pts)")

        # ── Build concrete legs ──
        concrete_legs, net_cost, max_loss, max_gain, breakevens = _build_concrete_trade(
            strat, spot, chain_df, dte
        )

        explanation = " | ".join(reasons)

        scored.append(StrategyRecommendation(
            strategy=strat,
            score=score,
            rank=0,
            concrete_legs=concrete_legs,
            net_cost=net_cost,
            max_loss_dollars=max_loss,
            max_gain_dollars=max_gain,
            breakeven=breakevens,
            explanation=explanation,
            entropy_adjustment=entropy_adj,
        ))

    # Sort by score descending
    scored.sort(key=lambda r: r.score, reverse=True)
    for i, rec in enumerate(scored):
        rec.rank = i + 1

    return scored


def _build_concrete_trade(strat, spot, chain_df, dte):
    """
    Build concrete trade details using the actual options chain.

    Returns:
        concrete_legs, net_cost, max_loss_dollars, max_gain_dollars, breakevens
    """
    concrete_legs = []
    net_cost = 0.0

    for leg in strat.legs:
        strike = round(spot + leg.strike_offset)

        # Try to find the actual contract in the chain
        estimated_price = _estimate_leg_price(
            chain_df, leg.option_type, strike, leg.action
        )

        concrete_legs.append({
            "type": leg.option_type,
            "action": leg.action,
            "strike": strike,
            "quantity": leg.quantity,
            "estimated_price": estimated_price,
        })

        if leg.action == "buy":
            net_cost += estimated_price * leg.quantity
        else:
            net_cost -= estimated_price * leg.quantity

    # Compute max loss, max gain, breakevens based on strategy type
    max_loss, max_gain, breakevens = _compute_risk_reward(
        strat, concrete_legs, net_cost, spot
    )

    return concrete_legs, net_cost, max_loss, max_gain, breakevens


def _estimate_leg_price(chain_df, option_type, target_strike, action):
    """Find the closest matching contract and return its mid price."""
    if chain_df is None or chain_df.empty:
        return 3.00  # fallback estimate

    subset = chain_df[chain_df["optionType"] == option_type]
    if subset.empty:
        return 3.00

    # Find closest strike
    idx = (subset["strike"] - target_strike).abs().idxmin()
    row = subset.loc[idx]

    bid = row.get("bid", 0) or 0
    ask = row.get("ask", 0) or 0

    if bid > 0 and ask > 0:
        if action == "buy":
            return ask  # pay the ask
        else:
            return bid  # receive the bid
    elif row.get("lastPrice", 0) > 0:
        return row["lastPrice"]
    else:
        return row.get("midPrice", 3.00) or 3.00


def _compute_risk_reward(strat, concrete_legs, net_cost, spot):
    """Compute max loss, max gain, and breakeven(s) for a strategy."""
    name = strat.name
    strikes = [leg["strike"] for leg in concrete_legs]

    if name == "Long Call":
        max_loss = abs(net_cost) * 100
        max_gain = None  # unlimited
        breakevens = [strikes[0] + abs(net_cost)]

    elif name == "Long Put":
        max_loss = abs(net_cost) * 100
        max_gain = (strikes[0] - abs(net_cost)) * 100
        breakevens = [strikes[0] - abs(net_cost)]

    elif name == "Bull Call Spread":
        width = strikes[1] - strikes[0]
        debit = abs(net_cost)
        max_loss = debit * 100
        max_gain = (width - debit) * 100
        breakevens = [strikes[0] + debit]

    elif name == "Bear Put Spread":
        width = strikes[0] - strikes[1]
        debit = abs(net_cost)
        max_loss = debit * 100
        max_gain = (width - debit) * 100
        breakevens = [strikes[0] - debit]

    elif name == "Bull Put Spread":
        width = strikes[0] - strikes[1]
        credit = abs(net_cost)
        max_loss = (width - credit) * 100
        max_gain = credit * 100
        breakevens = [strikes[0] - credit]

    elif name == "Bear Call Spread":
        width = strikes[1] - strikes[0]
        credit = abs(net_cost)
        max_loss = (width - credit) * 100
        max_gain = credit * 100
        breakevens = [strikes[0] + credit]

    elif name == "Cash-Secured Put":
        credit = abs(net_cost)
        max_loss = (strikes[0] - credit) * 100
        max_gain = credit * 100
        breakevens = [strikes[0] - credit]

    elif name == "Covered Call":
        credit = abs(net_cost)
        max_loss = (spot - credit) * 100  # stock to zero
        max_gain = (strikes[0] - spot + credit) * 100
        breakevens = [spot - credit]

    elif name == "Iron Condor":
        # legs: sell put, buy put, sell call, buy call
        put_width = strikes[0] - strikes[1]
        call_width = strikes[3] - strikes[2]
        total_credit = abs(net_cost)
        max_width = max(put_width, call_width)
        max_loss = (max_width - total_credit) * 100
        max_gain = total_credit * 100
        breakevens = [strikes[0] - total_credit, strikes[2] + total_credit]

    elif name == "Long Straddle":
        total_premium = abs(net_cost)
        max_loss = total_premium * 100
        max_gain = None  # unlimited
        breakevens = [strikes[0] - total_premium, strikes[0] + total_premium]

    elif name == "Long Strangle":
        total_premium = abs(net_cost)
        max_loss = total_premium * 100
        max_gain = None
        breakevens = [strikes[1] - total_premium, strikes[0] + total_premium]

    elif name == "Calendar Spread":
        debit = abs(net_cost)
        max_loss = debit * 100
        max_gain = debit * 100 * 0.5  # approximate
        breakevens = [strikes[0]]  # approximately ATM

    else:
        max_loss = abs(net_cost) * 100
        max_gain = abs(net_cost) * 100
        breakevens = [spot]

    return max_loss, max_gain, breakevens


# ─────────────────────────────────────────────
# Payoff Computation
# ─────────────────────────────────────────────

def compute_strategy_payoff(concrete_legs, net_cost, spot, price_range=None):
    """
    Compute the payoff at expiration for any multi-leg strategy.

    Returns:
        price_range (array), payoff (array, per-share), payoff_dollars (array, per-contract)
    """
    if price_range is None:
        price_range = np.linspace(spot * 0.70, spot * 1.30, 300)

    payoff = np.zeros_like(price_range)

    for leg in concrete_legs:
        k = leg["strike"]
        price = leg["estimated_price"]
        qty = leg["quantity"]

        if leg["type"] == "call":
            intrinsic = np.maximum(price_range - k, 0)
        else:
            intrinsic = np.maximum(k - price_range, 0)

        if leg["action"] == "buy":
            payoff += (intrinsic - price) * qty
        else:
            payoff += (price - intrinsic) * qty

    payoff_dollars = payoff * 100  # per contract

    return price_range, payoff, payoff_dollars
