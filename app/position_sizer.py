"""
Position Sizer & Trade Card — Computes position sizes and generates
complete trade cards for options strategy recommendations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TradeCard:
    """Complete actionable trade card."""
    # Identity
    strategy_name: str
    symbol: str
    expiration: str
    dte: int

    # Position sizing
    portfolio_value: float
    risk_per_trade_pct: float
    max_risk_dollars: float
    recommended_contracts: int
    total_capital_required: float

    # Trade details
    legs: List[dict]
    net_cost_per_share: float
    net_cost_per_contract: float
    is_credit: bool

    # Risk/Reward
    max_loss_per_contract: float
    max_loss_total: float
    max_gain_per_contract: Optional[float]
    max_gain_total: Optional[float]
    breakevens: List[float]
    risk_reward_ratio: Optional[float]  # reward / risk

    # Exit rules
    profit_target_pct: float
    profit_target_dollars: float
    stop_loss_pct: float
    stop_loss_dollars: float

    # Kelly criterion
    kelly_fraction: Optional[float]
    kelly_contracts: Optional[int]

    # Warnings
    warnings: List[str]


def compute_position_size(
    portfolio_value: float,
    risk_per_trade_pct: float,
    max_loss_per_contract: float,
    max_gain_per_contract: Optional[float],
    is_credit: bool,
    net_cost_per_contract: float,
    estimated_pop: float = 0.55,
) -> dict:
    """
    Compute position size using portfolio risk rules and Kelly criterion.

    Args:
        portfolio_value: Total portfolio value in dollars
        risk_per_trade_pct: Max % of portfolio to risk (e.g., 0.02 for 2%)
        max_loss_per_contract: Maximum dollar loss per contract
        max_gain_per_contract: Maximum dollar gain per contract (None = unlimited)
        is_credit: Whether the trade collects credit
        net_cost_per_contract: Net cost per contract (positive = debit)
        estimated_pop: Estimated probability of profit (default 55%)

    Returns:
        Dict with contracts, kelly info, capital required, warnings
    """
    warnings = []

    # ── 2% Rule ──
    max_risk_dollars = portfolio_value * risk_per_trade_pct

    if max_loss_per_contract <= 0:
        max_loss_per_contract = abs(net_cost_per_contract)
    if max_loss_per_contract <= 0:
        return {
            "contracts": 0,
            "max_risk_dollars": max_risk_dollars,
            "kelly_fraction": None,
            "kelly_contracts": None,
            "capital_required": 0,
            "warnings": ["Cannot compute position size: max loss is zero."],
        }

    contracts_by_risk = int(max_risk_dollars / max_loss_per_contract)
    contracts_by_risk = max(contracts_by_risk, 0)

    # ── Capital constraint ──
    if is_credit:
        # For credit trades, capital required = margin = max loss per contract
        capital_per_contract = max_loss_per_contract
    else:
        # For debit trades, capital required = net debit
        capital_per_contract = abs(net_cost_per_contract)

    if capital_per_contract > 0:
        max_contracts_by_capital = int(portfolio_value * 0.20 / capital_per_contract)
    else:
        max_contracts_by_capital = contracts_by_risk

    contracts = min(contracts_by_risk, max_contracts_by_capital)

    # ── Kelly Criterion ──
    kelly_fraction = None
    kelly_contracts = None

    if max_gain_per_contract is not None and max_gain_per_contract > 0:
        b = max_gain_per_contract / max_loss_per_contract  # odds ratio
        p = estimated_pop
        q = 1 - p

        kelly_full = (b * p - q) / b if b > 0 else 0
        kelly_half = kelly_full / 2  # half-Kelly for safety

        if kelly_half > 0:
            kelly_fraction = kelly_half
            kelly_dollars = portfolio_value * kelly_half
            kelly_contracts = max(int(kelly_dollars / max_loss_per_contract), 0)
        else:
            kelly_fraction = kelly_full  # may be negative (no edge)
            kelly_contracts = 0
            if kelly_full < 0:
                warnings.append(
                    f"Kelly criterion is negative ({kelly_full:.3f}) — "
                    f"risk/reward may not justify this trade at estimated "
                    f"{estimated_pop:.0%} probability of profit."
                )

    # ── Warnings ──
    if contracts == 0:
        warnings.append(
            f"Position size is 0 contracts. Your max risk "
            f"(${max_risk_dollars:,.0f}) is less than the max loss per "
            f"contract (${max_loss_per_contract:,.0f}). Consider a smaller "
            f"spread width or a larger portfolio allocation."
        )

    total_capital = contracts * capital_per_contract
    if total_capital > portfolio_value * 0.10:
        warnings.append(
            f"This position uses {total_capital / portfolio_value:.1%} of "
            f"your portfolio. Consider reducing to stay under 10% per position."
        )

    if contracts > 10:
        warnings.append(
            f"Large position ({contracts} contracts). For beginners, "
            f"consider starting with 1-3 contracts to limit risk."
        )

    return {
        "contracts": contracts,
        "max_risk_dollars": max_risk_dollars,
        "kelly_fraction": kelly_fraction,
        "kelly_contracts": kelly_contracts,
        "capital_required": total_capital,
        "warnings": warnings,
    }


def generate_trade_card(
    strategy_name: str,
    symbol: str,
    expiration: str,
    dte: int,
    spot: float,
    portfolio_value: float,
    risk_per_trade_pct: float,
    concrete_legs: List[dict],
    net_cost: float,
    max_loss_dollars: float,
    max_gain_dollars: Optional[float],
    breakevens: List[float],
) -> TradeCard:
    """
    Generate a complete trade card for a strategy recommendation.
    """
    is_credit = net_cost < 0
    net_cost_per_contract = abs(net_cost) * 100

    # Position sizing
    sizing = compute_position_size(
        portfolio_value=portfolio_value,
        risk_per_trade_pct=risk_per_trade_pct,
        max_loss_per_contract=max_loss_dollars,
        max_gain_per_contract=max_gain_dollars,
        is_credit=is_credit,
        net_cost_per_contract=net_cost_per_contract if not is_credit else -net_cost_per_contract,
    )

    contracts = sizing["contracts"]

    # Risk/reward ratio
    if max_gain_dollars is not None and max_loss_dollars > 0:
        risk_reward = max_gain_dollars / max_loss_dollars
    else:
        risk_reward = None  # unlimited upside

    # Exit rules
    if is_credit:
        # Credit trades: close at 50% of max profit, stop at 2x credit received
        profit_target_pct = 0.50
        profit_target_dollars = (max_gain_dollars or net_cost_per_contract) * profit_target_pct
        stop_loss_pct = 1.0  # 100% of max loss
        stop_loss_dollars = max_loss_dollars
    else:
        # Debit trades: close at 50% profit, stop at 50% loss of premium
        profit_target_pct = 0.50
        if max_gain_dollars is not None:
            profit_target_dollars = max_gain_dollars * profit_target_pct
        else:
            profit_target_dollars = net_cost_per_contract  # 100% return
        stop_loss_pct = 0.50
        stop_loss_dollars = net_cost_per_contract * stop_loss_pct

    return TradeCard(
        strategy_name=strategy_name,
        symbol=symbol,
        expiration=expiration,
        dte=dte,
        portfolio_value=portfolio_value,
        risk_per_trade_pct=risk_per_trade_pct,
        max_risk_dollars=sizing["max_risk_dollars"],
        recommended_contracts=contracts,
        total_capital_required=sizing["capital_required"],
        legs=concrete_legs,
        net_cost_per_share=abs(net_cost),
        net_cost_per_contract=net_cost_per_contract,
        is_credit=is_credit,
        max_loss_per_contract=max_loss_dollars,
        max_loss_total=max_loss_dollars * max(contracts, 1),
        max_gain_per_contract=max_gain_dollars,
        max_gain_total=max_gain_dollars * max(contracts, 1) if max_gain_dollars else None,
        breakevens=breakevens,
        risk_reward_ratio=risk_reward,
        profit_target_pct=profit_target_pct,
        profit_target_dollars=profit_target_dollars,
        stop_loss_pct=stop_loss_pct,
        stop_loss_dollars=stop_loss_dollars,
        kelly_fraction=sizing["kelly_fraction"],
        kelly_contracts=sizing["kelly_contracts"],
        warnings=sizing["warnings"],
    )
