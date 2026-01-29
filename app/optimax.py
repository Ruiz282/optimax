"""
OptiMax — Options Strategy Recommender + Portfolio Manager
All Sections: Data Engine + Strategy Recommender + Entropy Signals + Trade Cards + Portfolio

Run with:  streamlit run app/optimax.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
from functools import partial

from options_engine import (
    get_ticker_info,
    get_expirations,
    get_enriched_chain,
    compute_iv_percentile,
)
from strategy_engine import (
    recommend_strategies,
    compute_strategy_payoff,
)
from entropy_bridge import (
    compute_market_entropy,
    entropy_score_adjustment,
)
from position_sizer import generate_trade_card
from portfolio_manager import (
    fetch_security_data,
    create_holding,
    calculate_portfolio_summary,
    build_dividend_calendar,
    build_combined_calendar,
    get_annual_dividend_projection,
    search_tickers,
    get_company_color,
    get_stock_performance,
    get_portfolio_performance,
    POPULAR_STOCKS,
    POPULAR_ETFS,
    POPULAR_BOND_ETFS,
    POPULAR_REITS,
    TICKER_DATABASE,
    FedEvent,
)

# Set dark theme for matplotlib charts
plt.style.use('dark_background')
CHART_BG_COLOR = '#2E2E2E'
CHART_FACE_COLOR = '#3A3A3A'
from dateutil.relativedelta import relativedelta
import calendar as cal_module

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="OptiMax",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Sidebar — Inputs
# ─────────────────────────────────────────────

st.sidebar.title("OptiMax")
st.sidebar.caption("Options & Portfolio Manager")
st.sidebar.markdown("---")

symbol = st.sidebar.text_input(
    "Ticker Symbol",
    value="AAPL",
    max_chars=10,
    help="Enter any US stock ticker (e.g. AAPL, NVDA, SPY)",
).upper().strip()

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
    step=0.1,
    help="Current Treasury rate (used for BS pricing)",
) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Your Market View")

outlook = st.sidebar.radio(
    "Outlook",
    ["Bullish", "Bearish", "Neutral", "Volatile (big move, unsure direction)"],
    index=0,
    help="What do you think this stock will do?",
)
outlook_map = {
    "Bullish": "bullish",
    "Bearish": "bearish",
    "Neutral": "neutral",
    "Volatile (big move, unsure direction)": "volatile",
}
outlook_key = outlook_map[outlook]

risk_tolerance = st.sidebar.radio(
    "Risk Tolerance",
    ["Conservative", "Moderate", "Aggressive"],
    index=0,
    help="Conservative = defined-risk only. Aggressive = all strategies.",
)
risk_tol_key = risk_tolerance.lower()

st.sidebar.markdown("---")
st.sidebar.subheader("Position Sizing")

portfolio_value = st.sidebar.number_input(
    "Portfolio Value ($)",
    min_value=1000,
    max_value=10_000_000,
    value=25_000,
    step=1000,
    help="Your total trading portfolio size",
)

risk_per_trade = st.sidebar.slider(
    "Max Risk per Trade (%)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="2% is standard. 1% is conservative. 5% is aggressive.",
) / 100

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────

if not symbol:
    st.warning("Enter a ticker symbol in the sidebar.")
    st.stop()

try:
    with st.spinner(f"Fetching data for {symbol}..."):
        info = get_ticker_info(symbol)
        expirations = get_expirations(symbol)
except Exception as e:
    st.error(f"Could not fetch data for **{symbol}**: {e}")
    st.stop()

if not expirations:
    st.error(f"No options available for **{symbol}**.")
    st.stop()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

spot = info["spot"]

col_title, col_price = st.columns([3, 1])
with col_title:
    st.title(f"{info['name']} ({symbol})")
with col_price:
    st.metric("Spot Price", f"${spot:,.2f}")

# ─────────────────────────────────────────────
# IV Percentile Banner
# ─────────────────────────────────────────────

with st.spinner("Computing IV percentile..."):
    iv_data = compute_iv_percentile(symbol)

iv_percentile = 50.0
if iv_data:
    iv_percentile = iv_data["iv_percentile"]

    iv_cols = st.columns(5)
    with iv_cols[0]:
        color = (
            "inverse" if iv_data["iv_rank_label"] == "HIGH"
            else "off" if iv_data["iv_rank_label"] == "LOW"
            else "normal"
        )
        st.metric("IV Percentile", f"{iv_percentile:.0f}%",
                   delta=iv_data["iv_rank_label"], delta_color=color)
    with iv_cols[1]:
        st.metric("Current ATM IV", f"{iv_data['current_iv']:.1%}")
    with iv_cols[2]:
        st.metric("30d Realized Vol", f"{iv_data['rv_30']:.1%}")
    with iv_cols[3]:
        st.metric("60d Realized Vol", f"{iv_data['rv_60']:.1%}")
    with iv_cols[4]:
        st.metric("90d Realized Vol", f"{iv_data['rv_90']:.1%}")

    if iv_data["iv_rank_label"] == "HIGH":
        st.info(
            "**IV is elevated** — Options are expensive. "
            "Favor strategies that **sell premium**: credit spreads, iron condors, covered calls."
        )
    elif iv_data["iv_rank_label"] == "LOW":
        st.info(
            "**IV is depressed** — Options are cheap. "
            "Favor strategies that **buy premium**: long calls/puts, straddles, debit spreads."
        )
    else:
        st.info(
            "**IV is moderate** — No strong edge from volatility alone. "
            "Focus on your directional view."
        )

# ─────────────────────────────────────────────
# Entropy Signal
# ─────────────────────────────────────────────

with st.spinner("Computing market entropy..."):
    entropy_signal = compute_market_entropy(lookback_days=90)

if entropy_signal:
    e_cols = st.columns(4)
    with e_cols[0]:
        regime_colors = {"CONCENTRATION": "inverse", "UNCERTAINTY": "off", "STABLE": "normal"}
        st.metric(
            "Market Regime",
            entropy_signal.regime,
            delta=f"{entropy_signal.entropy_change:+.3f} bits",
            delta_color=regime_colors.get(entropy_signal.regime, "normal"),
        )
    with e_cols[1]:
        st.metric("Entropy", f"{entropy_signal.current_entropy:.3f} bits",
                   delta=f"Trend: {entropy_signal.entropy_trend}")
    with e_cols[2]:
        st.metric("HHI", f"{entropy_signal.current_hhi:.4f}",
                   help="Herfindahl Index. Higher = more concentrated.")
    with e_cols[3]:
        st.metric("Strategy Bias", entropy_signal.strategy_bias.replace("_", " "),
                   delta=entropy_signal.signal_strength, delta_color="normal")

st.markdown("---")

# ─────────────────────────────────────────────
# Expiration Selector
# ─────────────────────────────────────────────

exp_with_dte = []
for exp in expirations:
    dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
    exp_with_dte.append(f"{exp}  ({dte}d)")

selected_idx = st.selectbox(
    "Expiration Date",
    range(len(expirations)),
    format_func=lambda i: exp_with_dte[i],
    index=min(2, len(expirations) - 1),
)
selected_exp = expirations[selected_idx]
selected_dte = (datetime.strptime(selected_exp, "%Y-%m-%d") - datetime.now()).days

# ─────────────────────────────────────────────
# Fetch Chain
# ─────────────────────────────────────────────

with st.spinner(f"Loading options chain for {selected_exp}..."):
    chain_df, _ = get_enriched_chain(symbol, selected_exp, risk_free_rate)

# ─────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────

tab_portfolio, tab_recommend, tab_chain, tab_greeks, tab_entropy = st.tabs([
    "Portfolio Manager",
    "Strategy Recommender",
    "Options Chain",
    "Greeks & IV",
    "Entropy Analysis",
])

# ═════════════════════════════════════════════
# TAB 0: Portfolio Manager
# ═════════════════════════════════════════════

with tab_portfolio:
    st.subheader("Portfolio Manager")
    st.caption("Track your stocks, ETFs, bonds, and REITs with live prices and dividend yields.")

    # Initialize session state for holdings
    if "holdings" not in st.session_state:
        st.session_state.holdings = []
    if "editing_holding" not in st.session_state:
        st.session_state.editing_holding = None

    # ── Add New Holdings ──
    st.markdown("### Add Holdings")

    add_method = st.radio(
        "Add method",
        ["Single Entry", "Bulk Add (Multiple)", "Quick Add Popular"],
        horizontal=True,
        key="add_method",
    )

    if add_method == "Single Entry":
        # Simple single ticker entry without dropdown
        col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])

        with col1:
            new_symbol = st.text_input(
                "Ticker Symbol",
                placeholder="AAPL",
                key="single_ticker_input",
                help="Enter ticker symbol (e.g., AAPL, MSFT, SPY)"
            ).upper().strip()

        with col2:
            new_shares = st.number_input("Shares", min_value=0.0, value=10.0, step=1.0, key="new_shares")

        with col3:
            new_cost = st.number_input("Avg Cost ($)", min_value=0.0, value=0.0, step=1.0, key="new_cost",
                                        help="Leave at 0 to use current price")

        with col4:
            new_date = st.date_input("Purchase Date", value=datetime.now().date(), key="new_date",
                                      help="Date you purchased this holding")

        with col5:
            st.markdown("<br>", unsafe_allow_html=True)
            add_btn = st.button("Add", type="primary", key="add_btn")

        if add_btn and new_symbol:
            with st.spinner(f"Fetching data for {new_symbol}..."):
                data = fetch_security_data(new_symbol)
                if data:
                    cost = new_cost if new_cost > 0 else data["current_price"]
                    purchase_dt = datetime.combine(new_date, datetime.min.time())
                    holding = create_holding(new_symbol, new_shares, cost, purchase_dt)
                    if holding:
                        # Check if already exists - update instead
                        existing_idx = next(
                            (i for i, h in enumerate(st.session_state.holdings) if h.symbol == new_symbol),
                            None
                        )
                        if existing_idx is not None:
                            st.session_state.holdings[existing_idx] = holding
                            st.success(f"Updated {new_symbol} in portfolio.")
                        else:
                            st.session_state.holdings.append(holding)
                            st.success(f"Added {holding.name} ({holding.asset_type}) to portfolio.")
                        st.rerun()
                else:
                    st.error(f"Could not find data for {new_symbol}. Please check the symbol.")

    elif add_method == "Bulk Add (Multiple)":
        # Bulk add multiple tickers at once
        st.markdown("""
        **Enter multiple holdings** - one per line in format: `TICKER, SHARES, COST, DATE`

        Example:
        ```
        AAPL, 50, 175.00, 2023-06-15
        MSFT, 30, 380.00, 2024-01-10
        SPY, 100
        VTI, 25, 250.00
        ```
        *Cost and Date are optional - leave blank to use current price/today*
        """)

        bulk_input = st.text_area(
            "Holdings (TICKER, SHARES, COST, DATE)",
            height=150,
            placeholder="AAPL, 50, 175.00, 2023-06-15\nMSFT, 30, 380.00, 2024-01-10\nSPY, 100\nVTI, 25",
            key="bulk_input"
        )

        if st.button("Add All Holdings", type="primary", key="bulk_add_btn"):
            if bulk_input.strip():
                lines = [line.strip() for line in bulk_input.strip().split("\n") if line.strip()]
                success_count = 0
                errors = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, line in enumerate(lines):
                    try:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) < 2:
                            errors.append(f"Line '{line}': Need at least TICKER and SHARES")
                            continue

                        sym = parts[0].upper()
                        shares = float(parts[1])
                        cost = float(parts[2]) if len(parts) > 2 and parts[2] else 0

                        # Parse date if provided
                        purchase_dt = None
                        if len(parts) > 3 and parts[3]:
                            try:
                                purchase_dt = datetime.strptime(parts[3].strip(), "%Y-%m-%d")
                            except ValueError:
                                purchase_dt = datetime.now()
                        else:
                            purchase_dt = datetime.now()

                        status_text.text(f"Processing {sym}...")
                        data = fetch_security_data(sym)

                        if data:
                            final_cost = cost if cost > 0 else data["current_price"]
                            holding = create_holding(sym, shares, final_cost, purchase_dt)
                            if holding:
                                # Check if exists
                                existing_idx = next(
                                    (j for j, h in enumerate(st.session_state.holdings) if h.symbol == sym),
                                    None
                                )
                                if existing_idx is not None:
                                    st.session_state.holdings[existing_idx] = holding
                                else:
                                    st.session_state.holdings.append(holding)
                                success_count += 1
                        else:
                            errors.append(f"{sym}: Could not fetch data")

                    except ValueError as e:
                        errors.append(f"Line '{line}': Invalid format")

                    progress_bar.progress((i + 1) / len(lines))

                status_text.empty()
                progress_bar.empty()

                if success_count > 0:
                    st.success(f"Successfully added/updated {success_count} holdings!")
                if errors:
                    with st.expander(f"⚠ {len(errors)} error(s)", expanded=True):
                        for err in errors:
                            st.warning(err)
                if success_count > 0:
                    st.rerun()

    else:  # Quick Add Popular
        # Shares and price inputs for quick add
        qa_col1, qa_col2 = st.columns(2)
        with qa_col1:
            qa_shares = st.number_input(
                "Shares to add",
                min_value=0.01,
                value=10.0,
                step=1.0,
                key="qa_shares",
                help="Number of shares for quick-add buttons"
            )
        with qa_col2:
            qa_cost = st.number_input(
                "Avg Cost ($ per share)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key="qa_cost",
                help="Leave at 0 to use current market price"
            )

        st.caption("Click a ticker to add with the shares and cost above")

        pop_tab1, pop_tab2, pop_tab3, pop_tab4 = st.tabs(["Stocks", "ETFs", "Bond ETFs", "REITs"])

        with pop_tab1:
            pop_cols = st.columns(5)
            for i, (sym, name) in enumerate(POPULAR_STOCKS):
                with pop_cols[i % 5]:
                    if st.button(f"{sym}", key=f"pop_stock_{sym}", help=name):
                        with st.spinner(f"Adding {sym}..."):
                            data = fetch_security_data(sym)
                            if data:
                                cost = qa_cost if qa_cost > 0 else data["current_price"]
                                holding = create_holding(sym, qa_shares, cost)
                                if holding:
                                    st.session_state.holdings.append(holding)
                                    st.rerun()

        with pop_tab2:
            pop_cols = st.columns(5)
            for i, (sym, name) in enumerate(POPULAR_ETFS):
                with pop_cols[i % 5]:
                    if st.button(f"{sym}", key=f"pop_etf_{sym}", help=name):
                        with st.spinner(f"Adding {sym}..."):
                            data = fetch_security_data(sym)
                            if data:
                                cost = qa_cost if qa_cost > 0 else data["current_price"]
                                holding = create_holding(sym, qa_shares, cost)
                                if holding:
                                    st.session_state.holdings.append(holding)
                                    st.rerun()

        with pop_tab3:
            pop_cols = st.columns(5)
            for i, (sym, name) in enumerate(POPULAR_BOND_ETFS):
                with pop_cols[i % 5]:
                    if st.button(f"{sym}", key=f"pop_bond_{sym}", help=name):
                        with st.spinner(f"Adding {sym}..."):
                            data = fetch_security_data(sym)
                            if data:
                                cost = qa_cost if qa_cost > 0 else data["current_price"]
                                holding = create_holding(sym, qa_shares, cost)
                                if holding:
                                    st.session_state.holdings.append(holding)
                                    st.rerun()

        with pop_tab4:
            pop_cols = st.columns(5)
            for i, (sym, name) in enumerate(POPULAR_REITS):
                with pop_cols[i % 5]:
                    if st.button(f"{sym}", key=f"pop_reit_{sym}", help=name):
                        with st.spinner(f"Adding {sym}..."):
                            data = fetch_security_data(sym)
                            if data:
                                cost = qa_cost if qa_cost > 0 else data["current_price"]
                                holding = create_holding(sym, qa_shares, cost)
                                if holding:
                                    st.session_state.holdings.append(holding)
                                    st.rerun()

    st.markdown("---")

    # ── Portfolio Summary ──
    if st.session_state.holdings:
        summary = calculate_portfolio_summary(st.session_state.holdings)

        st.markdown("### Portfolio Summary")
        sum_cols = st.columns(6)
        with sum_cols[0]:
            st.metric("Total Value", f"${summary['total_value']:,.2f}")
        with sum_cols[1]:
            pnl_delta = f"{summary['total_pnl_pct']:+.2f}%"
            st.metric("Total P&L", f"${summary['total_pnl']:,.2f}", delta=pnl_delta,
                      delta_color="normal" if summary['total_pnl'] >= 0 else "inverse")
        with sum_cols[2]:
            st.metric("Annual Income", f"${summary['annual_income']:,.2f}")
        with sum_cols[3]:
            st.metric("Portfolio Yield", f"{summary['portfolio_yield']:.2f}%")
        with sum_cols[4]:
            st.metric("Weighted Beta", f"{summary['weighted_beta']:.2f}")
        with sum_cols[5]:
            st.metric("Holdings", f"{len(st.session_state.holdings)}")

        # ── Portfolio Allocation Charts ──
        if len(st.session_state.holdings) > 0:
            alloc_col1, alloc_col2 = st.columns(2)

            with alloc_col1:
                st.markdown("#### Holdings by Weight")

                # Build holdings data with weights and colors
                holdings_for_pie = []
                for h in st.session_state.holdings:
                    weight_pct = (h.market_value / summary["total_value"] * 100) if summary["total_value"] > 0 else 0
                    holdings_for_pie.append({
                        "symbol": h.symbol,
                        "name": h.name,
                        "weight": weight_pct,
                        "value": h.market_value,
                        "color": get_company_color(h.symbol),
                    })

                # Sort by weight descending
                holdings_for_pie.sort(key=lambda x: x["weight"], reverse=True)

                # If more than 10 holdings, group small ones as "Other"
                if len(holdings_for_pie) > 10:
                    top_holdings = holdings_for_pie[:9]
                    other_weight = sum(h["weight"] for h in holdings_for_pie[9:])
                    other_value = sum(h["value"] for h in holdings_for_pie[9:])
                    top_holdings.append({
                        "symbol": "Other",
                        "name": f"{len(holdings_for_pie) - 9} other holdings",
                        "weight": other_weight,
                        "value": other_value,
                        "color": "#CCCCCC",
                    })
                    holdings_for_pie = top_holdings

                # Create pie chart with company colors - dark background
                fig_hold, ax_hold = plt.subplots(figsize=(7, 5), facecolor=CHART_BG_COLOR)
                ax_hold.set_facecolor(CHART_BG_COLOR)
                colors = [h["color"] for h in holdings_for_pie]
                weights = [h["weight"] for h in holdings_for_pie]
                labels = [f"{h['symbol']}" for h in holdings_for_pie]

                # Create pie with percentages
                wedges, texts, autotexts = ax_hold.pie(
                    weights,
                    labels=labels,
                    autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                    colors=colors,
                    startangle=90,
                    pctdistance=0.75,
                    labeldistance=1.1,
                )

                # Style the text for dark background
                for text in texts:
                    text.set_fontsize(9)
                    text.set_fontweight('bold')
                    text.set_color('white')
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

                ax_hold.set_title("Portfolio Holdings", fontsize=12, fontweight='bold', color='white')
                fig_hold.tight_layout()
                st.pyplot(fig_hold)
                plt.close(fig_hold)

                # Show holdings breakdown table
                st.caption("**Holding Details:**")
                for h in holdings_for_pie:
                    st.markdown(
                        f"<span style='color: {h['color']}; font-weight: bold;'>●</span> "
                        f"**{h['symbol']}** — {h['weight']:.1f}% (${h['value']:,.0f})",
                        unsafe_allow_html=True
                    )

            with alloc_col2:
                # Asset Type breakdown - dark background
                st.markdown("#### By Asset Type")
                if summary["asset_allocation"]:
                    alloc_df = pd.DataFrame(
                        list(summary["asset_allocation"].items()),
                        columns=["Asset Type", "Weight (%)"]
                    ).sort_values("Weight (%)", ascending=False)

                    fig_type, ax_type = plt.subplots(figsize=(7, 3), facecolor=CHART_BG_COLOR)
                    ax_type.set_facecolor(CHART_FACE_COLOR)
                    type_colors = {
                        "Stock": "#1f77b4",
                        "ETF": "#2ca02c",
                        "Bond ETF": "#ff7f0e",
                        "REIT": "#9467bd",
                    }
                    colors_type = [type_colors.get(t, "#666666") for t in alloc_df["Asset Type"]]
                    bars = ax_type.barh(alloc_df["Asset Type"], alloc_df["Weight (%)"], color=colors_type)
                    ax_type.set_xlabel("Weight (%)", color='white')
                    ax_type.set_title("Asset Type Breakdown", color='white')
                    ax_type.tick_params(colors='white')
                    for bar, val in zip(bars, alloc_df["Weight (%)"]):
                        ax_type.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                                    f'{val:.1f}%', va='center', fontsize=9, color='white')
                    ax_type.set_xlim(0, max(alloc_df["Weight (%)"]) * 1.2)
                    ax_type.grid(True, alpha=0.3, axis="x")
                    fig_type.tight_layout()
                    st.pyplot(fig_type)
                    plt.close(fig_type)

                # Sector breakdown (for stocks) - dark background
                if summary["sector_allocation"]:
                    st.markdown("#### By Sector (Stocks Only)")
                    sector_df = pd.DataFrame(
                        list(summary["sector_allocation"].items()),
                        columns=["Sector", "Weight (%)"]
                    ).sort_values("Weight (%)", ascending=True)

                    fig_sec, ax_sec = plt.subplots(figsize=(7, 4), facecolor=CHART_BG_COLOR)
                    ax_sec.set_facecolor(CHART_FACE_COLOR)
                    ax_sec.barh(sector_df["Sector"], sector_df["Weight (%)"], color="#1f77b4")
                    ax_sec.set_xlabel("Weight (%)", color='white')
                    ax_sec.set_title("Sector Breakdown", color='white')
                    ax_sec.tick_params(colors='white')
                    for i, (idx, row) in enumerate(sector_df.iterrows()):
                        ax_sec.text(row["Weight (%)"] + 0.5, i, f'{row["Weight (%)"]:.1f}%',
                                   va='center', fontsize=8, color='white')
                    ax_sec.grid(True, alpha=0.3, axis="x")
                    fig_sec.tight_layout()
                    st.pyplot(fig_sec)
                    plt.close(fig_sec)

        st.markdown("---")

        # ── Holdings Table ──
        st.markdown("### Holdings")

        # Refresh button
        ref_col1, ref_col2 = st.columns([1, 5])
        with ref_col1:
            if st.button("Refresh Prices", key="refresh_prices"):
                with st.spinner("Refreshing..."):
                    refreshed = []
                    for h in st.session_state.holdings:
                        new_h = create_holding(h.symbol, h.shares, h.avg_cost)
                        if new_h:
                            refreshed.append(new_h)
                        else:
                            refreshed.append(h)
                    st.session_state.holdings = refreshed
                    st.rerun()

        # Build holdings dataframe
        holdings_data = []
        for h in st.session_state.holdings:
            holdings_data.append({
                "Symbol": h.symbol,
                "Name": h.name[:25] + "..." if len(h.name) > 25 else h.name,
                "Type": h.asset_type,
                "Shares": h.shares,
                "Price": h.current_price,
                "Avg Cost": h.avg_cost,
                "Purchased": h.purchase_date.strftime("%Y-%m-%d") if h.purchase_date else "-",
                "Value": h.market_value,
                "P&L": h.unrealized_pnl,
                "P&L %": h.unrealized_pnl_pct,
                "Div Yield": h.dividend_yield * 100,
                "Income": h.annual_income,
                "Beta": h.beta,
            })

        holdings_df = pd.DataFrame(holdings_data)

        # Format for display
        display_df = holdings_df.copy()
        display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:,.2f}")
        display_df["Avg Cost"] = display_df["Avg Cost"].apply(lambda x: f"${x:,.2f}")
        display_df["Value"] = display_df["Value"].apply(lambda x: f"${x:,.2f}")
        display_df["P&L"] = display_df["P&L"].apply(lambda x: f"${x:+,.2f}")
        display_df["P&L %"] = display_df["P&L %"].apply(lambda x: f"{x:+.2f}%")
        display_df["Div Yield"] = display_df["Div Yield"].apply(lambda x: f"{x:.2f}%")
        display_df["Income"] = display_df["Income"].apply(lambda x: f"${x:,.2f}")
        display_df["Beta"] = display_df["Beta"].apply(lambda x: f"{x:.2f}" if x else "-")
        display_df["Shares"] = display_df["Shares"].apply(lambda x: f"{x:,.2f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, 35 * len(holdings_df) + 38),
        )

        # ── Edit Holdings ──
        with st.expander("Edit Holdings"):
            st.caption("Adjust shares, cost, or purchase date for existing holdings")

            for i, h in enumerate(st.session_state.holdings):
                edit_cols = st.columns([0.8, 1, 1, 1.2, 0.8])
                with edit_cols[0]:
                    st.markdown(f"**{h.symbol}**")
                with edit_cols[1]:
                    edit_shares = st.number_input(
                        "Shares",
                        min_value=0.01,
                        value=float(h.shares),
                        step=1.0,
                        key=f"edit_shares_{h.symbol}_{i}",
                    )
                with edit_cols[2]:
                    edit_cost = st.number_input(
                        "Avg Cost",
                        min_value=0.01,
                        value=float(h.avg_cost),
                        step=1.0,
                        key=f"edit_cost_{h.symbol}_{i}",
                    )
                with edit_cols[3]:
                    edit_date = st.date_input(
                        "Purchase Date",
                        value=h.purchase_date.date() if h.purchase_date else datetime.now().date(),
                        key=f"edit_date_{h.symbol}_{i}",
                    )
                with edit_cols[4]:
                    if st.button("Save", key=f"update_{h.symbol}_{i}"):
                        edit_purchase_dt = datetime.combine(edit_date, datetime.min.time())
                        updated = create_holding(h.symbol, edit_shares, edit_cost, edit_purchase_dt)
                        if updated:
                            st.session_state.holdings[i] = updated
                            st.success(f"Updated {h.symbol}")
                            st.rerun()

        # ── Remove Holdings ──
        with st.expander("Remove Holdings"):
            del_cols = st.columns(6)
            for i, h in enumerate(st.session_state.holdings):
                with del_cols[i % 6]:
                    if st.button(f"❌ {h.symbol}", key=f"del_{h.symbol}_{i}"):
                        st.session_state.holdings.pop(i)
                        st.rerun()

            if st.button("Clear All Holdings", type="secondary", key="clear_all"):
                st.session_state.holdings = []
                st.rerun()

        st.markdown("---")

        # ══════════════════════════════════════════════
        # PERFORMANCE CHARTS
        # ══════════════════════════════════════════════
        st.markdown("### Performance Charts")

        # Initialize selected stock for performance view
        if "selected_perf_stock" not in st.session_state:
            st.session_state.selected_perf_stock = None

        perf_tab1, perf_tab2 = st.tabs(["Individual Stock", "Portfolio Performance"])

        with perf_tab1:
            st.caption("Click a stock to view its price performance")

            # Stock selector buttons
            stock_cols = st.columns(min(len(st.session_state.holdings), 8))
            for i, h in enumerate(st.session_state.holdings):
                with stock_cols[i % 8]:
                    btn_style = "primary" if st.session_state.selected_perf_stock == h.symbol else "secondary"
                    if st.button(h.symbol, key=f"perf_btn_{h.symbol}", type=btn_style):
                        st.session_state.selected_perf_stock = h.symbol
                        st.rerun()

            # Period selector
            period_col1, period_col2 = st.columns([1, 3])
            with period_col1:
                perf_period = st.selectbox(
                    "Time Period",
                    ["1D", "1W", "1M", "3M", "6M", "1Y", "All Time"],
                    index=3,
                    key="perf_period"
                )

            period_map = {
                "1D": "1d",
                "1W": "5d",
                "1M": "1mo",
                "3M": "3mo",
                "6M": "6mo",
                "1Y": "1y",
                "All Time": "max",
            }

            if st.session_state.selected_perf_stock:
                selected_sym = st.session_state.selected_perf_stock
                selected_holding = next(
                    (h for h in st.session_state.holdings if h.symbol == selected_sym),
                    None
                )

                if selected_holding:
                    with st.spinner(f"Loading {selected_sym} performance..."):
                        # Get performance data - use purchase date if "All Time" and we have it
                        if perf_period == "All Time" and selected_holding.purchase_date:
                            perf_data = get_stock_performance(
                                selected_sym,
                                period="max",
                                start_date=selected_holding.purchase_date
                            )
                        else:
                            perf_data = get_stock_performance(selected_sym, period_map[perf_period])

                    if perf_data:
                        # Stock info header
                        info_cols = st.columns(5)
                        with info_cols[0]:
                            st.metric("Current Price", f"${selected_holding.current_price:,.2f}")
                        with info_cols[1]:
                            st.metric(
                                f"{perf_period} Return",
                                f"{perf_data['total_return']:+.2f}%",
                                delta_color="normal" if perf_data['total_return'] >= 0 else "inverse"
                            )
                        with info_cols[2]:
                            st.metric("Period High", f"${perf_data['period_high']:,.2f}")
                        with info_cols[3]:
                            st.metric("Period Low", f"${perf_data['period_low']:,.2f}")
                        with info_cols[4]:
                            if selected_holding.purchase_date:
                                st.metric("Purchased", selected_holding.purchase_date.strftime("%b %d, %Y"))
                            else:
                                st.metric("Days", f"{perf_data['days']}")

                        # Price chart
                        fig_stock, ax_stock = plt.subplots(figsize=(10, 5), facecolor=CHART_BG_COLOR)
                        ax_stock.set_facecolor(CHART_FACE_COLOR)

                        hist = perf_data["history"]
                        color = "#00ff00" if perf_data['total_return'] >= 0 else "#ff4444"
                        ax_stock.plot(hist["Date"], hist["Close"], color=color, linewidth=2)
                        ax_stock.fill_between(hist["Date"], hist["Close"], alpha=0.2, color=color)

                        # Add cost basis line if available
                        if selected_holding.avg_cost > 0:
                            ax_stock.axhline(
                                selected_holding.avg_cost,
                                color="#ffaa00",
                                linestyle="--",
                                alpha=0.8,
                                label=f"Avg Cost: ${selected_holding.avg_cost:.2f}"
                            )

                        ax_stock.set_xlabel("Date", color='white')
                        ax_stock.set_ylabel("Price ($)", color='white')
                        ax_stock.set_title(f"{selected_holding.name} ({selected_sym}) - {perf_period}", color='white')
                        ax_stock.tick_params(colors='white')
                        ax_stock.legend(facecolor=CHART_FACE_COLOR, labelcolor='white')
                        ax_stock.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        fig_stock.tight_layout()
                        st.pyplot(fig_stock)
                        plt.close(fig_stock)

                        # Your position performance
                        if selected_holding.avg_cost > 0:
                            st.markdown("**Your Position:**")
                            pos_cols = st.columns(4)
                            with pos_cols[0]:
                                st.metric("Shares", f"{selected_holding.shares:,.2f}")
                            with pos_cols[1]:
                                st.metric("Total Cost", f"${selected_holding.total_cost:,.2f}")
                            with pos_cols[2]:
                                st.metric("Market Value", f"${selected_holding.market_value:,.2f}")
                            with pos_cols[3]:
                                st.metric(
                                    "Your P&L",
                                    f"${selected_holding.unrealized_pnl:+,.2f}",
                                    delta=f"{selected_holding.unrealized_pnl_pct:+.2f}%",
                                    delta_color="normal" if selected_holding.unrealized_pnl >= 0 else "inverse"
                                )
                    else:
                        st.warning(f"Could not load performance data for {selected_sym}")
            else:
                st.info("Click a stock above to view its performance chart")

        with perf_tab2:
            st.caption("Track your portfolio performance vs. S&P 500 benchmark")

            # Portfolio period selector
            port_period = st.selectbox(
                "Time Period",
                ["1M", "3M", "6M", "1Y", "All Time"],
                index=2,
                key="port_perf_period"
            )

            with st.spinner("Calculating portfolio performance..."):
                port_perf = get_portfolio_performance(
                    st.session_state.holdings,
                    period_map.get(port_period, "6mo")
                )

            if port_perf:
                # Portfolio summary metrics
                port_cols = st.columns(5)
                with port_cols[0]:
                    st.metric("Total Value", f"${port_perf['end_value']:,.2f}")
                with port_cols[1]:
                    st.metric("Total Cost", f"${port_perf['total_cost']:,.2f}")
                with port_cols[2]:
                    st.metric(
                        "Total P&L",
                        f"${port_perf['total_pnl']:+,.2f}",
                        delta=f"{port_perf['total_return']:+.2f}%",
                        delta_color="normal" if port_perf['total_pnl'] >= 0 else "inverse"
                    )
                with port_cols[3]:
                    st.metric("Period", f"{port_perf['days']} days")
                with port_cols[4]:
                    if "SPY_Return" in port_perf["history"].columns:
                        spy_return = port_perf["history"]["SPY_Return"].iloc[-1]
                        alpha = port_perf['total_return'] - spy_return
                        st.metric(
                            "Alpha vs SPY",
                            f"{alpha:+.2f}%",
                            delta="Outperforming" if alpha > 0 else "Underperforming",
                            delta_color="normal" if alpha >= 0 else "inverse"
                        )

                # Portfolio performance chart
                fig_port, ax_port = plt.subplots(figsize=(10, 5), facecolor=CHART_BG_COLOR)
                ax_port.set_facecolor(CHART_FACE_COLOR)

                hist = port_perf["history"]
                port_color = "#00ff00" if port_perf['total_return'] >= 0 else "#ff4444"

                ax_port.plot(
                    hist["Date"],
                    hist["Portfolio_Return"],
                    color=port_color,
                    linewidth=2,
                    label=f"Your Portfolio ({port_perf['total_return']:+.2f}%)"
                )

                # Add SPY benchmark
                if "SPY_Return" in hist.columns:
                    spy_final = hist["SPY_Return"].iloc[-1]
                    ax_port.plot(
                        hist["Date"],
                        hist["SPY_Return"],
                        color="#888888",
                        linewidth=2,
                        linestyle="--",
                        label=f"S&P 500 ({spy_final:+.2f}%)"
                    )

                ax_port.axhline(0, color="white", linewidth=0.5, alpha=0.5)
                ax_port.set_xlabel("Date", color='white')
                ax_port.set_ylabel("Return (%)", color='white')
                ax_port.set_title(f"Portfolio Performance - {port_period}", color='white')
                ax_port.tick_params(colors='white')
                ax_port.legend(facecolor=CHART_FACE_COLOR, labelcolor='white', loc='upper left')
                ax_port.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                fig_port.tight_layout()
                st.pyplot(fig_port)
                plt.close(fig_port)
            else:
                st.warning("Could not calculate portfolio performance. Make sure you have holdings with valid data.")

        st.markdown("---")

        # ── Top Dividend Payers ──
        if any(h.dividend_yield > 0 for h in st.session_state.holdings):
            st.markdown("### Top Dividend Payers")
            div_holdings = sorted(
                [h for h in st.session_state.holdings if h.dividend_yield > 0],
                key=lambda x: x.dividend_yield,
                reverse=True
            )[:5]

            div_cols = st.columns(len(div_holdings))
            for i, h in enumerate(div_holdings):
                with div_cols[i]:
                    st.metric(
                        h.symbol,
                        f"{h.dividend_yield * 100:.2f}%",
                        delta=f"${h.annual_income:,.0f}/yr",
                        delta_color="off",
                    )

            st.markdown("---")

            # ══════════════════════════════════════════════
            # DIVIDEND & EARNINGS CALENDAR
            # ══════════════════════════════════════════════
            st.markdown("### Dividend & Earnings Calendar")
            st.caption("Expected dividend ex-dates and earnings report dates")

            # Legend
            legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
            with legend_col1:
                st.markdown(
                    "<span style='background-color: #d4edda; padding: 2px 8px; border-radius: 3px;'>"
                    "🟢 Dividend</span>",
                    unsafe_allow_html=True
                )
            with legend_col2:
                st.markdown(
                    "<span style='background-color: #fff3cd; padding: 2px 8px; border-radius: 3px;'>"
                    "🟡 Earnings</span>",
                    unsafe_allow_html=True
                )
            with legend_col3:
                st.markdown(
                    "<span style='background-color: #f8d7da; padding: 2px 8px; border-radius: 3px;'>"
                    "🔴 Fed/Econ</span>",
                    unsafe_allow_html=True
                )
            with legend_col4:
                st.markdown(
                    "<span style='background-color: #cce5ff; padding: 2px 8px; border-radius: 3px;'>"
                    "🔵 Multiple</span>",
                    unsafe_allow_html=True
                )

            # Initialize calendar state
            if "cal_year" not in st.session_state:
                st.session_state.cal_year = datetime.now().year
            if "cal_month" not in st.session_state:
                st.session_state.cal_month = datetime.now().month

            # Month navigation
            nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])

            with nav_col1:
                if st.button("◀ Prev", key="prev_month"):
                    if st.session_state.cal_month == 1:
                        st.session_state.cal_month = 12
                        st.session_state.cal_year -= 1
                    else:
                        st.session_state.cal_month -= 1
                    st.rerun()

            with nav_col2:
                if st.button("Today", key="today_month"):
                    st.session_state.cal_year = datetime.now().year
                    st.session_state.cal_month = datetime.now().month
                    st.rerun()

            with nav_col3:
                month_name = cal_module.month_name[st.session_state.cal_month]
                st.markdown(
                    f"<h3 style='text-align: center; margin: 0;'>{month_name} {st.session_state.cal_year}</h3>",
                    unsafe_allow_html=True
                )

            with nav_col5:
                if st.button("Next ▶", key="next_month"):
                    if st.session_state.cal_month == 12:
                        st.session_state.cal_month = 1
                        st.session_state.cal_year += 1
                    else:
                        st.session_state.cal_month += 1
                    st.rerun()

            # Build combined calendar data
            with st.spinner("Loading calendar..."):
                cal_data = build_combined_calendar(
                    st.session_state.holdings,
                    st.session_state.cal_year,
                    st.session_state.cal_month
                )

            # Monthly summary
            sum_col1, sum_col2, sum_col3 = st.columns(3)
            with sum_col1:
                if cal_data["monthly_div_total"] > 0:
                    st.success(
                        f"**Dividends: ${cal_data['monthly_div_total']:,.2f}** from "
                        f"{len(cal_data['dividend_events'])} payment(s)"
                    )
                else:
                    st.info("No dividends expected this month")
            with sum_col2:
                if cal_data["earnings_events"]:
                    confirmed = sum(1 for e in cal_data["earnings_events"] if e.is_confirmed)
                    estimated = len(cal_data["earnings_events"]) - confirmed
                    st.warning(
                        f"**Earnings: {len(cal_data['earnings_events'])} report(s)** "
                        f"({confirmed} confirmed, {estimated} estimated)"
                    )
                else:
                    st.info("No earnings reports this month")
            with sum_col3:
                if cal_data.get("fed_events"):
                    high_importance = sum(1 for e in cal_data["fed_events"] if e.importance == "HIGH")
                    st.error(
                        f"**Fed/Econ: {len(cal_data['fed_events'])} event(s)** "
                        f"({high_importance} high importance)"
                    )
                else:
                    st.info("No Fed/economic events this month")

            # Calendar grid display
            st.markdown("#### Calendar View")

            # Header row
            day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            header_cols = st.columns(7)
            for i, day_name in enumerate(day_names):
                with header_cols[i]:
                    st.markdown(f"**{day_name}**")

            # Calendar weeks
            for week in cal_data["calendar_grid"]:
                week_cols = st.columns(7)
                for i, day_data in enumerate(week):
                    with week_cols[i]:
                        if day_data["day"] == 0:
                            st.markdown("&nbsp;")
                        else:
                            day_num = day_data["day"]
                            div_events = day_data.get("dividends", [])
                            earn_events = day_data.get("earnings", [])
                            fed_events = day_data.get("fed_events", [])

                            has_div = len(div_events) > 0
                            has_earn = len(earn_events) > 0
                            has_fed = len(fed_events) > 0
                            event_count = sum([has_div, has_earn, has_fed])

                            if event_count >= 2:
                                # Multiple event types - blue
                                lines = []
                                if has_div:
                                    div_symbols = ", ".join(e.symbol for e in div_events[:2])
                                    lines.append(f"💵 {div_symbols}")
                                if has_earn:
                                    earn_symbols = ", ".join(e.symbol for e in earn_events[:2])
                                    lines.append(f"📊 {earn_symbols}")
                                if has_fed:
                                    fed_types = ", ".join(e.event_type for e in fed_events[:2])
                                    lines.append(f"🏛 {fed_types}")

                                content = "<br>".join(
                                    f"<span style='font-size: 0.65em; color: #000000; font-weight: bold;'>{l}</span>"
                                    for l in lines
                                )
                                st.markdown(
                                    f"<div style='background-color: #cce5ff; padding: 4px; "
                                    f"border-radius: 4px; text-align: center; min-height: 70px; color: #000000;'>"
                                    f"<strong style='color: #000000;'>{day_num}</strong><br>"
                                    f"{content}</div>",
                                    unsafe_allow_html=True
                                )
                            elif has_div:
                                # Dividend only - green
                                total_income = sum(e.expected_income for e in div_events)
                                symbols = ", ".join(e.symbol for e in div_events)
                                st.markdown(
                                    f"<div style='background-color: #d4edda; padding: 4px; "
                                    f"border-radius: 4px; text-align: center; min-height: 70px; color: #000000;'>"
                                    f"<strong style='color: #000000;'>{day_num}</strong><br>"
                                    f"<span style='font-size: 0.75em; color: #000000; font-weight: bold;'>💵 {symbols}</span><br>"
                                    f"<span style='font-size: 0.7em; color: #000000; font-weight: bold;'>${total_income:,.0f}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                            elif has_earn:
                                # Earnings only - yellow
                                symbols = ", ".join(e.symbol for e in earn_events)
                                confirmed_mark = "✓" if all(e.is_confirmed for e in earn_events) else "?"
                                st.markdown(
                                    f"<div style='background-color: #fff3cd; padding: 4px; "
                                    f"border-radius: 4px; text-align: center; min-height: 70px; color: #000000;'>"
                                    f"<strong style='color: #000000;'>{day_num}</strong><br>"
                                    f"<span style='font-size: 0.75em; color: #000000; font-weight: bold;'>📊 {symbols}</span><br>"
                                    f"<span style='font-size: 0.7em; color: #000000;'>{confirmed_mark}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                            elif has_fed:
                                # Fed/Economic event only - red
                                event_types = ", ".join(e.event_type for e in fed_events)
                                importance = "⚠" if any(e.importance == "HIGH" for e in fed_events) else ""
                                st.markdown(
                                    f"<div style='background-color: #f8d7da; padding: 4px; "
                                    f"border-radius: 4px; text-align: center; min-height: 70px; color: #000000;'>"
                                    f"<strong style='color: #000000;'>{day_num}</strong><br>"
                                    f"<span style='font-size: 0.75em; color: #000000; font-weight: bold;'>🏛 {event_types}</span><br>"
                                    f"<span style='font-size: 0.7em; color: #000000;'>{importance}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                            else:
                                # Regular day
                                st.markdown(
                                    f"<div style='padding: 4px; text-align: center; "
                                    f"min-height: 70px; color: #333333;'>{day_num}</div>",
                                    unsafe_allow_html=True
                                )

            # Detailed events - Dividends
            if cal_data["dividend_events"]:
                st.markdown("#### Dividend Details")
                div_event_data = []
                for event in cal_data["dividend_events"]:
                    div_event_data.append({
                        "Ex-Date": event.ex_date.strftime("%b %d, %Y"),
                        "Symbol": event.symbol,
                        "Name": event.name[:20] + "..." if len(event.name) > 20 else event.name,
                        "Frequency": event.frequency,
                        "Amount/Share": f"${event.amount:.4f}",
                        "Shares": f"{event.shares:,.0f}",
                        "Expected Income": f"${event.expected_income:,.2f}",
                    })

                st.dataframe(
                    pd.DataFrame(div_event_data),
                    use_container_width=True,
                    hide_index=True,
                )

            # Detailed events - Earnings
            if cal_data["earnings_events"]:
                st.markdown("#### Earnings Report Details")
                earn_event_data = []
                for event in cal_data["earnings_events"]:
                    earn_event_data.append({
                        "Date": event.earnings_date.strftime("%b %d, %Y"),
                        "Symbol": event.symbol,
                        "Name": event.name[:20] + "..." if len(event.name) > 20 else event.name,
                        "Status": "✓ Confirmed" if event.is_confirmed else "? Estimated",
                        "EPS Est.": f"${event.eps_estimate:.2f}" if event.eps_estimate else "-",
                    })

                st.dataframe(
                    pd.DataFrame(earn_event_data),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Note: Earnings dates marked as 'Estimated' are projections based on typical "
                    "quarterly reporting patterns. Confirm with official company announcements."
                )

            # Detailed events - Fed/Economic
            if cal_data.get("fed_events"):
                st.markdown("#### Fed & Economic Events")
                fed_event_data = []
                for event in cal_data["fed_events"]:
                    fed_event_data.append({
                        "Date": event.event_date.strftime("%b %d, %Y"),
                        "Event": event.event_type,
                        "Description": event.description,
                        "Importance": f"{'⚠ ' if event.importance == 'HIGH' else ''}{event.importance}",
                    })

                st.dataframe(
                    pd.DataFrame(fed_event_data),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Fed/Economic events can cause significant market volatility. "
                    "Consider adjusting positions before high-importance events."
                )

            # 12-Month Projection Chart
            st.markdown("#### 12-Month Dividend Income Projection")
            with st.spinner("Calculating projections..."):
                projection = get_annual_dividend_projection(st.session_state.holdings)

            if projection:
                proj_df = pd.DataFrame([
                    {
                        "Month": f"{v['month_name'][:3]} {v['year']}",
                        "Income": v["total"],
                        "Payments": v["event_count"],
                    }
                    for k, v in projection.items()
                ])

                if proj_df["Income"].sum() > 0:
                    fig_proj, ax_proj = plt.subplots(figsize=(10, 4), facecolor=CHART_BG_COLOR)
                    ax_proj.set_facecolor(CHART_FACE_COLOR)
                    bars = ax_proj.bar(proj_df["Month"], proj_df["Income"], color="#28a745")
                    ax_proj.set_xlabel("Month", color='white')
                    ax_proj.set_ylabel("Expected Income ($)", color='white')
                    ax_proj.set_title("12-Month Dividend Income Projection", color='white')
                    ax_proj.tick_params(colors='white')
                    plt.xticks(rotation=45, ha="right")

                    for bar, val in zip(bars, proj_df["Income"]):
                        if val > 0:
                            ax_proj.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 5,
                                f"${val:,.0f}",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                color='white'
                            )

                    ax_proj.grid(True, alpha=0.3, axis="y")
                    fig_proj.tight_layout()
                    st.pyplot(fig_proj)
                    plt.close(fig_proj)

                    # Annual summary
                    annual_total = proj_df["Income"].sum()
                    st.info(
                        f"**Projected Annual Dividend Income: ${annual_total:,.2f}** "
                        f"(${annual_total/12:,.2f}/month average)"
                    )

    else:
        st.info("Your portfolio is empty. Add some holdings above to get started.")
        st.markdown("""
        **Quick Start:**
        1. Use "Quick Add Popular" to add common stocks, ETFs, or bonds with one click
        2. Or enter a custom symbol with your shares and average cost
        3. The portfolio will track prices, P&L, and dividend income automatically
        """)


# ═════════════════════════════════════════════
# TAB 1: Strategy Recommender + Trade Cards
# ═════════════════════════════════════════════

with tab_recommend:
    st.subheader("Recommended Strategies")

    regime_str = f" | Regime: **{entropy_signal.regime}**" if entropy_signal else ""
    st.caption(
        f"View: **{outlook}** | "
        f"IV: **{iv_percentile:.0f}th %ile** | "
        f"DTE: **{selected_dte}d** | "
        f"Risk: **{risk_tolerance}** | "
        f"Portfolio: **${portfolio_value:,.0f}**"
        f"{regime_str}"
    )

    # Build entropy adjustment function
    entropy_fn = None
    if entropy_signal:
        entropy_fn = lambda name, sig=entropy_signal: entropy_score_adjustment(name, sig)

    recommendations = recommend_strategies(
        outlook=outlook_key,
        iv_percentile=iv_percentile,
        dte=selected_dte,
        risk_tolerance=risk_tol_key,
        spot=spot,
        chain_df=chain_df,
        entropy_adjustment_fn=entropy_fn,
    )

    if not recommendations:
        st.warning("No strategies match your criteria. Try adjusting your outlook or risk tolerance.")
    else:
        for i, rec in enumerate(recommendations[:5]):
            strat = rec.strategy

            # Generate trade card
            card = generate_trade_card(
                strategy_name=strat.name,
                symbol=symbol,
                expiration=selected_exp,
                dte=selected_dte,
                spot=spot,
                portfolio_value=portfolio_value,
                risk_per_trade_pct=risk_per_trade,
                concrete_legs=rec.concrete_legs,
                net_cost=rec.net_cost,
                max_loss_dollars=rec.max_loss_dollars,
                max_gain_dollars=rec.max_gain_dollars,
                breakevens=rec.breakeven,
            )

            # Entropy badge
            entropy_badge = ""
            if rec.entropy_adjustment >= 5:
                entropy_badge = " [ENTROPY FAVORED]"
            elif rec.entropy_adjustment <= -5:
                entropy_badge = " [ENTROPY PENALIZED]"

            with st.expander(
                f"#{rec.rank}  {strat.name}{entropy_badge}  —  "
                f"Score: {rec.score:.0f}/100  |  "
                f"{'Credit' if card.is_credit else 'Debit'}: "
                f"${card.net_cost_per_share:.2f}  |  "
                f"Size: {card.recommended_contracts} contracts",
                expanded=(i == 0),
            ):
                # ══════════════ TRADE CARD ══════════════

                st.markdown("### Trade Card")

                # Row 1: Key metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Contracts", f"{card.recommended_contracts}")
                with m2:
                    st.metric("Max Loss (total)",
                              f"${card.max_loss_total:,.0f}")
                with m3:
                    if card.max_gain_total is not None:
                        st.metric("Max Gain (total)",
                                  f"${card.max_gain_total:,.0f}")
                    else:
                        st.metric("Max Gain", "Unlimited")
                with m4:
                    if card.risk_reward_ratio is not None:
                        st.metric("Risk/Reward",
                                  f"1:{card.risk_reward_ratio:.2f}")
                    else:
                        st.metric("Risk/Reward", "Unlimited upside")

                # Row 2: Position sizing details
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric("Capital Required",
                              f"${card.total_capital_required:,.0f}")
                with s2:
                    st.metric("% of Portfolio",
                              f"{card.total_capital_required / portfolio_value:.1%}"
                              if portfolio_value > 0 else "-")
                with s3:
                    st.metric("Max Risk Budget",
                              f"${card.max_risk_dollars:,.0f}")
                with s4:
                    if len(card.breakevens) == 1:
                        st.metric("Break-even", f"${card.breakevens[0]:,.2f}")
                    else:
                        be_str = " / ".join(f"${b:.2f}" for b in card.breakevens)
                        st.metric("Break-evens", be_str)

                # Trade construction table
                st.markdown("**Trade Construction:**")
                leg_data = []
                for leg in rec.concrete_legs:
                    leg_data.append({
                        "Action": "BUY" if leg["action"] == "buy" else "SELL",
                        "Type": leg["type"].upper(),
                        "Strike": f"${leg['strike']:.0f}",
                        "Price": f"${leg['estimated_price']:.2f}",
                        "Qty": card.recommended_contracts,
                    })
                st.dataframe(
                    pd.DataFrame(leg_data),
                    use_container_width=False,
                    hide_index=True,
                )

                cost_label = "Net Credit" if card.is_credit else "Net Debit"
                st.markdown(
                    f"**{cost_label}: ${card.net_cost_per_share:.2f}/share "
                    f"(${card.net_cost_per_contract:.2f}/contract "
                    f"x {card.recommended_contracts} = "
                    f"${card.net_cost_per_contract * card.recommended_contracts:,.2f} total)**"
                )

                # ── Exit Rules ──
                st.markdown("**Exit Rules:**")
                ex1, ex2 = st.columns(2)
                with ex1:
                    st.success(
                        f"**Profit Target:** Close at "
                        f"{card.profit_target_pct:.0%} of max profit = "
                        f"${card.profit_target_dollars:,.0f} per contract"
                    )
                with ex2:
                    st.error(
                        f"**Stop Loss:** Close at "
                        f"${card.stop_loss_dollars:,.0f} per contract loss"
                    )

                # ── Kelly Criterion ──
                if card.kelly_fraction is not None:
                    if card.kelly_fraction > 0:
                        st.caption(
                            f"Half-Kelly suggests {card.kelly_contracts} contracts "
                            f"({card.kelly_fraction:.1%} of portfolio). "
                            f"Risk-rule sizing: {card.recommended_contracts} contracts."
                        )
                    elif card.kelly_fraction < 0:
                        st.caption(
                            f"Kelly criterion is negative ({card.kelly_fraction:.3f}) — "
                            f"risk/reward may not justify this trade at estimated probability."
                        )

                # ── Warnings ──
                for warning in card.warnings:
                    st.warning(warning)

                # ── Payoff Diagram ──
                price_range, payoff, payoff_dollars = compute_strategy_payoff(
                    rec.concrete_legs, rec.net_cost, spot
                )
                payoff_total = payoff_dollars * max(card.recommended_contracts, 1)

                fig, ax = plt.subplots(figsize=(9, 4), facecolor=CHART_BG_COLOR)
                ax.set_facecolor(CHART_FACE_COLOR)
                ax.plot(price_range, payoff_total, color="#1f77b4", linewidth=2)
                ax.axhline(0, color="white", linewidth=0.5)
                ax.axvline(spot, color="gray", linestyle="--", alpha=0.7,
                           label=f"Spot ${spot:.2f}")
                for be in rec.breakeven:
                    ax.axvline(be, color="#00ff00", linestyle=":", alpha=0.7,
                               label=f"B/E ${be:.2f}")
                ax.fill_between(price_range, payoff_total, 0,
                                where=(payoff_total > 0), color="green", alpha=0.2)
                ax.fill_between(price_range, payoff_total, 0,
                                where=(payoff_total < 0), color="red", alpha=0.2)
                ax.set_xlabel("Stock Price at Expiration", color='white')
                ax.set_ylabel("Profit / Loss ($)", color='white')
                contracts_label = max(card.recommended_contracts, 1)
                ax.set_title(f"{strat.name} — P&L at Expiration ({contracts_label} contract{'s' if contracts_label > 1 else ''})", color='white')
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))
                ax.tick_params(colors='white')
                ax.legend(fontsize=8, facecolor=CHART_FACE_COLOR, labelcolor='white')
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # ── Why + Scoring ──
                st.markdown("**Why this strategy works here:**")
                st.markdown(f"> {strat.why_it_works}")

                with st.expander("Scoring breakdown", expanded=False):
                    for reason in rec.explanation.split(" | "):
                        st.markdown(f"- {reason}")
                    st.markdown(f"- **Greeks profile:** {strat.greeks_profile}")
                    st.markdown(f"- **Risk type:** {strat.risk_type.title()} | "
                                f"**Complexity:** {'Simple' if strat.complexity == 1 else 'Intermediate' if strat.complexity == 2 else 'Advanced'}")


# ═════════════════════════════════════════════
# TAB 2: Options Chain Explorer
# ═════════════════════════════════════════════

with tab_chain:
    st.subheader("Options Chain Explorer")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        moneyness_filter = st.multiselect(
            "Moneyness", ["ITM", "ATM", "OTM"],
            default=["ITM", "ATM", "OTM"], key="chain_moneyness",
        )
    with filter_col2:
        min_oi = st.number_input("Min Open Interest", value=0, min_value=0,
                                  step=10, key="chain_oi")
    with filter_col3:
        min_vol = st.number_input("Min Volume", value=0, min_value=0,
                                   step=10, key="chain_vol")

    filtered = chain_df[chain_df["moneyness_label"].isin(moneyness_filter)]
    if min_oi > 0:
        filtered = filtered[filtered["openInterest"].fillna(0) >= min_oi]
    if min_vol > 0:
        filtered = filtered[filtered["volume"].fillna(0) >= min_vol]

    display_cols = [
        "strike", "bid", "ask", "midPrice", "lastPrice", "bsPrice",
        "impliedVolatility", "delta", "gamma", "theta", "vega",
        "openInterest", "volume", "moneyness_label",
    ]

    def format_chain(df):
        display = df[display_cols].copy()
        display.columns = [
            "Strike", "Bid", "Ask", "Mid", "Last", "BS Price",
            "IV", "Delta", "Gamma", "Theta", "Vega", "OI", "Vol", "Money",
        ]
        for col in ["Bid", "Ask", "Mid", "Last", "BS Price"]:
            display[col] = display[col].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) else "-")
        display["IV"] = display["IV"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "-")
        display["Delta"] = display["Delta"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        display["Gamma"] = display["Gamma"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "-")
        display["Theta"] = display["Theta"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        display["Vega"] = display["Vega"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        display["Strike"] = display["Strike"].apply(lambda x: f"${x:.2f}")
        display["OI"] = display["OI"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else "-")
        display["Vol"] = display["Vol"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else "-")
        return display

    chain_tab_calls, chain_tab_puts = st.tabs(["Calls", "Puts"])

    with chain_tab_calls:
        calls = filtered[filtered["optionType"] == "call"].sort_values("strike")
        if calls.empty:
            st.warning("No calls match your filters.")
        else:
            st.caption(f"{len(calls)} contracts | Spot: ${spot:.2f} | DTE: {chain_df['dte'].iloc[0]}d")
            st.dataframe(format_chain(calls), use_container_width=True,
                         hide_index=True, height=min(500, 35 * len(calls) + 38))

    with chain_tab_puts:
        puts = filtered[filtered["optionType"] == "put"].sort_values("strike")
        if puts.empty:
            st.warning("No puts match your filters.")
        else:
            st.caption(f"{len(puts)} contracts | Spot: ${spot:.2f} | DTE: {chain_df['dte'].iloc[0]}d")
            st.dataframe(format_chain(puts), use_container_width=True,
                         hide_index=True, height=min(500, 35 * len(puts) + 38))


# ═════════════════════════════════════════════
# TAB 3: Greeks & IV
# ═════════════════════════════════════════════

with tab_greeks:
    st.subheader("Greeks Across Strikes")

    liquid = chain_df[chain_df["openInterest"].fillna(0) > 10].copy()
    if liquid.empty:
        liquid = chain_df.copy()
    liquid = liquid[(liquid["moneyness"] >= 0.80) & (liquid["moneyness"] <= 1.20)]

    greek_type = st.selectbox("Select Greek", ["Delta", "Gamma", "Theta", "Vega"],
                               index=0, key="greek_select")
    greek_col = greek_type.lower()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CHART_BG_COLOR)
    ax.set_facecolor(CHART_FACE_COLOR)
    calls_g = liquid[liquid["optionType"] == "call"].sort_values("strike")
    puts_g = liquid[liquid["optionType"] == "put"].sort_values("strike")

    if not calls_g.empty:
        ax.plot(calls_g["strike"], calls_g[greek_col], "b-o", markersize=3,
                label=f"Call {greek_type}", linewidth=1.5)
    if not puts_g.empty:
        ax.plot(puts_g["strike"], puts_g[greek_col], "r-o", markersize=3,
                label=f"Put {greek_type}", linewidth=1.5)

    ax.axvline(spot, color="gray", linestyle="--", alpha=0.7, label=f"Spot ${spot:.2f}")
    ax.set_xlabel("Strike Price", color='white')
    ax.set_ylabel(greek_type, color='white')
    ax.set_title(f"{symbol} — {greek_type} by Strike ({selected_exp})", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor=CHART_FACE_COLOR, labelcolor='white')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # IV Smile
    st.subheader("Volatility Smile")
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor=CHART_BG_COLOR)
    ax2.set_facecolor(CHART_FACE_COLOR)
    if not calls_g.empty:
        ax2.plot(calls_g["strike"], calls_g["impliedVolatility"] * 100,
                 "b-o", markersize=3, label="Call IV", linewidth=1.5)
    if not puts_g.empty:
        ax2.plot(puts_g["strike"], puts_g["impliedVolatility"] * 100,
                 "r-o", markersize=3, label="Put IV", linewidth=1.5)
    ax2.axvline(spot, color="gray", linestyle="--", alpha=0.7, label=f"Spot ${spot:.2f}")
    ax2.set_xlabel("Strike Price", color='white')
    ax2.set_ylabel("Implied Volatility (%)", color='white')
    ax2.set_title(f"{symbol} — Volatility Smile ({selected_exp})", color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor=CHART_FACE_COLOR, labelcolor='white')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # IV vs RV
    if iv_data:
        st.subheader("Implied vs. Realized Volatility")
        vol_data = {
            "Metric": ["Current ATM IV", "30d RV", "60d RV", "90d RV"],
            "Value": [iv_data["current_iv"], iv_data["rv_30"],
                      iv_data["rv_60"], iv_data["rv_90"]],
        }
        vol_df = pd.DataFrame(vol_data)

        fig3, ax3 = plt.subplots(figsize=(8, 4), facecolor=CHART_BG_COLOR)
        ax3.set_facecolor(CHART_FACE_COLOR)
        colors = ["#ff7f0e", "#1f77b4", "#1f77b4", "#1f77b4"]
        bars = ax3.barh(vol_df["Metric"], vol_df["Value"] * 100, color=colors)
        ax3.set_xlabel("Volatility (%)", color='white')
        ax3.set_title(f"{symbol} — IV vs. Realized Volatility", color='white')
        ax3.tick_params(colors='white')
        for bar, val in zip(bars, vol_df["Value"]):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1%}", va="center", fontsize=10, color='white')
        ax3.grid(True, alpha=0.3, axis="x")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        premium_ratio = iv_data["current_iv"] / iv_data["rv_30"] if iv_data["rv_30"] > 0 else 1
        if premium_ratio > 1.3:
            st.success(
                f"**IV/RV Ratio: {premium_ratio:.2f}x** — Options priced for "
                f"{(premium_ratio - 1) * 100:.0f}% more movement than realized. "
                f"Edge in selling premium.")
        elif premium_ratio < 0.8:
            st.success(
                f"**IV/RV Ratio: {premium_ratio:.2f}x** — Options cheap vs. "
                f"actual movement. Edge in buying premium.")
        else:
            st.info(f"**IV/RV Ratio: {premium_ratio:.2f}x** — IV and RV roughly in line.")


# ═════════════════════════════════════════════
# TAB 4: Entropy Analysis
# ═════════════════════════════════════════════

with tab_entropy:
    st.subheader("Market Entropy Analysis")
    st.caption(
        "Information-theoretic analysis of market structure. "
        "Based on the sample space expansion framework from Case 2 (ChatGPT Launch)."
    )

    if entropy_signal is None:
        st.warning("Could not compute market entropy. Check internet connection.")
    else:
        # ── Regime Summary ──
        if entropy_signal.regime == "CONCENTRATION":
            st.error(
                f"**CONCENTRATION REGIME** — Market entropy has decreased by "
                f"{abs(entropy_signal.entropy_change):.3f} bits "
                f"({abs(entropy_signal.entropy_change_pct):.1f}%) from baseline. "
                f"Fewer sectors are driving returns. Favor selling premium."
            )
        elif entropy_signal.regime == "UNCERTAINTY":
            st.warning(
                f"**UNCERTAINTY REGIME** — Market entropy has increased by "
                f"{abs(entropy_signal.entropy_change):.3f} bits "
                f"({abs(entropy_signal.entropy_change_pct):.1f}%) from baseline. "
                f"More sectors participating. Favor buying premium / owning optionality."
            )
        else:
            st.info(
                f"**STABLE REGIME** — Market entropy near baseline "
                f"(change: {entropy_signal.entropy_change:+.3f} bits). "
                f"No strong regime signal. Use standard IV-based selection."
            )

        # ── Entropy Metrics ──
        st.markdown("### Entropy Metrics")
        me1, me2, me3, me4 = st.columns(4)
        with me1:
            st.metric("Current Entropy", f"{entropy_signal.current_entropy:.3f} bits")
        with me2:
            st.metric("Baseline (Nov 2022)", f"{entropy_signal.baseline_entropy:.3f} bits")
        with me3:
            st.metric("Normalized", f"{entropy_signal.current_normalized:.3f}",
                       help="0 = one sector dominates, 1 = perfectly equal")
        with me4:
            st.metric("CR7 (Top 7)", f"{entropy_signal.current_cr7:.1%}",
                       help="Concentration ratio: weight of top 7 sectors")

        # ── Sector Weights Chart ──
        st.markdown("### Current Sector Weights")
        sector_df = pd.DataFrame(
            list(entropy_signal.sector_weights.items()),
            columns=["Sector", "Weight"],
        ).sort_values("Weight", ascending=True)

        fig4, ax4 = plt.subplots(figsize=(8, 5), facecolor=CHART_BG_COLOR)
        ax4.set_facecolor(CHART_FACE_COLOR)
        colors_s = ["#ff7f0e" if w > 0.12 else "#1f77b4"
                     for w in sector_df["Weight"]]
        bars = ax4.barh(sector_df["Sector"], sector_df["Weight"] * 100,
                        color=colors_s)
        ax4.set_xlabel("Weight (%)", color='white')
        ax4.set_title("S&P 500 Sector Weights (Estimated from ETF Performance)", color='white')
        ax4.tick_params(colors='white')

        for bar, val in zip(bars, sector_df["Weight"]):
            ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1%}", va="center", fontsize=9, color='white')

        ax4.grid(True, alpha=0.3, axis="x")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # ── Entropy Explanation ──
        st.markdown("### Analysis")
        st.markdown(entropy_signal.explanation)

        # ── Strategy Implications ──
        st.markdown("### Strategy Implications")
        for imp in entropy_signal.strategy_implications:
            if imp.startswith("FAVOR:"):
                st.markdown(f"- :green[{imp}]")
            elif imp.startswith("AVOID:"):
                st.markdown(f"- :red[{imp}]")
            elif imp.startswith("TREND") or imp.startswith("CAUTION"):
                st.markdown(f"- :orange[{imp}]")
            else:
                st.markdown(f"- {imp}")

        # ── Connection to Course Framework ──
        with st.expander("How this connects to the information theory framework"):
            st.markdown("""
**From Case 2 (ChatGPT Launch 2022):**

- **Shannon Entropy** H = -Sum(p * log2(p)) measures the information content of market structure
- **Lower entropy** = more concentrated = fewer sectors driving returns = "the market knows its narrative"
- **Higher entropy** = more dispersed = more uncertainty about which sectors lead = "the game may be changing"

**The Sample Space Expansion Paradox:**

When a new asset class emerges (like "AI infrastructure" after ChatGPT), you'd expect entropy to *increase* (more choices). But the ChatGPT case showed entropy *decreased* because the new category's winners (Mag 7) dominated everything else.

**For options trading, this means:**

| Entropy Signal | Market State | Options Strategy |
|---|---|---|
| Entropy falling | Concentration (few winners) | Sell premium on indices, directional spreads on leaders |
| Entropy rising | Dispersion (broad participation) | Buy premium, straddles/strangles, tail hedges |
| Entropy stable | Normal market | Standard IV-based strategy selection |

The entropy signal adds a *structural* dimension that VIX alone cannot capture. VIX measures expected volatility. Entropy measures whether the *game itself* is changing.
            """)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("---")
st.caption(
    "OptiMax v2.0 | Data: yfinance (15-min delayed) | "
    "Greeks: Black-Scholes (European approx.) | "
    "Entropy: Sector ETF-based Shannon entropy | "
    "Portfolio: Live prices & dividend yields | "
    "Not financial advice. For educational purposes only."
)
