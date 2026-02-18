"""
OptiMax — Options Strategy Recommender + Portfolio Manager
All Sections: Data Engine + Strategy Recommender + Entropy Signals + Trade Cards + Portfolio

Run with:  streamlit run app/optimax.py
"""

import os
import io
import calendar as cal_module
import colorsys
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

from options_engine import (
    get_ticker_info, get_expirations, get_enriched_chain, compute_iv_percentile,
)
from strategy_engine import recommend_strategies, compute_strategy_payoff
from entropy_bridge import compute_market_entropy, entropy_score_adjustment
from position_sizer import generate_trade_card
from portfolio_manager import (
    fetch_security_data, create_holding, calculate_portfolio_summary,
    build_dividend_calendar, build_combined_calendar, get_annual_dividend_projection,
    search_tickers, get_stock_performance, get_portfolio_performance,
    create_watchlist_item, refresh_watchlist_item,
    export_holdings_to_csv, import_holdings_from_csv, export_watchlist_to_csv,
    get_dividend_payment_history, get_monthly_dividend_totals,
    calculate_drip_projection, calculate_drip_vs_no_drip,
    get_stock_news, get_portfolio_news,
    POPULAR_STOCKS, POPULAR_ETFS, POPULAR_BOND_ETFS, POPULAR_REITS,
    TICKER_DATABASE, FedEvent, WatchlistItem, NewsItem,
)

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Dark theme for matplotlib
plt.style.use('dark_background')
CHART_BG = '#0E1117'
CHART_FACE = '#1A1D26'

# ─── Helpers ───

BRAND_COLORS = {
    "AAPL": "#A2AAAD", "MSFT": "#00A4EF", "GOOGL": "#4285F4", "GOOG": "#4285F4",
    "AMZN": "#FF9900", "META": "#0668E1", "NVDA": "#76B900", "TSLA": "#CC0000",
    "JPM": "#003087", "V": "#1A1F71", "MA": "#EB001B", "SPY": "#1E88E5",
    "QQQ": "#76B900", "SCHD": "#5A2D82", "O": "#003399", "VTI": "#C70000", "VOO": "#C70000",
}

def get_company_color(symbol: str) -> str:
    if symbol in BRAND_COLORS:
        return BRAND_COLORS[symbol]
    hue = ((sum(ord(c) for c in symbol) * 137) % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.65)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def render_news_card(news, badge_color, badge_text):
    """Render a single news card with badge."""
    time_ago = datetime.now() - news.published
    if time_ago.days > 0:
        time_str = f"{time_ago.days}d ago"
    elif time_ago.seconds > 3600:
        time_str = f"{time_ago.seconds // 3600}h ago"
    else:
        time_str = f"{time_ago.seconds // 60}m ago"

    st.markdown(
        f"<div style='padding:12px;margin:8px 0;background:{CHART_FACE};border-radius:8px;border-left:4px solid {badge_color};'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
        f"<span style='background:{badge_color};padding:3px 10px;border-radius:4px;font-size:0.85em;font-weight:bold;color:white;'>{badge_text}</span>"
        f"<span style='color:#888;font-size:0.8em;'>{time_str} &bull; {news.publisher}</span></div>"
        f"<a href='{news.link}' target='_blank' style='color:#4da6ff;text-decoration:none;font-weight:500;font-size:1.05em;'>{news.title} &nearr;</a></div>",
        unsafe_allow_html=True,
    )


def add_popular_holdings(items, key_prefix, qa_shares, qa_cost):
    """Render quick-add buttons for a list of (symbol, name) tuples."""
    cols = st.columns(5)
    for i, (sym, name) in enumerate(items):
        with cols[i % 5]:
            if st.button(sym, key=f"{key_prefix}_{sym}", help=name):
                with st.spinner(f"Adding {sym}..."):
                    data = fetch_security_data(sym)
                    if data:
                        cost = qa_cost if qa_cost > 0 else data["current_price"]
                        holding = create_holding(sym, qa_shares, cost)
                        if holding:
                            st.session_state.holdings.append(holding)
                            st.rerun()


def dark_plotly_layout(fig, title, xaxis="", yaxis="", height=400, **kwargs):
    """Apply consistent dark theme to plotly figures."""
    fig.update_layout(
        title=title, xaxis_title=xaxis, yaxis_title=yaxis,
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_FACE,
        font=dict(color='white'), height=height,
        hovermode='x unified', **kwargs,
    )


def dark_mpl_figure(figsize=(10, 5)):
    """Create a matplotlib figure with dark theme."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=CHART_BG)
    ax.set_facecolor(CHART_FACE)
    return fig, ax


PERIOD_MAP = {"1D": "1d", "1W": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "All Time": "max"}

# ─── Page Config ───

st.set_page_config(
    page_title="OptiMax — Options & Portfolio Intelligence",
    page_icon="📊", layout="wide", initial_sidebar_state="expanded",
)

# ─── Session State Defaults ───

_defaults = {
    "symbol": "AAPL", "outlook_key": "bullish", "risk_tol_key": "conservative",
    "portfolio_value": 25000, "risk_per_trade": 0.02, "risk_free_rate": 0.045,
    "holdings": [], "editing_holding": None, "watchlist": [],
    "sidebar_chat_messages": [], "cash_balance": 0.0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

symbol = st.session_state.symbol
outlook_key = st.session_state.outlook_key
risk_tol_key = st.session_state.risk_tol_key
portfolio_value = st.session_state.portfolio_value
risk_per_trade = st.session_state.risk_per_trade
risk_free_rate = st.session_state.risk_free_rate

# ─── Custom CSS ───

st.markdown("""
<style>
    .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] {
        min-width: 360px; max-width: 420px;
        background: linear-gradient(180deg, #0E1117 0%, #131620 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.15);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px 8px 0 0; font-weight: 500; font-size: 0.9em; }
    [data-testid="stMetric"] {
        background-color: rgba(108, 99, 255, 0.06); border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 10px; padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.8em; opacity: 0.7; }
    .streamlit-expanderHeader { font-weight: 600; font-size: 0.95em; border-radius: 8px; }
    .stButton > button { border-radius: 8px; font-weight: 500; transition: all 0.2s ease; }
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, #6C63FF 0%, #5A4FE0 100%); border: none; }
    .stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #7B73FF 0%, #6C63FF 100%); box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3); }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .chat-msg-user { background: linear-gradient(135deg, #6C63FF 0%, #5A4FE0 100%); color: white; padding: 10px 14px; border-radius: 16px 16px 4px 16px; margin: 6px 0 6px 15%; font-size: 0.88em; line-height: 1.5; box-shadow: 0 2px 8px rgba(108, 99, 255, 0.2); }
    .chat-msg-advisor { background-color: #1A1D26; color: #E8E8EC; padding: 10px 14px; border-radius: 16px 16px 16px 4px; margin: 6px 15% 6px 0; font-size: 0.88em; line-height: 1.5; border: 1px solid rgba(108, 99, 255, 0.15); }
    .chat-label { font-size: 0.7em; color: #666; margin-bottom: 2px; padding-left: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .chat-welcome { text-align: center; padding: 25px 15px; color: #888; }
    .chat-welcome h4 { color: #B8B5FF; margin-bottom: 6px; font-weight: 600; }
    .stAlert { border-radius: 10px; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar — AI Investment Advisor Chat ───

st.sidebar.markdown(
    "<h2 style='text-align:center;margin-bottom:0;'>OptiMax</h2>"
    "<p style='text-align:center;color:#888;font-size:0.85em;margin-top:2px;'>AI Investment Advisor</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

chat_container = st.sidebar.container(height=450)
with chat_container:
    if not st.session_state.sidebar_chat_messages:
        st.markdown(
            '<div class="chat-welcome"><h4>Welcome to OptiMax AI</h4>'
            '<p>Ask me anything about investing, trading strategies, portfolio ideas, or market analysis.</p></div>',
            unsafe_allow_html=True,
        )
        suggestions = ["Best dividend stocks?", "Explain covered calls", "How to diversify my portfolio?", "Bull vs bear spread?"]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"sug_{i}", use_container_width=True):
                    st.session_state.sidebar_chat_messages.append({"role": "user", "content": s})
                    st.session_state.pending_chat = True
                    st.rerun()
    else:
        for msg in st.session_state.sidebar_chat_messages:
            cls = "chat-msg-user" if msg["role"] == "user" else "chat-msg-advisor"
            label = "You" if msg["role"] == "user" else "Advisor"
            st.markdown(f'<div class="chat-label">{label}</div><div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)

user_input = st.sidebar.chat_input("Ask about investments...")

if st.session_state.sidebar_chat_messages:
    if st.sidebar.button("Clear conversation", key="clear_chat", use_container_width=True):
        st.session_state.sidebar_chat_messages = []
        st.rerun()

pending = st.session_state.pop("pending_chat", False)
if user_input and user_input.strip():
    st.session_state.sidebar_chat_messages.append({"role": "user", "content": user_input.strip()})
    pending = True

if pending and st.session_state.sidebar_chat_messages and st.session_state.sidebar_chat_messages[-1]["role"] == "user":
    groq_key = os.getenv("GROQ_API_KEY", "") or st.secrets.get("GROQ_API_KEY", "")
    if not groq_key or groq_key == "your-api-key-here":
        st.sidebar.error("Set your GROQ_API_KEY in the .env file.")
    else:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            context = (f"The user is analyzing {symbol}. Outlook: {outlook_key}, "
                       f"risk tolerance: {risk_tol_key}, portfolio: ${portfolio_value:,}, "
                       f"max risk/trade: {risk_per_trade*100:.1f}%.")
            system_prompt = (
                f"You are an expert AI investment advisor in OptiMax, an options and portfolio tool. "
                f"Help with investment strategies, stock analysis, options, portfolio diversification, market trends. "
                f"Be concise (sidebar format). Use bullet points. Keep under 200 words. "
                f"Context: {context} "
                f"Remind users this is educational, not financial advice."
            )
            messages_for_api = [{"role": "system", "content": system_prompt}]
            messages_for_api.extend({"role": m["role"], "content": m["content"]} for m in st.session_state.sidebar_chat_messages)

            response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages_for_api, max_tokens=1024)
            st.session_state.sidebar_chat_messages.append({"role": "assistant", "content": response.choices[0].message.content})
        except Exception as e:
            st.session_state.sidebar_chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
        st.rerun()

# ─── Load Data (options — lazy) ───

info = None
expirations = []
spot = 0.0
iv_data = None
entropy_signal = None
iv_percentile = 50.0
exp_with_dte = []

if symbol:
    try:
        with st.spinner(f"Fetching data for {symbol}..."):
            info = get_ticker_info(symbol)
            expirations = get_expirations(symbol)
    except Exception:
        info = None
        expirations = []

    if info:
        spot = info["spot"]
        col_title, col_price = st.columns([3, 1])
        with col_title:
            st.title(f"{info['name']} ({symbol})")
        with col_price:
            st.metric("Spot Price", f"${spot:,.2f}")

        with st.spinner("Loading options data..."):
            iv_data = compute_iv_percentile(symbol)
            entropy_signal = compute_market_entropy(lookback_days=90)

        if iv_data:
            iv_percentile = iv_data["iv_percentile"]

        for exp in expirations:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
            exp_with_dte.append(f"{exp}  ({dte}d)")

# ─── Main Tabs ───

tab_portfolio, tab_calendar, tab_options, tab_entropy = st.tabs([
    "Portfolio Manager", "Calendar & News", "Options Trading", "Entropy Analysis",
])

# ═══════════════════════════════════════
# TAB: Portfolio Manager
# ═══════════════════════════════════════

with tab_portfolio:
    st.subheader("Portfolio Manager")
    st.caption("Track your stocks, ETFs, bonds, and REITs with live prices and dividend yields.")

    # ── Add New Holdings ──
    st.markdown("### Add Holdings")

    add_method = st.radio("Add method", ["Single Entry", "Bulk Add (Multiple)", "Quick Add Popular"], horizontal=True, key="add_method")

    if add_method == "Single Entry":
        col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
        with col1:
            new_symbol = st.text_input("Ticker Symbol", placeholder="AAPL", key="single_ticker_input", help="Enter ticker symbol (e.g., AAPL, MSFT, SPY)").upper().strip()
        with col2:
            new_shares = st.number_input("Shares", min_value=0.0, value=10.0, step=1.0, key="new_shares")
        with col3:
            new_cost = st.number_input("Avg Cost ($)", min_value=0.0, value=0.0, step=1.0, key="new_cost", help="Leave at 0 to use current price")
        with col4:
            new_date = st.date_input("Purchase Date", value=datetime.now().date(), key="new_date")
        with col5:
            st.markdown("<br>", unsafe_allow_html=True)
            add_btn = st.button("Add", type="primary", key="add_btn")

        if add_btn and new_symbol:
            with st.spinner(f"Fetching data for {new_symbol}..."):
                data = fetch_security_data(new_symbol)
                if data:
                    cost = new_cost if new_cost > 0 else data["current_price"]
                    holding = create_holding(new_symbol, new_shares, cost, datetime.combine(new_date, datetime.min.time()))
                    if holding:
                        existing_idx = next((i for i, h in enumerate(st.session_state.holdings) if h.symbol == new_symbol), None)
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
        st.markdown("**Enter multiple holdings** — one per line: `TICKER, SHARES, COST, DATE`\n\n"
                     "Example: `AAPL, 50, 175.00, 2023-06-15`\n\n*Cost and Date are optional*")
        bulk_input = st.text_area("Holdings (TICKER, SHARES, COST, DATE)", height=150,
                                   placeholder="AAPL, 50, 175.00, 2023-06-15\nMSFT, 30\nSPY, 100", key="bulk_input")

        if st.button("Add All Holdings", type="primary", key="bulk_add_btn") and bulk_input.strip():
            lines = [l.strip() for l in bulk_input.strip().split("\n") if l.strip()]
            success_count, errors = 0, []
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
                    try:
                        purchase_dt = datetime.strptime(parts[3].strip(), "%Y-%m-%d") if len(parts) > 3 and parts[3] else datetime.now()
                    except (ValueError, IndexError):
                        purchase_dt = datetime.now()

                    status_text.text(f"Processing {sym}...")
                    data = fetch_security_data(sym)
                    if data:
                        holding = create_holding(sym, shares, cost if cost > 0 else data["current_price"], purchase_dt)
                        if holding:
                            existing_idx = next((j for j, h in enumerate(st.session_state.holdings) if h.symbol == sym), None)
                            if existing_idx is not None:
                                st.session_state.holdings[existing_idx] = holding
                            else:
                                st.session_state.holdings.append(holding)
                            success_count += 1
                    else:
                        errors.append(f"{sym}: Could not fetch data")
                except ValueError:
                    errors.append(f"Line '{line}': Invalid format")
                progress_bar.progress((i + 1) / len(lines))

            status_text.empty()
            progress_bar.empty()
            if success_count > 0:
                st.success(f"Successfully added/updated {success_count} holdings!")
            if errors:
                with st.expander(f"{len(errors)} error(s)", expanded=True):
                    for err in errors:
                        st.warning(err)
            if success_count > 0:
                st.rerun()

    else:  # Quick Add Popular
        qa_col1, qa_col2 = st.columns(2)
        with qa_col1:
            qa_shares = st.number_input("Shares to add", min_value=0.01, value=10.0, step=1.0, key="qa_shares")
        with qa_col2:
            qa_cost = st.number_input("Avg Cost ($/share)", min_value=0.0, value=0.0, step=1.0, key="qa_cost", help="Leave at 0 for current price")

        st.caption("Click a ticker to add with the shares and cost above")
        pop_tabs = st.tabs(["Stocks", "ETFs", "Bond ETFs", "REITs"])
        pop_data = [POPULAR_STOCKS, POPULAR_ETFS, POPULAR_BOND_ETFS, POPULAR_REITS]
        pop_keys = ["pop_stock", "pop_etf", "pop_bond", "pop_reit"]
        for tab, items, key in zip(pop_tabs, pop_data, pop_keys):
            with tab:
                add_popular_holdings(items, key, qa_shares, qa_cost)

    st.markdown("---")

    # ── Portfolio Summary ──
    if st.session_state.holdings:
        summary = calculate_portfolio_summary(st.session_state.holdings)

        st.markdown("### Portfolio Summary")
        sum_cols = st.columns(6)
        labels = ["Total Value", "Total P&L", "Annual Income", "Portfolio Yield", "Weighted Beta", "Holdings"]
        values = [
            f"${summary['total_value']:,.2f}",
            f"${summary['total_pnl']:,.2f}",
            f"${summary['annual_income']:,.2f}",
            f"{summary['portfolio_yield']:.2f}%",
            f"{summary['weighted_beta']:.2f}",
            f"{len(st.session_state.holdings)}",
        ]
        deltas = [None, f"{summary['total_pnl_pct']:+.2f}%", None, None, None, None]
        delta_colors = [None, "normal" if summary['total_pnl'] >= 0 else "inverse", None, None, None, None]
        for col, label, val, delta, dc in zip(sum_cols, labels, values, deltas, delta_colors):
            with col:
                if delta:
                    st.metric(label, val, delta=delta, delta_color=dc)
                else:
                    st.metric(label, val)

        # ── Allocation Charts ──
        if len(st.session_state.holdings) > 0:
            alloc_col1, alloc_col2 = st.columns(2)

            with alloc_col1:
                st.markdown("#### Holdings by Weight")
                holdings_for_pie = sorted([{
                    "symbol": h.symbol, "name": h.name,
                    "weight": (h.market_value / summary["total_value"] * 100) if summary["total_value"] > 0 else 0,
                    "value": h.market_value, "color": get_company_color(h.symbol),
                } for h in st.session_state.holdings], key=lambda x: x["weight"], reverse=True)

                if len(holdings_for_pie) > 10:
                    other = holdings_for_pie[9:]
                    holdings_for_pie = holdings_for_pie[:9] + [{
                        "symbol": "Other", "name": f"{len(other)} other holdings",
                        "weight": sum(h["weight"] for h in other),
                        "value": sum(h["value"] for h in other), "color": "#CCCCCC",
                    }]

                fig_hold = go.Figure(data=[go.Pie(
                    labels=[h["symbol"] for h in holdings_for_pie],
                    values=[h["weight"] for h in holdings_for_pie],
                    marker=dict(colors=[h["color"] for h in holdings_for_pie]),
                    hovertemplate="<b>%{label}</b><br>%{customdata}<br>Weight: %{percent}<br>Value: $%{value:,.2f}<extra></extra>",
                    customdata=[h["name"] for h in holdings_for_pie],
                    textinfo="label+percent", textposition="outside",
                    textfont=dict(color="white", size=11), hole=0.3,
                )])
                fig_hold.update_layout(
                    title=dict(text="Portfolio Holdings", font=dict(color="white", size=14)),
                    paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                    font=dict(color="white"), showlegend=False, height=400,
                    margin=dict(t=50, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_hold, use_container_width=True)

                st.caption("**Holding Details:**")
                for h in holdings_for_pie:
                    st.markdown(
                        f"<span style='color:{h['color']};font-weight:bold;'>●</span> "
                        f"**{h['symbol']}** — {h['weight']:.1f}% (${h['value']:,.0f})",
                        unsafe_allow_html=True,
                    )

            with alloc_col2:
                st.markdown("#### By Asset Type")
                if summary["asset_allocation"]:
                    alloc_df = pd.DataFrame(list(summary["asset_allocation"].items()), columns=["Asset Type", "Weight (%)"]).sort_values("Weight (%)", ascending=False)
                    type_colors = {"Stock": "#1f77b4", "ETF": "#2ca02c", "Bond ETF": "#ff7f0e", "REIT": "#9467bd"}

                    fig_type, ax_type = dark_mpl_figure((7, 3))
                    colors_type = [type_colors.get(t, "#666666") for t in alloc_df["Asset Type"]]
                    bars = ax_type.barh(alloc_df["Asset Type"], alloc_df["Weight (%)"], color=colors_type)
                    ax_type.set_xlabel("Weight (%)", color='white')
                    ax_type.set_title("Asset Type Breakdown", color='white')
                    ax_type.tick_params(colors='white')
                    for bar, val in zip(bars, alloc_df["Weight (%)"]):
                        ax_type.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=9, color='white')
                    ax_type.set_xlim(0, max(alloc_df["Weight (%)"]) * 1.2)
                    ax_type.grid(True, alpha=0.3, axis="x")
                    fig_type.tight_layout()
                    st.pyplot(fig_type)
                    plt.close(fig_type)

                if summary["sector_allocation"]:
                    st.markdown("#### By Sector (Stocks Only)")
                    sector_df = pd.DataFrame(list(summary["sector_allocation"].items()), columns=["Sector", "Weight (%)"]).sort_values("Weight (%)", ascending=True)

                    fig_sec, ax_sec = dark_mpl_figure((7, 4))
                    ax_sec.barh(sector_df["Sector"], sector_df["Weight (%)"], color="#1f77b4")
                    ax_sec.set_xlabel("Weight (%)", color='white')
                    ax_sec.set_title("Sector Breakdown", color='white')
                    ax_sec.tick_params(colors='white')
                    for i, (idx, row) in enumerate(sector_df.iterrows()):
                        ax_sec.text(row["Weight (%)"] + 0.5, i, f'{row["Weight (%)"]:.1f}%', va='center', fontsize=8, color='white')
                    ax_sec.grid(True, alpha=0.3, axis="x")
                    fig_sec.tight_layout()
                    st.pyplot(fig_sec)
                    plt.close(fig_sec)

        st.markdown("---")

        # ── Holdings Table ──
        st.markdown("### Holdings")
        if st.button("Refresh Prices", key="refresh_prices"):
            with st.spinner("Refreshing..."):
                st.session_state.holdings = [create_holding(h.symbol, h.shares, h.avg_cost) or h for h in st.session_state.holdings]
                st.rerun()

        holdings_data = [{
            "Symbol": h.symbol,
            "Name": h.name[:25] + "..." if len(h.name) > 25 else h.name,
            "Type": h.asset_type, "Shares": f"{h.shares:,.2f}",
            "Price": f"${h.current_price:,.2f}", "Avg Cost": f"${h.avg_cost:,.2f}",
            "Purchased": h.purchase_date.strftime("%Y-%m-%d") if h.purchase_date else "-",
            "Value": f"${h.market_value:,.2f}", "P&L": f"${h.unrealized_pnl:+,.2f}",
            "P&L %": f"{h.unrealized_pnl_pct:+.2f}%",
            "Div Yield": f"{h.dividend_yield * 100:.2f}%",
            "Income": f"${h.annual_income:,.2f}",
            "Beta": f"{h.beta:.2f}" if h.beta else "-",
        } for h in st.session_state.holdings]

        st.dataframe(pd.DataFrame(holdings_data), use_container_width=True, hide_index=True,
                      height=min(400, 35 * len(holdings_data) + 38))

        # ── Edit / Remove Holdings ──
        with st.expander("Edit Holdings"):
            st.caption("Adjust shares, cost, or purchase date for existing holdings")
            for i, h in enumerate(st.session_state.holdings):
                edit_cols = st.columns([0.8, 1, 1, 1.2, 0.8])
                with edit_cols[0]:
                    st.markdown(f"**{h.symbol}**")
                with edit_cols[1]:
                    edit_shares = st.number_input("Shares", min_value=0.01, value=float(h.shares), step=1.0, key=f"edit_shares_{h.symbol}_{i}")
                with edit_cols[2]:
                    edit_cost = st.number_input("Avg Cost", min_value=0.01, value=float(h.avg_cost), step=1.0, key=f"edit_cost_{h.symbol}_{i}")
                with edit_cols[3]:
                    edit_date = st.date_input("Purchase Date", value=h.purchase_date.date() if h.purchase_date else datetime.now().date(), key=f"edit_date_{h.symbol}_{i}")
                with edit_cols[4]:
                    if st.button("Save", key=f"update_{h.symbol}_{i}"):
                        updated = create_holding(h.symbol, edit_shares, edit_cost, datetime.combine(edit_date, datetime.min.time()))
                        if updated:
                            st.session_state.holdings[i] = updated
                            st.success(f"Updated {h.symbol}")
                            st.rerun()

        with st.expander("Remove Holdings"):
            del_cols = st.columns(6)
            for i, h in enumerate(st.session_state.holdings):
                with del_cols[i % 6]:
                    if st.button(f"X {h.symbol}", key=f"del_{h.symbol}_{i}"):
                        st.session_state.holdings.pop(i)
                        st.rerun()
            if st.button("Clear All Holdings", type="secondary", key="clear_all"):
                st.session_state.holdings = []
                st.rerun()

        st.markdown("---")

        # ── Performance Charts ──
        st.markdown("### Performance Charts")
        if "selected_perf_stock" not in st.session_state:
            st.session_state.selected_perf_stock = None

        perf_tab1, perf_tab2 = st.tabs(["Individual Stock", "Portfolio Performance"])

        with perf_tab1:
            st.caption("Click a stock to view its price performance")
            stock_cols = st.columns(min(len(st.session_state.holdings), 8))
            for i, h in enumerate(st.session_state.holdings):
                with stock_cols[i % 8]:
                    btn_style = "primary" if st.session_state.selected_perf_stock == h.symbol else "secondary"
                    if st.button(h.symbol, key=f"perf_btn_{h.symbol}", type=btn_style):
                        st.session_state.selected_perf_stock = h.symbol
                        st.rerun()

            perf_period = st.selectbox("Time Period", list(PERIOD_MAP.keys()), index=3, key="perf_period")

            if st.session_state.selected_perf_stock:
                selected_sym = st.session_state.selected_perf_stock
                selected_holding = next((h for h in st.session_state.holdings if h.symbol == selected_sym), None)

                if selected_holding:
                    with st.spinner(f"Loading {selected_sym} performance..."):
                        if perf_period == "All Time" and selected_holding.purchase_date:
                            perf_data = get_stock_performance(selected_sym, period="max", start_date=selected_holding.purchase_date)
                        else:
                            perf_data = get_stock_performance(selected_sym, PERIOD_MAP[perf_period])

                    if perf_data:
                        info_cols = st.columns(5)
                        with info_cols[0]:
                            st.metric("Current Price", f"${selected_holding.current_price:,.2f}")
                        with info_cols[1]:
                            st.metric(f"{perf_period} Return", f"{perf_data['total_return']:+.2f}%",
                                      delta_color="normal" if perf_data['total_return'] >= 0 else "inverse")
                        with info_cols[2]:
                            st.metric("Period High", f"${perf_data['period_high']:,.2f}")
                        with info_cols[3]:
                            st.metric("Period Low", f"${perf_data['period_low']:,.2f}")
                        with info_cols[4]:
                            if selected_holding.purchase_date:
                                st.metric("Purchased", selected_holding.purchase_date.strftime("%b %d, %Y"))
                            else:
                                st.metric("Days", f"{perf_data['days']}")

                        hist = perf_data["history"]
                        color = "#00ff00" if perf_data['total_return'] >= 0 else "#ff4444"

                        fig_stock = go.Figure()
                        fig_stock.add_trace(go.Scatter(
                            x=hist["Date"], y=hist["Close"], mode='lines', name='Price',
                            line=dict(color=color, width=2), fill='tozeroy',
                            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
                            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Price: $%{y:.2f}<extra></extra>',
                        ))
                        if selected_holding.avg_cost > 0:
                            fig_stock.add_hline(y=selected_holding.avg_cost, line_dash="dash", line_color="#ffaa00",
                                                annotation_text=f"Avg Cost: ${selected_holding.avg_cost:.2f}", annotation_position="right")

                        dark_plotly_layout(fig_stock, f"{selected_holding.name} ({selected_sym}) - {perf_period}",
                                           "Date", "Price ($)", template="plotly_dark", showlegend=False)
                        st.plotly_chart(fig_stock, use_container_width=True)

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
                                st.metric("Your P&L", f"${selected_holding.unrealized_pnl:+,.2f}",
                                          delta=f"{selected_holding.unrealized_pnl_pct:+.2f}%",
                                          delta_color="normal" if selected_holding.unrealized_pnl >= 0 else "inverse")
                    else:
                        st.warning(f"Could not load performance data for {selected_sym}")
            else:
                st.info("Click a stock above to view its performance chart")

        with perf_tab2:
            st.caption("Track your portfolio performance vs. S&P 500 benchmark")
            port_period = st.selectbox("Time Period", ["1M", "3M", "6M", "1Y", "All Time"], index=2, key="port_perf_period")

            with st.spinner("Calculating portfolio performance..."):
                port_perf = get_portfolio_performance(st.session_state.holdings, PERIOD_MAP.get(port_period, "6mo"))

            if port_perf:
                port_cols = st.columns(5)
                with port_cols[0]:
                    st.metric("Total Value", f"${port_perf['end_value']:,.2f}")
                with port_cols[1]:
                    st.metric("Total Cost", f"${port_perf['total_cost']:,.2f}")
                with port_cols[2]:
                    st.metric("Total P&L", f"${port_perf['total_pnl']:+,.2f}",
                              delta=f"{port_perf['total_return']:+.2f}%",
                              delta_color="normal" if port_perf['total_pnl'] >= 0 else "inverse")
                with port_cols[3]:
                    st.metric("Period", f"{port_perf['days']} days")
                with port_cols[4]:
                    if "SPY_Return" in port_perf["history"].columns:
                        spy_return = port_perf["history"]["SPY_Return"].iloc[-1]
                        alpha = port_perf['total_return'] - spy_return
                        st.metric("Alpha vs SPY", f"{alpha:+.2f}%",
                                  delta="Outperforming" if alpha > 0 else "Underperforming",
                                  delta_color="normal" if alpha >= 0 else "inverse")

                hist = port_perf["history"]
                port_color = "#00ff00" if port_perf['total_return'] >= 0 else "#ff4444"
                fig_port = go.Figure()
                fig_port.add_trace(go.Scatter(
                    x=hist["Date"], y=hist["Portfolio_Return"], mode='lines',
                    name=f"Your Portfolio ({port_perf['total_return']:+.2f}%)",
                    line=dict(color=port_color, width=2),
                    hovertemplate='<b>%{x|%b %d, %Y}</b><br>Portfolio: %{y:+.2f}%<extra></extra>',
                ))
                if "SPY_Return" in hist.columns:
                    spy_final = hist["SPY_Return"].iloc[-1]
                    fig_port.add_trace(go.Scatter(
                        x=hist["Date"], y=hist["SPY_Return"], mode='lines',
                        name=f"S&P 500 ({spy_final:+.2f}%)",
                        line=dict(color="#888888", width=2, dash='dash'),
                        hovertemplate='<b>%{x|%b %d, %Y}</b><br>S&P 500: %{y:+.2f}%<extra></extra>',
                    ))
                fig_port.add_hline(y=0, line_color="white", line_width=0.5, opacity=0.5)
                dark_plotly_layout(fig_port, f"Portfolio Performance - {port_period}", "Date", "Return (%)",
                                   template="plotly_dark", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                st.plotly_chart(fig_port, use_container_width=True)
            else:
                st.warning("Could not calculate portfolio performance. Make sure you have holdings with valid data.")

        st.markdown("---")

        # ── Top Dividend Payers ──
        div_holdings = sorted([h for h in st.session_state.holdings if h.dividend_yield > 0], key=lambda x: x.dividend_yield, reverse=True)[:5]
        if div_holdings:
            st.markdown("### Top Dividend Payers")
            div_cols = st.columns(len(div_holdings))
            for i, h in enumerate(div_holdings):
                with div_cols[i]:
                    st.metric(h.symbol, f"{h.dividend_yield * 100:.2f}%", delta=f"${h.annual_income:,.0f}/yr", delta_color="off")

        st.markdown("---")

        # ── Dividend Income Tools ──
        st.markdown("### Dividend Income Tools")
        div_tool_tab1, div_tool_tab2 = st.tabs(["Dividend History", "DRIP Calculator"])

        with div_tool_tab1:
            st.caption("Actual dividend payments received based on your holdings and purchase dates")
            with st.spinner("Loading dividend history..."):
                div_history = get_dividend_payment_history(st.session_state.holdings, years=3)

            if not div_history.empty:
                monthly_divs = get_monthly_dividend_totals(st.session_state.holdings, years=2)
                if not monthly_divs.empty:
                    fig_div_hist = go.Figure()
                    fig_div_hist.add_trace(go.Bar(
                        x=monthly_divs["Month"], y=monthly_divs["Total"], marker_color="#28a745",
                        text=[f"${v:,.0f}" for v in monthly_divs["Total"]], textposition='outside',
                        textfont=dict(color='white', size=9),
                        hovertemplate='<b>%{x}</b><br>Dividends: $%{y:,.2f}<extra></extra>',
                    ))
                    dark_plotly_layout(fig_div_hist, "Historical Dividend Income", "Month", "Dividends Received ($)",
                                       xaxis=dict(tickangle=45, gridcolor='rgba(255,255,255,0.1)'),
                                       yaxis=dict(gridcolor='rgba(255,255,255,0.2)'), margin=dict(t=50, b=80))
                    st.plotly_chart(fig_div_hist, use_container_width=True)

                total_received = div_history["Total"].sum()
                avg_monthly = total_received / max(len(monthly_divs) if not monthly_divs.empty else 1, 1)
                st.success(f"**Total Dividends Received: ${total_received:,.2f}** | Average: ${avg_monthly:,.2f}/month")

                st.markdown("**Recent Dividend Payments:**")
                recent_divs = div_history.head(15).copy()
                recent_divs["Date"] = recent_divs["Date"].dt.strftime("%Y-%m-%d")
                recent_divs["Amount"] = recent_divs["Amount"].apply(lambda x: f"${x:.4f}")
                recent_divs["Total"] = recent_divs["Total"].apply(lambda x: f"${x:.2f}")
                recent_divs["Shares"] = recent_divs["Shares"].apply(lambda x: f"{x:,.0f}")
                st.dataframe(recent_divs, use_container_width=True, hide_index=True)
            else:
                st.info("No dividend history available yet. Dividends will appear here after ex-dates pass.")

        with div_tool_tab2:
            st.caption("See the power of dividend reinvestment over time")
            drip_col1, drip_col2 = st.columns(2)
            with drip_col1:
                drip_symbols = [h.symbol for h in st.session_state.holdings if h.dividend_yield > 0]
                drip_holding = None
                if drip_symbols:
                    drip_symbol = st.selectbox("Select Holding", drip_symbols, key="drip_symbol")
                    drip_holding = next((h for h in st.session_state.holdings if h.symbol == drip_symbol), None)
                else:
                    st.warning("No dividend-paying holdings in portfolio")
            with drip_col2:
                drip_years = st.slider("Projection Years", 5, 30, 10, key="drip_years")

            if drip_holding:
                drip_col3, drip_col4 = st.columns(2)
                with drip_col3:
                    price_growth = st.slider("Annual Price Growth %", 0, 15, 5, key="drip_price_growth") / 100
                with drip_col4:
                    div_growth = st.slider("Annual Dividend Growth %", 0, 15, 3, key="drip_div_growth") / 100

                initial_value = drip_holding.market_value
                drip_result = calculate_drip_vs_no_drip(
                    initial_investment=initial_value, share_price=drip_holding.current_price,
                    dividend_yield=drip_holding.dividend_yield, years=drip_years,
                    annual_price_growth=price_growth, annual_dividend_growth=div_growth,
                )

                st.markdown(f"**{drip_symbol} DRIP Projection ({drip_years} years)**")
                drip_sum_cols = st.columns(4)
                with drip_sum_cols[0]:
                    st.metric("Starting Value", f"${initial_value:,.0f}")
                with drip_sum_cols[1]:
                    st.metric("With DRIP", f"${drip_result['drip_final_value']:,.0f}")
                with drip_sum_cols[2]:
                    st.metric("Without DRIP", f"${drip_result['no_drip_final_value']:,.0f}")
                with drip_sum_cols[3]:
                    st.metric("DRIP Advantage", f"${drip_result['drip_advantage']:,.0f}",
                              delta=f"+{drip_result['drip_advantage_pct']:.1f}%", delta_color="normal")

                drip_df = drip_result["drip"]
                no_drip_df = drip_result["no_drip"]
                fig_drip = go.Figure()
                fig_drip.add_trace(go.Scatter(
                    x=drip_df["Year"], y=drip_df["Portfolio Value"], mode='lines', name='With DRIP',
                    line=dict(color='#00ff00', width=3), fill='tonexty', fillcolor='rgba(0,255,0,0.1)',
                    hovertemplate='<b>Year %{x}</b><br>With DRIP: $%{y:,.0f}<br>Shares: %{customdata:,.2f}<extra></extra>',
                    customdata=drip_df["Shares"],
                ))
                fig_drip.add_trace(go.Scatter(
                    x=no_drip_df["Year"], y=no_drip_df["Portfolio Value"] + no_drip_df["Cumulative Dividends"],
                    mode='lines', name='Without DRIP', line=dict(color='#ff6666', width=3, dash='dash'),
                    hovertemplate='<b>Year %{x}</b><br>Without DRIP: $%{y:,.0f}<br>Cash Dividends: $%{customdata:,.0f}<extra></extra>',
                    customdata=no_drip_df["Cumulative Dividends"],
                ))
                dark_plotly_layout(fig_drip, f"{drip_symbol} - DRIP vs No DRIP Projection", "Years", "Total Value ($)",
                                   height=450, yaxis=dict(gridcolor='rgba(255,255,255,0.2)', tickformat='$,.0f'),
                                   xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                                   legend=dict(bgcolor=CHART_FACE, font=dict(color='white')))
                st.plotly_chart(fig_drip, use_container_width=True)
                st.caption(f"Shares: {drip_holding.shares:,.2f} -> {drip_result['drip_final_shares']:,.2f} with DRIP")

    # ── Cash & Money Market ──
    st.markdown("---")
    st.markdown("### Cash & Money Market")
    cash_tab1, cash_tab2 = st.tabs(["Cash Holdings", "Money Market Calculator"])

    with cash_tab1:
        st.caption("Track your cash position in the portfolio")
        cash_col1, cash_col2, cash_col3 = st.columns([2, 1, 1])
        with cash_col1:
            new_cash = st.number_input("Cash Balance ($)", min_value=0.0, value=st.session_state.cash_balance, step=100.0, key="cash_input")
        with cash_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Update Cash", key="update_cash"):
                st.session_state.cash_balance = new_cash
                st.success(f"Cash updated to ${new_cash:,.2f}")
                st.rerun()
        with cash_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Clear Cash", key="clear_cash"):
                st.session_state.cash_balance = 0.0
                st.rerun()

        if st.session_state.cash_balance > 0 or st.session_state.holdings:
            total_invested = sum(h.market_value for h in st.session_state.holdings) if st.session_state.holdings else 0
            total_portfolio = total_invested + st.session_state.cash_balance
            cash_pct = (st.session_state.cash_balance / total_portfolio * 100) if total_portfolio > 0 else 0
            cash_metrics = st.columns(4)
            with cash_metrics[0]:
                st.metric("Cash Balance", f"${st.session_state.cash_balance:,.2f}")
            with cash_metrics[1]:
                st.metric("Invested", f"${total_invested:,.2f}")
            with cash_metrics[2]:
                st.metric("Total Portfolio", f"${total_portfolio:,.2f}")
            with cash_metrics[3]:
                st.metric("Cash %", f"{cash_pct:.1f}%")

    with cash_tab2:
        st.caption("Calculate money market account growth with interest and contributions")
        mm_col1, mm_col2 = st.columns(2)
        with mm_col1:
            mm_initial = st.number_input("Initial Deposit ($)", min_value=0.0, value=10000.0, step=1000.0, key="mm_initial")
            mm_rate = st.slider("Annual Interest Rate (%)", min_value=0.0, max_value=10.0, value=4.5, step=0.1, key="mm_rate")
        with mm_col2:
            mm_contribution = st.number_input("Monthly Contribution ($)", min_value=0.0, value=500.0, step=100.0, key="mm_contribution")
            mm_years = st.slider("Time Period (Years)", min_value=1, max_value=30, value=5, key="mm_years")

        if st.button("Calculate Growth", key="calc_mm"):
            monthly_rate = mm_rate / 100 / 12
            months = mm_years * 12
            balance_history = [mm_initial]
            balance = mm_initial
            total_contributions = mm_initial
            total_interest = 0.0

            for month in range(1, months + 1):
                interest = balance * monthly_rate
                total_interest += interest
                balance += interest + mm_contribution
                total_contributions += mm_contribution
                if month % 12 == 0:
                    balance_history.append(balance)

            result_cols = st.columns(4)
            with result_cols[0]:
                st.metric("Final Balance", f"${balance:,.2f}")
            with result_cols[1]:
                st.metric("Total Contributions", f"${total_contributions:,.2f}")
            with result_cols[2]:
                st.metric("Total Interest Earned", f"${total_interest:,.2f}")
            with result_cols[3]:
                growth_pct = ((balance - mm_initial) / mm_initial * 100) if mm_initial > 0 else 0
                st.metric("Total Growth", f"{growth_pct:.1f}%")

            years_list = list(range(mm_years + 1))
            fig_mm = go.Figure()
            fig_mm.add_trace(go.Scatter(
                x=years_list, y=balance_history, mode='lines+markers', name='Balance',
                line=dict(color='#4da6ff', width=3), fill='tozeroy', fillcolor='rgba(77,166,255,0.2)',
                hovertemplate='<b>Year %{x}</b><br>Balance: $%{y:,.2f}<extra></extra>',
            ))
            fig_mm.add_trace(go.Scatter(
                x=years_list, y=[mm_initial + (mm_contribution * 12 * y) for y in years_list],
                mode='lines', name='Contributions Only',
                line=dict(color='#888888', width=2, dash='dash'),
                hovertemplate='<b>Year %{x}</b><br>Contributions: $%{y:,.2f}<extra></extra>',
            ))
            dark_plotly_layout(fig_mm, f"Money Market Growth ({mm_rate}% APY)", "Years", "Balance ($)",
                               legend=dict(bgcolor=CHART_FACE, font=dict(color='white')))
            st.plotly_chart(fig_mm, use_container_width=True)

            st.info(f"At {mm_rate}% APY with ${mm_contribution:,.0f}/month, "
                    f"your ${mm_initial:,.0f} grows to **${balance:,.0f}** in {mm_years} years. "
                    f"Interest earned: **${total_interest:,.0f}**")

    # ── Import / Export ──
    st.markdown("---")
    st.markdown("### Import / Export")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("**Export Portfolio**")
        if st.session_state.holdings:
            st.download_button("Download Holdings CSV", export_holdings_to_csv(st.session_state.holdings),
                               file_name="portfolio_holdings.csv", mime="text/csv", key="export_holdings")
        else:
            st.caption("No holdings to export")
        if st.session_state.watchlist:
            st.download_button("Download Watchlist CSV", export_watchlist_to_csv(st.session_state.watchlist),
                               file_name="watchlist.csv", mime="text/csv", key="export_watchlist")

    with exp_col2:
        st.markdown("**Import Portfolio**")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="import_file",
                                          help="Upload a CSV or Excel file with columns: Symbol, Shares, Cost, Date, Notes")
        if uploaded_file:
            file_name = uploaded_file.name.lower()
            parsed = None

            if file_name.endswith(".csv"):
                parsed = import_holdings_from_csv(uploaded_file.read().decode("utf-8"))
            elif file_name.endswith((".xlsx", ".xls")):
                try:
                    df = pd.read_excel(uploaded_file)
                    st.markdown("**Preview:**")
                    st.dataframe(df.head(10), use_container_width=True)

                    col_map = {}
                    col_aliases = {
                        "symbol": ("symbol", "ticker", "stock", "sym"),
                        "shares": ("shares", "quantity", "qty", "units", "amount"),
                        "cost": ("cost", "price", "avg cost", "avg_cost", "cost basis", "cost_basis", "purchase price", "buy price"),
                        "date": ("date", "purchase date", "purchase_date", "buy date", "buy_date"),
                        "notes": ("notes", "note", "comment", "comments"),
                    }
                    for col in df.columns:
                        col_lower = str(col).lower().strip()
                        for field, aliases in col_aliases.items():
                            if col_lower in aliases:
                                col_map[field] = col

                    if "symbol" in col_map and "shares" in col_map:
                        st.success(f"Detected {len(df)} rows")
                        parsed = []
                        for _, row in df.iterrows():
                            sym = str(row[col_map["symbol"]]).upper().strip()
                            if not sym or sym == "NAN":
                                continue
                            shares = float(row[col_map["shares"]]) if pd.notna(row[col_map["shares"]]) else 0
                            cost = float(row[col_map.get("cost", "")]) if "cost" in col_map and pd.notna(row.get(col_map["cost"])) else 0
                            try:
                                date_val = pd.to_datetime(row[col_map["date"]]) if "date" in col_map and pd.notna(row.get(col_map["date"])) else datetime.now()
                            except Exception:
                                date_val = datetime.now()
                            notes = str(row[col_map["notes"]]) if "notes" in col_map and pd.notna(row.get(col_map["notes"])) else None
                            parsed.append({"symbol": sym, "shares": shares, "cost": cost, "purchase_date": date_val, "notes": notes})
                    else:
                        st.warning("Could not auto-detect columns. Need **Symbol** (or Ticker) and **Shares** (or Quantity).")
                except ImportError:
                    st.error("Excel support requires openpyxl. Run: `pip install openpyxl`")
                except Exception as e:
                    st.error(f"Error reading Excel file: {e}")

            if parsed:
                st.success(f"Found {len(parsed)} holdings")
                if st.button("Import All", key="do_import"):
                    with st.spinner("Importing..."):
                        success = 0
                        for item in parsed:
                            holding = create_holding(item["symbol"], item["shares"], item["cost"], item["purchase_date"], item.get("notes"))
                            if holding:
                                existing_idx = next((i for i, h in enumerate(st.session_state.holdings) if h.symbol == item["symbol"]), None)
                                if existing_idx is not None:
                                    st.session_state.holdings[existing_idx] = holding
                                else:
                                    st.session_state.holdings.append(holding)
                                success += 1
                        st.success(f"Imported {success} holdings!")
                        st.rerun()
            elif uploaded_file and file_name.endswith(".csv"):
                st.warning("Could not parse CSV. Use format: Symbol,Shares,Cost,Date,Notes")


# ═══════════════════════════════════════
# TAB: Calendar & News
# ═══════════════════════════════════════

with tab_calendar:
    st.subheader("Calendar & News")
    st.caption("Dividend calendar, earnings dates, Fed events, and news for your holdings")

    if not st.session_state.holdings:
        st.info("Add holdings in the Portfolio Manager tab to see calendar events and news.")
    else:
        st.markdown("### Dividend & Earnings Calendar")

        # Legend
        legend_items = [
            ("#d4edda", "Dividend"), ("#fff3cd", "Earnings"), ("#f8d7da", "Fed/Econ"), ("#cce5ff", "Multiple"),
        ]
        legend_cols = st.columns(4)
        icons = ["🟢", "🟡", "🔴", "🔵"]
        for col, (bg, label), icon in zip(legend_cols, legend_items, icons):
            with col:
                st.markdown(f"<span style='background-color:{bg};padding:2px 8px;border-radius:3px;'>{icon} {label}</span>", unsafe_allow_html=True)

        if "cal_year" not in st.session_state:
            st.session_state.cal_year = datetime.now().year
        if "cal_month" not in st.session_state:
            st.session_state.cal_month = datetime.now().month

        nav_col1, nav_col2, nav_col3, _, nav_col5 = st.columns([1, 1, 2, 1, 1])
        with nav_col1:
            if st.button("< Prev", key="cal_prev_month"):
                if st.session_state.cal_month == 1:
                    st.session_state.cal_month = 12
                    st.session_state.cal_year -= 1
                else:
                    st.session_state.cal_month -= 1
                st.rerun()
        with nav_col2:
            if st.button("Today", key="cal_today"):
                st.session_state.cal_year = datetime.now().year
                st.session_state.cal_month = datetime.now().month
                st.rerun()
        with nav_col3:
            month_name = cal_module.month_name[st.session_state.cal_month]
            st.markdown(f"<h3 style='text-align:center;margin:0;'>{month_name} {st.session_state.cal_year}</h3>", unsafe_allow_html=True)
        with nav_col5:
            if st.button("Next >", key="cal_next_month"):
                if st.session_state.cal_month == 12:
                    st.session_state.cal_month = 1
                    st.session_state.cal_year += 1
                else:
                    st.session_state.cal_month += 1
                st.rerun()

        with st.spinner("Loading calendar..."):
            cal_data = build_combined_calendar(st.session_state.holdings, st.session_state.cal_year, st.session_state.cal_month)

        # Month summary
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            if cal_data["dividend_events"]:
                total_div = sum(e.expected_income for e in cal_data["dividend_events"])
                st.success(f"**Dividends: ${total_div:,.2f}** from {len(cal_data['dividend_events'])} payment(s)")
            else:
                st.info("No dividends this month")
        with sum_col2:
            if cal_data["earnings_events"]:
                confirmed = sum(1 for e in cal_data["earnings_events"] if e.is_confirmed)
                st.warning(f"**Earnings: {len(cal_data['earnings_events'])} report(s)** ({confirmed} confirmed, {len(cal_data['earnings_events']) - confirmed} estimated)")
            else:
                st.info("No earnings reports this month")
        with sum_col3:
            if cal_data.get("fed_events"):
                high_imp = sum(1 for e in cal_data["fed_events"] if e.importance == "HIGH")
                st.error(f"**Fed/Econ: {len(cal_data['fed_events'])} event(s)** ({high_imp} high importance)")
            else:
                st.info("No Fed/economic events this month")

        # Calendar grid
        st.markdown("#### Calendar View")
        header_cols = st.columns(7)
        for i, day_name in enumerate(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]):
            with header_cols[i]:
                st.markdown(f"**{day_name}**")

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
                        has_div, has_earn, has_fed = bool(div_events), bool(earn_events), bool(fed_events)
                        event_count = sum([has_div, has_earn, has_fed])

                        if event_count >= 2:
                            lines = []
                            if has_div:
                                d = [f"{e.symbol}: ${e.expected_income:.0f}" for e in div_events[:2]]
                                if len(div_events) > 2: d.append(f"+{len(div_events)-2}")
                                lines.append(f"💵 {', '.join(d)}")
                            if has_earn:
                                lines.append(f"📊 {', '.join(e.symbol for e in earn_events[:2])}")
                            if has_fed:
                                lines.append(f"🏛 {', '.join(e.event_type for e in fed_events[:2])}")
                            content = "<br>".join(f"<span style='font-size:0.65em;color:#000;font-weight:bold;'>{l}</span>" for l in lines)
                            st.markdown(f"<div style='background:#cce5ff;padding:4px;border-radius:4px;text-align:center;min-height:70px;color:#000;'><strong style='color:#000;'>{day_num}</strong><br>{content}</div>", unsafe_allow_html=True)
                        elif has_div:
                            total_income = sum(e.expected_income for e in div_events)
                            div_lines = [f"{e.symbol}: ${e.expected_income:.2f}" for e in div_events[:3]]
                            if len(div_events) > 3: div_lines.append(f"+{len(div_events)-3}")
                            div_content = "<br>".join(f"<span style='font-size:0.65em;color:#000;'>{l}</span>" for l in div_lines)
                            st.markdown(f"<div style='background:#d4edda;padding:4px;border-radius:4px;text-align:center;min-height:70px;color:#000;'><strong style='color:#000;'>{day_num}</strong><br><span style='font-size:0.7em;color:#000;font-weight:bold;'>💵 ${total_income:,.0f}</span><br>{div_content}</div>", unsafe_allow_html=True)
                        elif has_earn:
                            symbols = ", ".join(e.symbol for e in earn_events)
                            mark = "✓" if all(e.is_confirmed for e in earn_events) else "?"
                            st.markdown(f"<div style='background:#fff3cd;padding:4px;border-radius:4px;text-align:center;min-height:70px;color:#000;'><strong style='color:#000;'>{day_num}</strong><br><span style='font-size:0.75em;color:#000;font-weight:bold;'>📊 {symbols}</span><br><span style='font-size:0.7em;color:#000;'>{mark}</span></div>", unsafe_allow_html=True)
                        elif has_fed:
                            event_types = ", ".join(e.event_type for e in fed_events)
                            imp = "⚠" if any(e.importance == "HIGH" for e in fed_events) else ""
                            st.markdown(f"<div style='background:#f8d7da;padding:4px;border-radius:4px;text-align:center;min-height:70px;color:#000;'><strong style='color:#000;'>{day_num}</strong><br><span style='font-size:0.75em;color:#000;font-weight:bold;'>🏛 {event_types}</span><br><span style='font-size:0.7em;color:#000;'>{imp}</span></div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='padding:4px;text-align:center;min-height:70px;color:#333;'>{day_num}</div>", unsafe_allow_html=True)

        # Detailed events
        if cal_data["dividend_events"]:
            st.markdown("#### Dividend Details")
            st.dataframe(pd.DataFrame([{
                "Ex-Date": e.ex_date.strftime("%b %d, %Y"), "Symbol": e.symbol,
                "Name": e.name[:20] + "..." if len(e.name) > 20 else e.name,
                "Frequency": e.frequency, "Amount/Share": f"${e.amount:.4f}",
                "Shares": f"{e.shares:,.0f}", "Expected Income": f"${e.expected_income:,.2f}",
            } for e in cal_data["dividend_events"]]), use_container_width=True, hide_index=True)

        if cal_data["earnings_events"]:
            st.markdown("#### Earnings Report Details")
            st.dataframe(pd.DataFrame([{
                "Date": e.earnings_date.strftime("%b %d, %Y"), "Symbol": e.symbol,
                "Name": e.name[:20] + "..." if len(e.name) > 20 else e.name,
                "Status": "✓ Confirmed" if e.is_confirmed else "? Estimated",
                "EPS Est.": f"${e.eps_estimate:.2f}" if e.eps_estimate else "-",
            } for e in cal_data["earnings_events"]]), use_container_width=True, hide_index=True)
            st.caption("Note: 'Estimated' dates are projections. Confirm with official announcements.")

        if cal_data.get("fed_events"):
            st.markdown("#### Fed & Economic Events")
            for event in cal_data["fed_events"]:
                badge = "🔴" if event.importance == "HIGH" else "🟡" if event.importance == "MEDIUM" else "🟢"
                col_date, col_event, col_link = st.columns([2, 4, 1])
                with col_date:
                    st.markdown(f"**{event.event_date.strftime('%b %d, %Y')}**")
                with col_event:
                    st.markdown(f"{badge} **{event.event_type}** - {event.description}")
                with col_link:
                    if event.url:
                        st.markdown(f"[Details]({event.url})")
            st.caption("Fed/Economic events can cause significant market volatility.")

        # 12-Month Projection
        st.markdown("#### 12-Month Dividend Income Projection")
        with st.spinner("Calculating projections..."):
            projection = get_annual_dividend_projection(st.session_state.holdings)

        if projection:
            proj_df = pd.DataFrame([{"Month": f"{v['month_name'][:3]} {v['year']}", "Income": v["total"], "Payments": v["event_count"]} for k, v in projection.items()])
            if proj_df["Income"].sum() > 0:
                fig_proj = go.Figure()
                fig_proj.add_trace(go.Bar(
                    x=proj_df["Month"], y=proj_df["Income"], marker_color="#28a745",
                    text=[f"${v:,.0f}" for v in proj_df["Income"]], textposition='outside',
                    textfont=dict(color='white', size=10),
                    hovertemplate='<b>%{x}</b><br>Expected: $%{y:,.2f}<br>Payments: %{customdata}<extra></extra>',
                    customdata=proj_df["Payments"],
                ))
                dark_plotly_layout(fig_proj, "12-Month Dividend Income Projection", "Month", "Expected Income ($)",
                                   xaxis=dict(tickangle=45, gridcolor='rgba(255,255,255,0.1)'),
                                   yaxis=dict(gridcolor='rgba(255,255,255,0.2)'), margin=dict(t=50, b=80))
                st.plotly_chart(fig_proj, use_container_width=True)
                annual_total = proj_df["Income"].sum()
                st.info(f"**Projected Annual Dividend Income: ${annual_total:,.2f}** (${annual_total/12:,.2f}/month average)")

        st.markdown("---")

        # News Feed
        st.markdown("### News Feed")
        st.caption("Recent headlines for your holdings")
        with st.spinner("Loading news..."):
            news_items = get_portfolio_news(st.session_state.holdings, limit_per_stock=3)

        if news_items:
            news_symbols = ["All"] + sorted(set(n.symbol for n in news_items))
            news_filter = st.selectbox("Filter by Stock", news_symbols, key="cal_news_filter")
            filtered_news = news_items if news_filter == "All" else [n for n in news_items if n.symbol == news_filter]
            for news in filtered_news[:20]:
                render_news_card(news, get_company_color(news.symbol), news.symbol)
        else:
            st.info("No recent news found for your holdings")

    # Watchlist
    st.markdown("---")
    st.markdown("### Watchlist")
    st.caption("Track stocks you're interested in but don't own yet")

    watch_col1, watch_col2, watch_col3, watch_col4 = st.columns([1.5, 1, 2, 1])
    with watch_col1:
        watch_symbol = st.text_input("Symbol", placeholder="TSLA", key="cal_watch_symbol").upper().strip()
    with watch_col2:
        watch_target = st.number_input("Target Price ($)", min_value=0.0, value=0.0, key="cal_watch_target")
    with watch_col3:
        watch_notes = st.text_input("Notes", placeholder="Waiting for pullback...", key="cal_watch_notes")
    with watch_col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add to Watchlist", key="cal_add_watch") and watch_symbol:
            with st.spinner(f"Adding {watch_symbol}..."):
                item = create_watchlist_item(watch_symbol, watch_target if watch_target > 0 else None, watch_notes if watch_notes else None)
                if item:
                    st.session_state.watchlist.append(item)
                    st.success(f"Added {watch_symbol} to watchlist")
                    st.rerun()
                else:
                    st.error(f"Could not find {watch_symbol}")

    if st.session_state.watchlist:
        if st.button("Refresh Prices", key="cal_refresh_watchlist"):
            with st.spinner("Refreshing..."):
                st.session_state.watchlist = [refresh_watchlist_item(w) for w in st.session_state.watchlist]
                st.rerun()

        st.dataframe(pd.DataFrame([{
            "Symbol": w.symbol, "Name": w.name[:25] + "..." if len(w.name) > 25 else w.name,
            "Price": f"${w.current_price:.2f}",
            "Target": f"${w.target_price:.2f}" if w.target_price else "-",
            "At Target": "✓" if w.target_price and w.current_price <= w.target_price else "",
            "Change": f"{w.change_since_add_pct:+.1f}%", "Yield": f"{w.dividend_yield*100:.2f}%",
            "52W High": f"${w.fifty_two_week_high:.2f}" if w.fifty_two_week_high else "-",
            "52W Low": f"${w.fifty_two_week_low:.2f}" if w.fifty_two_week_low else "-",
            "Notes": w.notes or "-",
        } for w in st.session_state.watchlist]), use_container_width=True, hide_index=True)

        watch_action_cols = st.columns(min(len(st.session_state.watchlist), 6))
        for i, w in enumerate(st.session_state.watchlist):
            with watch_action_cols[i % 6]:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(f"Buy {w.symbol}", key=f"cal_buy_watch_{w.symbol}_{i}"):
                        holding = create_holding(w.symbol, 10, w.current_price, datetime.now(), w.notes)
                        if holding:
                            st.session_state.holdings.append(holding)
                            st.session_state.watchlist.pop(i)
                            st.rerun()
                with col_b:
                    if st.button("X", key=f"cal_del_watch_{w.symbol}_{i}"):
                        st.session_state.watchlist.pop(i)
                        st.rerun()

        # Watchlist News
        st.markdown("---")
        st.markdown("### Watchlist News")
        st.caption("Recent news for stocks on your watchlist (last 30 days)")
        with st.spinner("Loading watchlist news..."):
            watchlist_news = []
            thirty_days_ago = datetime.now() - relativedelta(days=30)
            for w in st.session_state.watchlist:
                for n in get_stock_news(w.symbol, limit=5):
                    if n.published >= thirty_days_ago:
                        watchlist_news.append(n)
            watchlist_news.sort(key=lambda x: x.published, reverse=True)

        if watchlist_news:
            watch_news_symbols = ["All"] + sorted(set(n.symbol for n in watchlist_news))
            watch_news_filter = st.selectbox("Filter by Stock", watch_news_symbols, key="watchlist_news_filter")
            filtered_watch_news = watchlist_news if watch_news_filter == "All" else [n for n in watchlist_news if n.symbol == watch_news_filter]
            ticker_names = {w.symbol: w.name for w in st.session_state.watchlist}
            for news in filtered_watch_news[:20]:
                render_news_card(news, get_company_color(news.symbol), f"{news.symbol} - {ticker_names.get(news.symbol, news.symbol)[:20]}")
        else:
            st.info("No recent news found for your watchlist stocks")
    else:
        st.info("Your watchlist is empty. Add stocks you're watching above.")


# ═══════════════════════════════════════
# TAB: Options Trading
# ═══════════════════════════════════════

with tab_options:
    st.subheader("Options Trading")

    with st.expander("**Settings** — Ticker, Market View & Position Sizing", expanded=True):
        opt_col1, opt_col2, opt_col3 = st.columns(3)

        with opt_col1:
            st.markdown("**Ticker & Rate**")
            new_symbol = st.text_input("Ticker Symbol", value=st.session_state.symbol, max_chars=10,
                                        help="Enter any US stock ticker", key="opt_symbol").upper().strip()
            if new_symbol != st.session_state.symbol:
                st.session_state.symbol = new_symbol
                st.rerun()
            new_rfr = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.5, 0.1, key="opt_rfr") / 100
            st.session_state.risk_free_rate = new_rfr
            risk_free_rate = new_rfr

        with opt_col2:
            st.markdown("**Your Market View**")
            outlook = st.radio("Outlook", ["Bullish", "Bearish", "Neutral", "Volatile (big move, unsure direction)"], index=0, key="opt_outlook")
            outlook_map = {"Bullish": "bullish", "Bearish": "bearish", "Neutral": "neutral", "Volatile (big move, unsure direction)": "volatile"}
            outlook_key = outlook_map[outlook]
            st.session_state.outlook_key = outlook_key
            risk_tolerance = st.radio("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=0, key="opt_risk_tol")
            risk_tol_key = risk_tolerance.lower()
            st.session_state.risk_tol_key = risk_tol_key

        with opt_col3:
            st.markdown("**Position Sizing**")
            portfolio_value = st.number_input("Portfolio Value ($)", min_value=1000, max_value=10_000_000, value=25_000, step=1000, key="opt_portfolio_val")
            st.session_state.portfolio_value = portfolio_value
            risk_per_trade = st.slider("Max Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5, key="opt_risk_trade") / 100
            st.session_state.risk_per_trade = risk_per_trade

    symbol = st.session_state.symbol
    if not info or not expirations:
        st.warning(f"Enter a valid ticker symbol above to see options data. Current: **{symbol}**")

    # IV Percentile Banner
    if iv_data:
        iv_cols = st.columns(5)
        iv_labels = ["IV Percentile", "Current ATM IV", "30d Realized Vol", "60d Realized Vol", "90d Realized Vol"]
        iv_values = [f"{iv_percentile:.0f}%", f"{iv_data['current_iv']:.1%}", f"{iv_data['rv_30']:.1%}", f"{iv_data['rv_60']:.1%}", f"{iv_data['rv_90']:.1%}"]
        with iv_cols[0]:
            color = "inverse" if iv_data["iv_rank_label"] == "HIGH" else "off" if iv_data["iv_rank_label"] == "LOW" else "normal"
            st.metric(iv_labels[0], iv_values[0], delta=iv_data["iv_rank_label"], delta_color=color)
        for j in range(1, 5):
            with iv_cols[j]:
                st.metric(iv_labels[j], iv_values[j])

        if iv_data["iv_rank_label"] == "HIGH":
            st.info("**IV is elevated** — Favor strategies that **sell premium**: credit spreads, iron condors, covered calls.")
        elif iv_data["iv_rank_label"] == "LOW":
            st.info("**IV is depressed** — Favor strategies that **buy premium**: long calls/puts, straddles, debit spreads.")
        else:
            st.info("**IV is moderate** — Focus on your directional view.")

    # Entropy Signal
    if entropy_signal:
        e_cols = st.columns(4)
        regime_colors = {"CONCENTRATION": "inverse", "UNCERTAINTY": "off", "STABLE": "normal"}
        with e_cols[0]:
            st.metric("Market Regime", entropy_signal.regime, delta=f"{entropy_signal.entropy_change:+.3f} bits",
                      delta_color=regime_colors.get(entropy_signal.regime, "normal"))
        with e_cols[1]:
            st.metric("Entropy", f"{entropy_signal.current_entropy:.3f} bits", delta=f"Trend: {entropy_signal.entropy_trend}")
        with e_cols[2]:
            st.metric("HHI", f"{entropy_signal.current_hhi:.4f}", help="Herfindahl Index. Higher = more concentrated.")
        with e_cols[3]:
            st.metric("Strategy Bias", entropy_signal.strategy_bias.replace("_", " "), delta=entropy_signal.signal_strength, delta_color="normal")

    st.markdown("---")

    # Expiration Selector
    if expirations:
        selected_idx = st.selectbox("Expiration Date", range(len(expirations)),
                                     format_func=lambda i: exp_with_dte[i],
                                     index=min(2, len(expirations) - 1), key="options_exp_selector")
        selected_exp = expirations[selected_idx]
        selected_dte = (datetime.strptime(selected_exp, "%Y-%m-%d") - datetime.now()).days

        with st.spinner(f"Loading options chain for {selected_exp}..."):
            chain_df, _ = get_enriched_chain(symbol, selected_exp, risk_free_rate)

        st.markdown("---")

        options_subtab1, options_subtab2, options_subtab3 = st.tabs(["Strategy Recommender", "Options Chain", "Greeks & IV"])

        with options_subtab1:
            st.markdown("#### Recommended Strategies")
            regime_str = f" | Regime: **{entropy_signal.regime}**" if entropy_signal else ""
            st.caption(f"View: **{outlook}** | IV: **{iv_percentile:.0f}th %ile** | DTE: **{selected_dte}d** | Risk: **{risk_tolerance}** | Portfolio: **${portfolio_value:,.0f}**{regime_str}")

            entropy_fn = (lambda name, sig=entropy_signal: entropy_score_adjustment(name, sig)) if entropy_signal else None
            recommendations = recommend_strategies(
                outlook=outlook_key, iv_percentile=iv_percentile, dte=selected_dte,
                risk_tolerance=risk_tol_key, spot=spot, chain_df=chain_df, entropy_adjustment_fn=entropy_fn,
            )

            if not recommendations:
                st.warning("No strategies match your criteria. Try adjusting your outlook or risk tolerance.")
            else:
                for i, rec in enumerate(recommendations[:5]):
                    strat = rec.strategy
                    card = generate_trade_card(
                        strategy_name=strat.name, symbol=symbol, expiration=selected_exp,
                        dte=selected_dte, spot=spot, portfolio_value=portfolio_value,
                        risk_per_trade_pct=risk_per_trade, concrete_legs=rec.concrete_legs,
                        net_cost=rec.net_cost, max_loss_dollars=rec.max_loss_dollars,
                        max_gain_dollars=rec.max_gain_dollars, breakevens=rec.breakeven,
                    )
                    entropy_badge = " [ENTROPY FAVORED]" if rec.entropy_adjustment >= 5 else " [ENTROPY PENALIZED]" if rec.entropy_adjustment <= -5 else ""

                    with st.expander(
                        f"#{rec.rank}  {strat.name}{entropy_badge}  —  Score: {rec.score:.0f}/100  |  "
                        f"{'Credit' if card.is_credit else 'Debit'}: ${card.net_cost_per_share:.2f}  |  Size: {card.recommended_contracts} contracts",
                        expanded=(i == 0),
                    ):
                        st.markdown("### Trade Card")
                        m1, m2, m3, m4 = st.columns(4)
                        with m1: st.metric("Contracts", f"{card.recommended_contracts}")
                        with m2: st.metric("Max Loss (total)", f"${card.max_loss_total:,.0f}")
                        with m3: st.metric("Max Gain (total)", f"${card.max_gain_total:,.0f}" if card.max_gain_total is not None else "Unlimited")
                        with m4: st.metric("Risk/Reward", f"1:{card.risk_reward_ratio:.2f}" if card.risk_reward_ratio is not None else "Unlimited upside")

                        s1, s2, s3, s4 = st.columns(4)
                        with s1: st.metric("Capital Required", f"${card.total_capital_required:,.0f}")
                        with s2: st.metric("% of Portfolio", f"{card.total_capital_required / portfolio_value:.1%}" if portfolio_value > 0 else "-")
                        with s3: st.metric("Max Risk Budget", f"${card.max_risk_dollars:,.0f}")
                        with s4:
                            if len(card.breakevens) == 1:
                                st.metric("Break-even", f"${card.breakevens[0]:,.2f}")
                            else:
                                st.metric("Break-evens", " / ".join(f"${b:.2f}" for b in card.breakevens))

                        st.markdown("**Trade Construction:**")
                        st.dataframe(pd.DataFrame([{
                            "Action": "BUY" if leg["action"] == "buy" else "SELL",
                            "Type": leg["type"].upper(), "Strike": f"${leg['strike']:.0f}",
                            "Price": f"${leg['estimated_price']:.2f}", "Qty": card.recommended_contracts,
                        } for leg in rec.concrete_legs]), use_container_width=False, hide_index=True)

                        cost_label = "Net Credit" if card.is_credit else "Net Debit"
                        st.markdown(f"**{cost_label}: ${card.net_cost_per_share:.2f}/share "
                                    f"(${card.net_cost_per_contract:.2f}/contract x {card.recommended_contracts} = "
                                    f"${card.net_cost_per_contract * card.recommended_contracts:,.2f} total)**")

                        st.markdown("**Exit Rules:**")
                        ex1, ex2 = st.columns(2)
                        with ex1:
                            st.success(f"**Profit Target:** Close at {card.profit_target_pct:.0%} of max profit = ${card.profit_target_dollars:,.0f}/contract")
                        with ex2:
                            st.error(f"**Stop Loss:** Close at ${card.stop_loss_dollars:,.0f}/contract loss")

                        if card.kelly_fraction is not None:
                            if card.kelly_fraction > 0:
                                st.caption(f"Half-Kelly suggests {card.kelly_contracts} contracts ({card.kelly_fraction:.1%} of portfolio). Risk-rule sizing: {card.recommended_contracts} contracts.")
                            elif card.kelly_fraction < 0:
                                st.caption(f"Kelly criterion is negative ({card.kelly_fraction:.3f}) — risk/reward may not justify this trade.")

                        for warning in card.warnings:
                            st.warning(warning)

                        # Payoff Diagram
                        price_range, payoff, payoff_dollars = compute_strategy_payoff(rec.concrete_legs, rec.net_cost, spot)
                        payoff_total = payoff_dollars * max(card.recommended_contracts, 1)
                        contracts_label = max(card.recommended_contracts, 1)

                        fig, ax = dark_mpl_figure((9, 4))
                        ax.plot(price_range, payoff_total, color="#1f77b4", linewidth=2)
                        ax.axhline(0, color="white", linewidth=0.5)
                        ax.axvline(spot, color="gray", linestyle="--", alpha=0.7, label=f"Spot ${spot:.2f}")
                        for be in rec.breakeven:
                            ax.axvline(be, color="#00ff00", linestyle=":", alpha=0.7, label=f"B/E ${be:.2f}")
                        ax.fill_between(price_range, payoff_total, 0, where=(payoff_total > 0), color="green", alpha=0.2)
                        ax.fill_between(price_range, payoff_total, 0, where=(payoff_total < 0), color="red", alpha=0.2)
                        ax.set_xlabel("Stock Price at Expiration", color='white')
                        ax.set_ylabel("Profit / Loss ($)", color='white')
                        ax.set_title(f"{strat.name} — P&L at Expiration ({contracts_label} contract{'s' if contracts_label > 1 else ''})", color='white')
                        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))
                        ax.tick_params(colors='white')
                        ax.legend(fontsize=8, facecolor=CHART_FACE, labelcolor='white')
                        ax.grid(True, alpha=0.3)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        st.markdown("**Why this strategy works here:**")
                        st.markdown(f"> {strat.why_it_works}")
                        with st.expander("Scoring breakdown", expanded=False):
                            for reason in rec.explanation.split(" | "):
                                st.markdown(f"- {reason}")
                            st.markdown(f"- **Greeks profile:** {strat.greeks_profile}")
                            st.markdown(f"- **Risk type:** {strat.risk_type.title()} | **Complexity:** {'Simple' if strat.complexity == 1 else 'Intermediate' if strat.complexity == 2 else 'Advanced'}")

        # Options Chain Explorer
        with options_subtab2:
            st.markdown("#### Options Chain Explorer")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                moneyness_filter = st.multiselect("Moneyness", ["ITM", "ATM", "OTM"], default=["ITM", "ATM", "OTM"], key="chain_moneyness")
            with filter_col2:
                min_oi = st.number_input("Min Open Interest", value=0, min_value=0, step=10, key="chain_oi")
            with filter_col3:
                min_vol = st.number_input("Min Volume", value=0, min_value=0, step=10, key="chain_vol")

            filtered = chain_df[chain_df["moneyness_label"].isin(moneyness_filter)]
            if min_oi > 0: filtered = filtered[filtered["openInterest"].fillna(0) >= min_oi]
            if min_vol > 0: filtered = filtered[filtered["volume"].fillna(0) >= min_vol]

            display_cols = ["strike", "bid", "ask", "midPrice", "lastPrice", "bsPrice",
                            "impliedVolatility", "delta", "gamma", "theta", "vega", "openInterest", "volume", "moneyness_label"]
            col_names = ["Strike", "Bid", "Ask", "Mid", "Last", "BS Price", "IV", "Delta", "Gamma", "Theta", "Vega", "OI", "Vol", "Money"]

            def format_chain(df):
                d = df[display_cols].copy()
                d.columns = col_names
                for c in ["Bid", "Ask", "Mid", "Last", "BS Price"]:
                    d[c] = d[c].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
                d["IV"] = d["IV"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
                d["Delta"] = d["Delta"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
                d["Gamma"] = d["Gamma"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
                d["Theta"] = d["Theta"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
                d["Vega"] = d["Vega"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
                d["Strike"] = d["Strike"].apply(lambda x: f"${x:.2f}")
                d["OI"] = d["OI"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
                d["Vol"] = d["Vol"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
                return d

            chain_tab_calls, chain_tab_puts = st.tabs(["Calls", "Puts"])
            for tab, opt_type in [(chain_tab_calls, "call"), (chain_tab_puts, "put")]:
                with tab:
                    data = filtered[filtered["optionType"] == opt_type].sort_values("strike")
                    if data.empty:
                        st.warning(f"No {opt_type}s match your filters.")
                    else:
                        st.caption(f"{len(data)} contracts | Spot: ${spot:.2f} | DTE: {chain_df['dte'].iloc[0]}d")
                        st.dataframe(format_chain(data), use_container_width=True, hide_index=True, height=min(500, 35 * len(data) + 38))

        # Greeks & IV
        with options_subtab3:
            st.markdown("#### Greeks Across Strikes")
            liquid = chain_df[chain_df["openInterest"].fillna(0) > 10].copy()
            if liquid.empty: liquid = chain_df.copy()
            liquid = liquid[(liquid["moneyness"] >= 0.80) & (liquid["moneyness"] <= 1.20)]

            greek_type = st.selectbox("Select Greek", ["Delta", "Gamma", "Theta", "Vega"], index=0, key="greek_select")
            greek_col = greek_type.lower()

            fig, ax = dark_mpl_figure()
            calls_g = liquid[liquid["optionType"] == "call"].sort_values("strike")
            puts_g = liquid[liquid["optionType"] == "put"].sort_values("strike")
            if not calls_g.empty: ax.plot(calls_g["strike"], calls_g[greek_col], "b-o", markersize=3, label=f"Call {greek_type}", linewidth=1.5)
            if not puts_g.empty: ax.plot(puts_g["strike"], puts_g[greek_col], "r-o", markersize=3, label=f"Put {greek_type}", linewidth=1.5)
            ax.axvline(spot, color="gray", linestyle="--", alpha=0.7, label=f"Spot ${spot:.2f}")
            ax.set_xlabel("Strike Price", color='white')
            ax.set_ylabel(greek_type, color='white')
            ax.set_title(f"{symbol} — {greek_type} by Strike ({selected_exp})", color='white')
            ax.tick_params(colors='white')
            ax.legend(facecolor=CHART_FACE, labelcolor='white')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # IV Smile
            st.subheader("Volatility Smile")
            fig2, ax2 = dark_mpl_figure()
            if not calls_g.empty: ax2.plot(calls_g["strike"], calls_g["impliedVolatility"] * 100, "b-o", markersize=3, label="Call IV", linewidth=1.5)
            if not puts_g.empty: ax2.plot(puts_g["strike"], puts_g["impliedVolatility"] * 100, "r-o", markersize=3, label="Put IV", linewidth=1.5)
            ax2.axvline(spot, color="gray", linestyle="--", alpha=0.7, label=f"Spot ${spot:.2f}")
            ax2.set_xlabel("Strike Price", color='white')
            ax2.set_ylabel("Implied Volatility (%)", color='white')
            ax2.set_title(f"{symbol} — Volatility Smile ({selected_exp})", color='white')
            ax2.tick_params(colors='white')
            ax2.legend(facecolor=CHART_FACE, labelcolor='white')
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            # IV vs RV
            if iv_data:
                st.subheader("Implied vs. Realized Volatility")
                vol_df = pd.DataFrame({
                    "Metric": ["Current ATM IV", "30d RV", "60d RV", "90d RV"],
                    "Value": [iv_data["current_iv"], iv_data["rv_30"], iv_data["rv_60"], iv_data["rv_90"]],
                })
                fig3, ax3 = dark_mpl_figure((8, 4))
                colors = ["#ff7f0e", "#1f77b4", "#1f77b4", "#1f77b4"]
                bars = ax3.barh(vol_df["Metric"], vol_df["Value"] * 100, color=colors)
                ax3.set_xlabel("Volatility (%)", color='white')
                ax3.set_title(f"{symbol} — IV vs. Realized Volatility", color='white')
                ax3.tick_params(colors='white')
                for bar, val in zip(bars, vol_df["Value"]):
                    ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", fontsize=10, color='white')
                ax3.grid(True, alpha=0.3, axis="x")
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)

                premium_ratio = iv_data["current_iv"] / iv_data["rv_30"] if iv_data["rv_30"] > 0 else 1
                if premium_ratio > 1.3:
                    st.success(f"**IV/RV Ratio: {premium_ratio:.2f}x** — Options priced {(premium_ratio - 1) * 100:.0f}% above realized. Edge in selling premium.")
                elif premium_ratio < 0.8:
                    st.success(f"**IV/RV Ratio: {premium_ratio:.2f}x** — Options cheap vs. actual movement. Edge in buying premium.")
                else:
                    st.info(f"**IV/RV Ratio: {premium_ratio:.2f}x** — IV and RV roughly in line.")


# ═══════════════════════════════════════
# TAB: Entropy Analysis
# ═══════════════════════════════════════

with tab_entropy:
    st.subheader("Market Entropy Analysis")
    st.caption("Information-theoretic analysis of market structure. Based on the sample space expansion framework from Case 2 (ChatGPT Launch).")

    if entropy_signal is None:
        st.warning("Could not compute market entropy. Check internet connection.")
    else:
        regime_messages = {
            "CONCENTRATION": ("error", f"**CONCENTRATION REGIME** — Entropy decreased {abs(entropy_signal.entropy_change):.3f} bits ({abs(entropy_signal.entropy_change_pct):.1f}%). Fewer sectors driving returns. Favor selling premium."),
            "UNCERTAINTY": ("warning", f"**UNCERTAINTY REGIME** — Entropy increased {abs(entropy_signal.entropy_change):.3f} bits ({abs(entropy_signal.entropy_change_pct):.1f}%). More sectors participating. Favor buying premium."),
        }
        method, msg = regime_messages.get(entropy_signal.regime, ("info", f"**STABLE REGIME** — Entropy near baseline (change: {entropy_signal.entropy_change:+.3f} bits). Use standard IV-based selection."))
        getattr(st, method)(msg)

        st.markdown("### Entropy Metrics")
        me1, me2, me3, me4 = st.columns(4)
        with me1: st.metric("Current Entropy", f"{entropy_signal.current_entropy:.3f} bits")
        with me2: st.metric("Baseline (Nov 2022)", f"{entropy_signal.baseline_entropy:.3f} bits")
        with me3: st.metric("Normalized", f"{entropy_signal.current_normalized:.3f}", help="0 = one sector dominates, 1 = perfectly equal")
        with me4: st.metric("CR7 (Top 7)", f"{entropy_signal.current_cr7:.1%}", help="Concentration ratio: weight of top 7 sectors")

        st.markdown("### Current Sector Weights")
        sector_df = pd.DataFrame(list(entropy_signal.sector_weights.items()), columns=["Sector", "Weight"]).sort_values("Weight", ascending=True)

        fig4, ax4 = dark_mpl_figure((8, 5))
        colors_s = ["#ff7f0e" if w > 0.12 else "#1f77b4" for w in sector_df["Weight"]]
        bars = ax4.barh(sector_df["Sector"], sector_df["Weight"] * 100, color=colors_s)
        ax4.set_xlabel("Weight (%)", color='white')
        ax4.set_title("S&P 500 Sector Weights (Estimated from ETF Performance)", color='white')
        ax4.tick_params(colors='white')
        for bar, val in zip(bars, sector_df["Weight"]):
            ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", fontsize=9, color='white')
        ax4.grid(True, alpha=0.3, axis="x")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        st.markdown("### Analysis")
        st.markdown(entropy_signal.explanation)

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


# ─── Footer ───

st.markdown("---")
st.caption(
    "OptiMax v2.0 | Data: yfinance (15-min delayed) | "
    "Greeks: Black-Scholes (European approx.) | "
    "Entropy: Sector ETF-based Shannon entropy | "
    "Portfolio: Live prices & dividend yields | "
    "Not financial advice. For educational purposes only."
)
