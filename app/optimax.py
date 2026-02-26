"""
RCR Portfolio Trackers — Options Strategy Recommender + Portfolio Manager
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
    get_stock_performance,
    get_portfolio_performance,
    create_watchlist_item,
    refresh_watchlist_item,
    export_holdings_to_csv,
    import_holdings_from_csv,
    create_holdings_batch,
    export_watchlist_to_csv,
    get_dividend_payment_history,
    get_monthly_dividend_totals,
    calculate_drip_projection,
    calculate_drip_vs_no_drip,
    get_stock_news,
    get_portfolio_news,
    POPULAR_STOCKS,
    POPULAR_ETFS,
    POPULAR_BOND_ETFS,
    POPULAR_REITS,
    TICKER_DATABASE,
    FedEvent,
    WatchlistItem,
    NewsItem,
)
import io
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from pathlib import Path
from fallback_cache import save_fallback, load_fallback

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# ─────────────────────────────────────────────
# Cached Entropy (avoid rate limits on yfinance)
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_entropy():
    """Cache entropy computation for 1 hour. Falls back to disk cache on failure."""
    try:
        result = compute_market_entropy(lookback_days=90)
        if result:
            save_fallback("entropy", result)
        return result
    except Exception:
        data, _ = load_fallback("entropy", max_age_hours=24)
        return data


@st.cache_data(ttl=300, show_spinner=False)
def _cached_ticker_info(sym):
    """Cache ticker info for 5 min. Falls back to disk cache on failure."""
    try:
        result = get_ticker_info(sym)
        if result and result.get("spot"):
            save_fallback(f"ticker_{sym}", result)
        return result
    except Exception:
        data, _ = load_fallback(f"ticker_{sym}", max_age_hours=4)
        return data


@st.cache_data(ttl=300, show_spinner=False)
def _cached_expirations(sym):
    """Cache expirations for 5 min."""
    try:
        result = get_expirations(sym)
        if result:
            save_fallback(f"expirations_{sym}", result)
        return result
    except Exception:
        data, _ = load_fallback(f"expirations_{sym}", max_age_hours=4)
        return data or []


@st.cache_data(ttl=300, show_spinner=False)
def _cached_iv_percentile(sym):
    """Cache IV percentile for 5 min. Falls back to disk cache on failure."""
    try:
        result = compute_iv_percentile(sym)
        if result:
            save_fallback(f"iv_{sym}", result)
        return result
    except Exception:
        data, _ = load_fallback(f"iv_{sym}", max_age_hours=4)
        return data


@st.cache_data(ttl=120, show_spinner=False)
def _cached_enriched_chain(sym, expiration, rfr, _spot):
    """Cache enriched chain for 2 min."""
    return get_enriched_chain(sym, expiration, rfr, spot=_spot)


# ─────────────────────────────────────────────
# Delta-Based Probability of Profit
# ─────────────────────────────────────────────

def _compute_pop_from_delta(strategy_name: str, concrete_legs: list, chain_df) -> float:
    """
    Compute probability of profit from option delta in the chain.

    For credit strategies: POP = 1 - |short_leg_delta|
    For debit strategies: POP = |long_leg_delta|
    For long options: POP = |delta|
    Fallback: 0.50 if chain data unavailable.
    """
    if chain_df is None or chain_df.empty or not concrete_legs:
        return 0.50

    def _get_delta(strike, option_type):
        """Look up delta for a strike/type in the chain."""
        subset = chain_df[chain_df["optionType"] == option_type]
        if subset.empty:
            return None
        idx = (subset["strike"] - strike).abs().idxmin()
        return float(subset.loc[idx, "delta"])

    credit_strategies = {"Bull Put Spread", "Bear Call Spread", "Iron Condor",
                         "Cash-Secured Put", "Covered Call"}
    debit_strategies = {"Bull Call Spread", "Bear Put Spread"}
    long_strategies = {"Long Call", "Long Put"}
    vol_strategies = {"Long Straddle", "Long Strangle"}

    try:
        if strategy_name in credit_strategies:
            # Find the short leg and use 1 - |delta|
            for leg in concrete_legs:
                if leg["action"] == "sell":
                    d = _get_delta(leg["strike"], leg["type"])
                    if d is not None:
                        return max(0.05, min(0.95, 1.0 - abs(d)))
        elif strategy_name in debit_strategies:
            # Find the long leg and use |delta|
            for leg in concrete_legs:
                if leg["action"] == "buy":
                    d = _get_delta(leg["strike"], leg["type"])
                    if d is not None:
                        return max(0.05, min(0.95, abs(d)))
        elif strategy_name in long_strategies:
            # Single long option: POP = |delta|
            leg = concrete_legs[0]
            d = _get_delta(leg["strike"], leg["type"])
            if d is not None:
                return max(0.05, min(0.95, abs(d)))
        elif strategy_name in vol_strategies:
            # Straddle/strangle: approximate POP as probability of big move
            call_leg = next((l for l in concrete_legs if l["type"] == "call"), None)
            put_leg = next((l for l in concrete_legs if l["type"] == "put"), None)
            if call_leg and put_leg:
                cd = _get_delta(call_leg["strike"], "call")
                pd_ = _get_delta(put_leg["strike"], "put")
                if cd is not None and pd_ is not None:
                    # POP of straddle ≈ prob of finishing outside breakevens
                    pop = abs(cd) + abs(pd_) - 1.0
                    return max(0.05, min(0.95, abs(pop)))
    except Exception:
        pass

    return 0.50  # fallback


# ─────────────────────────────────────────────
# User Data Persistence
# ─────────────────────────────────────────────
USER_DATA_DIR = os.path.join(os.path.dirname(__file__), "user_data")

def ensure_user_data_dir():
    """Create user data directory if it doesn't exist."""
    if not os.path.exists(USER_DATA_DIR):
        os.makedirs(USER_DATA_DIR)

def get_user_data_path(username: str) -> str:
    """Get path to user's data file."""
    ensure_user_data_dir()
    safe_username = "".join(c for c in username if c.isalnum() or c in "-_").lower()
    return os.path.join(USER_DATA_DIR, f"{safe_username}.json")

def save_user_data(username: str, holdings: list, cash_balance: float = 0, watchlist: list = None):
    """Save user's portfolio data to JSON file."""
    if not username:
        return

    holdings_data = []
    for h in holdings:
        holdings_data.append({
            "symbol": h.symbol,
            "shares": h.shares,
            "avg_cost": h.avg_cost,
        })

    watchlist_data = []
    if watchlist:
        for w in watchlist:
            watchlist_data.append({
                "symbol": w.symbol,
                "added_date": w.added_date.isoformat() if hasattr(w.added_date, 'isoformat') else str(w.added_date),
            })

    data = {
        "holdings": holdings_data,
        "cash_balance": cash_balance,
        "watchlist": watchlist_data,
        "last_updated": datetime.now().isoformat(),
    }

    filepath = get_user_data_path(username)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_user_data(username: str) -> dict:
    """Load user's portfolio data from JSON file."""
    if not username:
        return None

    filepath = get_user_data_path(username)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

# Set dark theme for matplotlib charts
plt.style.use('dark_background')
CHART_BG_COLOR = '#0E1117'
CHART_FACE_COLOR = '#1A1D26'
from dateutil.relativedelta import relativedelta
import calendar as cal_module


def get_company_color(symbol: str) -> str:
    """Get a consistent color for a company symbol."""
    import colorsys
    # Known company brand colors
    brand_colors = {
        "AAPL": "#A2AAAD",
        "MSFT": "#00A4EF",
        "GOOGL": "#4285F4",
        "GOOG": "#4285F4",
        "AMZN": "#FF9900",
        "META": "#0668E1",
        "NVDA": "#76B900",
        "TSLA": "#CC0000",
        "JPM": "#003087",
        "V": "#1A1F71",
        "MA": "#EB001B",
        "SPY": "#1E88E5",
        "QQQ": "#76B900",
        "SCHD": "#5A2D82",
        "O": "#003399",
        "VTI": "#C70000",
        "VOO": "#C70000",
    }

    if symbol in brand_colors:
        return brand_colors[symbol]

    # Generate consistent hex color based on symbol hash (matplotlib compatible)
    hash_val = sum(ord(c) for c in symbol)
    hue = ((hash_val * 137) % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.65)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def get_sparkline_data(symbol: str, days: int = 7) -> list:
    """Get price history for sparkline chart."""
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days}d")
        if not hist.empty:
            return hist['Close'].tolist()
    except:
        pass
    return []


def create_sparkline_svg(prices: list, color: str = "#00ff00", width: int = 80, height: int = 25) -> str:
    """Create an SVG sparkline chart."""
    if not prices or len(prices) < 2:
        return ""

    # Normalize prices to fit in the SVG
    min_p, max_p = min(prices), max(prices)
    range_p = max_p - min_p if max_p != min_p else 1

    # Create points for the polyline
    points = []
    for i, p in enumerate(prices):
        x = (i / (len(prices) - 1)) * width
        y = height - ((p - min_p) / range_p) * height
        points.append(f"{x:.1f},{y:.1f}")

    # Determine color based on trend
    trend_color = "#00C853" if prices[-1] >= prices[0] else "#FF5252"

    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <polyline points="{' '.join(points)}" fill="none" stroke="{trend_color}" stroke-width="2"/>
    </svg>'''
    return svg


def show_dollar_spinner(message: str = "Loading...") -> str:
    """Return HTML for a golden spinning dollar sign animation."""
    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px;">
        <style>
            @keyframes spin {{
                0% {{ transform: rotateY(0deg); }}
                100% {{ transform: rotateY(360deg); }}
            }}
            @keyframes glow {{
                0%, 100% {{ text-shadow: 0 0 20px #FFD700, 0 0 40px #FFA500, 0 0 60px #FF8C00; }}
                50% {{ text-shadow: 0 0 30px #FFD700, 0 0 60px #FFA500, 0 0 90px #FF8C00; }}
            }}
            .dollar-spin {{
                font-size: 80px;
                color: #FFD700;
                animation: spin 1s linear infinite, glow 2s ease-in-out infinite;
                display: inline-block;
                font-weight: bold;
                font-family: 'Arial Black', sans-serif;
            }}
            .loading-text {{
                color: #FFD700;
                font-size: 18px;
                margin-top: 20px;
                animation: glow 2s ease-in-out infinite;
            }}
        </style>
        <div class="dollar-spin">$</div>
        <div class="loading-text">{message}</div>
    </div>
    """


def show_skeleton(num_cards: int = 3, height: int = 80):
    """Show animated shimmer loading skeleton placeholders."""
    st.markdown("""
    <style>
    @keyframes shimmer {
        0% { background-position: -400px 0; }
        100% { background-position: 400px 0; }
    }
    .skeleton-card {
        background: linear-gradient(90deg, #1A1D26 25%, #2A2D36 50%, #1A1D26 75%);
        background-size: 800px 100%;
        animation: shimmer 1.5s ease-in-out infinite;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    for _ in range(num_cards):
        st.markdown(f"<div class='skeleton-card' style='height:{height}px;'></div>",
                    unsafe_allow_html=True)


def show_styled_empty_state(icon: str, title: str, message: str, steps: list = None):
    """Show a styled empty state card with optional steps."""
    steps_html = ""
    if steps:
        steps_html = "<ol style='text-align:left;margin-top:12px;color:#B8B5FF;'>"
        for step in steps:
            steps_html += f"<li style='margin:6px 0;color:#ccc;'>{step}</li>"
        steps_html += "</ol>"
    st.markdown(f"""
    <div style="text-align:center;padding:40px 20px;background:#1A1D26;border-radius:12px;
                border:1px solid #2A2D36;margin:20px 0;">
        <div style="font-size:48px;margin-bottom:12px;">{icon}</div>
        <h3 style="color:#B8B5FF;margin-bottom:8px;">{title}</h3>
        <p style="color:#888;font-size:1.05em;">{message}</p>
        {steps_html}
    </div>
    """, unsafe_allow_html=True)


def get_tax_loss_opportunities(holdings: list, tax_rate: float = 0.25) -> list:
    """Identify tax-loss harvesting opportunities."""
    opportunities = []

    # Get total gains to offset
    total_gains = sum(h.unrealized_pnl for h in holdings if h.unrealized_pnl > 0)

    for h in holdings:
        if h.unrealized_pnl < 0:  # Loss position
            loss_amount = abs(h.unrealized_pnl)
            tax_savings = loss_amount * tax_rate

            # Find similar replacement stocks (same sector)
            replacements = []
            if h.sector:
                sector_stocks = {
                    "Technology": ["MSFT", "GOOGL", "CRM", "ADBE", "ORCL"],
                    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "SBUX"],
                    "Financial Services": ["JPM", "BAC", "GS", "MS", "V"],
                    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
                    "Communication Services": ["META", "GOOG", "DIS", "NFLX", "T"],
                    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST"],
                    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
                    "Industrials": ["CAT", "BA", "UNP", "HON", "UPS"],
                    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG"],
                    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
                    "Basic Materials": ["LIN", "APD", "ECL", "SHW", "DD"],
                }
                sector_list = sector_stocks.get(h.sector, [])
                replacements = [s for s in sector_list if s != h.symbol][:3]

            opportunities.append({
                "symbol": h.symbol,
                "name": h.name,
                "shares": h.shares,
                "loss": h.unrealized_pnl,
                "loss_pct": h.unrealized_pnl_pct,
                "tax_savings": tax_savings,
                "current_price": h.current_price,
                "avg_cost": h.avg_cost,
                "sector": h.sector,
                "replacements": replacements,
            })

    # Sort by largest loss (most tax savings)
    opportunities.sort(key=lambda x: x["loss"])
    return opportunities


def get_similar_stocks_recommendations(holdings: list) -> list:
    """Get stock recommendations based on current holdings."""
    recommendations = []
    current_symbols = set(h.symbol for h in holdings)
    current_sectors = set(h.sector for h in holdings if h.sector)

    # Stock clusters (stocks often held together)
    stock_clusters = {
        "AAPL": ["MSFT", "GOOGL", "AMZN", "META", "NVDA"],
        "MSFT": ["AAPL", "GOOGL", "AMZN", "CRM", "ADBE"],
        "GOOGL": ["META", "MSFT", "AMZN", "NFLX", "AAPL"],
        "AMZN": ["MSFT", "GOOGL", "AAPL", "SHOP", "COST"],
        "NVDA": ["AMD", "AAPL", "MSFT", "TSM", "AVGO"],
        "TSLA": ["RIVN", "NIO", "LCID", "F", "GM"],
        "JPM": ["BAC", "GS", "MS", "WFC", "C"],
        "V": ["MA", "PYPL", "SQ", "AXP", "COF"],
        "JNJ": ["PFE", "UNH", "ABBV", "MRK", "LLY"],
        "SPY": ["QQQ", "VTI", "VOO", "IWM", "DIA"],
        "QQQ": ["SPY", "VGT", "XLK", "ARKK", "SMH"],
        "VTI": ["VOO", "SPY", "VXUS", "BND", "VEA"],
        "SCHD": ["VYM", "VIG", "DGRO", "HDV", "NOBL"],
    }

    # Popular additions by sector
    sector_additions = {
        "Technology": [
            {"symbol": "NVDA", "reason": "AI leader, GPU dominance"},
            {"symbol": "MSFT", "reason": "Cloud + AI integration"},
            {"symbol": "AVGO", "reason": "Semiconductor diversification"},
        ],
        "Consumer Cyclical": [
            {"symbol": "COST", "reason": "Defensive retail, membership model"},
            {"symbol": "HD", "reason": "Housing market exposure"},
        ],
        "Financial Services": [
            {"symbol": "V", "reason": "Payment network, recession-resistant"},
            {"symbol": "BRK-B", "reason": "Diversified value play"},
        ],
        "Healthcare": [
            {"symbol": "UNH", "reason": "Insurance + services combo"},
            {"symbol": "LLY", "reason": "Weight loss drug momentum"},
        ],
    }

    # Find recommendations based on holdings
    for h in holdings:
        if h.symbol in stock_clusters:
            for rec in stock_clusters[h.symbol]:
                if rec not in current_symbols:
                    # Check if already recommended
                    if not any(r["symbol"] == rec for r in recommendations):
                        recommendations.append({
                            "symbol": rec,
                            "reason": f"Often held with {h.symbol}",
                            "based_on": h.symbol,
                            "type": "cluster"
                        })

    # Add sector-based recommendations
    for sector in current_sectors:
        if sector in sector_additions:
            for rec in sector_additions[sector]:
                if rec["symbol"] not in current_symbols:
                    if not any(r["symbol"] == rec["symbol"] for r in recommendations):
                        recommendations.append({
                            "symbol": rec["symbol"],
                            "reason": rec["reason"],
                            "based_on": sector,
                            "type": "sector"
                        })

    # Diversification suggestions (sectors not represented)
    missing_sectors = {
        "Technology": {"symbol": "VGT", "reason": "Tech sector ETF for diversification"},
        "Healthcare": {"symbol": "XLV", "reason": "Healthcare exposure"},
        "Financial Services": {"symbol": "XLF", "reason": "Financial sector exposure"},
        "Energy": {"symbol": "XLE", "reason": "Energy sector exposure"},
        "Real Estate": {"symbol": "VNQ", "reason": "Real estate exposure, dividend income"},
    }

    for sector, rec in missing_sectors.items():
        if sector not in current_sectors and rec["symbol"] not in current_symbols:
            if not any(r["symbol"] == rec["symbol"] for r in recommendations):
                recommendations.append({
                    "symbol": rec["symbol"],
                    "reason": rec["reason"],
                    "based_on": f"Missing {sector}",
                    "type": "diversify"
                })

    return recommendations[:10]  # Limit to top 10


def generate_portfolio_snapshot(holdings: list, summary: dict, username: str = "Investor") -> str:
    """Generate a shareable text snapshot of portfolio."""
    snapshot = f"""
📊 {username}'s Portfolio Snapshot
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

💰 SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━
Total Value:    ${summary.get('total_value', 0):,.2f}
Total P&L:      ${summary.get('total_pnl', 0):+,.2f} ({summary.get('total_pnl_pct', 0):+.2f}%)
Holdings:       {len(holdings)}
Dividend Yield: {summary.get('dividend_yield', 0):.2f}%

📈 TOP HOLDINGS
━━━━━━━━━━━━━━━━━━━━━━━━"""

    # Sort by value
    sorted_holdings = sorted(holdings, key=lambda h: h.market_value, reverse=True)

    for h in sorted_holdings[:5]:
        pnl_emoji = "🟢" if h.unrealized_pnl >= 0 else "🔴"
        snapshot += f"\n{pnl_emoji} {h.symbol}: ${h.market_value:,.0f} ({h.unrealized_pnl_pct:+.1f}%)"

    if len(holdings) > 5:
        snapshot += f"\n   ...and {len(holdings) - 5} more"

    snapshot += """

━━━━━━━━━━━━━━━━━━━━━━━━
Powered by RCR Portfolio Trackers 📱
"""
    return snapshot


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="RCR Portfolio Trackers",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS for Loading Animation & Styling
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* Loading animation */
@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

.loading-pulse {
    animation: pulse 1.5s ease-in-out infinite;
}

/* Custom spinner */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.custom-spinner {
    border: 4px solid #3a3a3a;
    border-top: 4px solid #4da6ff;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

/* Professional card styling */
.metric-card {
    background: linear-gradient(135deg, #2E2E2E 0%, #3A3A3A 100%);
    border-radius: 10px;
    padding: 20px;
    border-left: 4px solid #4da6ff;
    margin: 10px 0;
}

/* Health score colors */
.health-excellent { color: #00C853; }
.health-good { color: #4da6ff; }
.health-fair { color: #FFB300; }
.health-poor { color: #FF5252; }

/* Insight cards */
.insight-card {
    background: #2E2E2E;
    border-radius: 8px;
    padding: 15px;
    margin: 8px 0;
    border-left: 3px solid;
}
.insight-positive { border-color: #00C853; }
.insight-warning { border-color: #FFB300; }
.insight-negative { border-color: #FF5252; }
.insight-info { border-color: #4da6ff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Simple Authentication
# ─────────────────────────────────────────────

def check_authentication():
    """Simple authentication system."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    return st.session_state.authenticated

def login_form():
    """Display login form in sidebar."""
    # Initialize authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

    if not st.session_state.authenticated:
        with st.sidebar.expander("🔐 Login / Sign Up", expanded=False):
            auth_tab1, auth_tab2 = st.tabs(["Login", "Sign Up"])

            with auth_tab1:
                login_user = st.text_input("Username", key="login_user")
                login_pass = st.text_input("Password", type="password", key="login_pass")
                if st.button("Login", key="login_btn"):
                    # Simple demo authentication (in production, use proper auth)
                    if login_user and login_pass:
                        st.session_state.authenticated = True
                        st.session_state.username = login_user
                        # Load saved user data
                        saved_data = load_user_data(login_user)
                        if saved_data:
                            st.session_state.holdings = []
                            for h_data in saved_data.get("holdings", []):
                                try:
                                    holding = create_holding(h_data["symbol"], h_data["shares"], h_data["avg_cost"])
                                    if holding:
                                        st.session_state.holdings.append(holding)
                                except:
                                    pass
                            st.session_state.cash_balance = saved_data.get("cash_balance", 0)
                            st.success(f"Welcome back, {login_user}! Portfolio loaded.")
                        else:
                            st.success(f"Welcome, {login_user}!")
                        st.rerun()
                    else:
                        st.error("Please enter username and password")

            with auth_tab2:
                new_user = st.text_input("Choose Username", key="new_user")
                new_pass = st.text_input("Choose Password", type="password", key="new_pass")
                new_pass2 = st.text_input("Confirm Password", type="password", key="new_pass2")
                if st.button("Sign Up", key="signup_btn"):
                    if new_user and new_pass and new_pass == new_pass2:
                        st.session_state.authenticated = True
                        st.session_state.username = new_user
                        # Check if user already has saved data
                        saved_data = load_user_data(new_user)
                        if saved_data:
                            st.session_state.holdings = []
                            for h_data in saved_data.get("holdings", []):
                                try:
                                    holding = create_holding(h_data["symbol"], h_data["shares"], h_data["avg_cost"])
                                    if holding:
                                        st.session_state.holdings.append(holding)
                                except:
                                    pass
                            st.session_state.cash_balance = saved_data.get("cash_balance", 0)
                            st.success(f"Welcome back, {new_user}! Portfolio loaded.")
                        else:
                            st.success(f"Account created! Welcome, {new_user}!")
                        st.rerun()
                    elif new_pass != new_pass2:
                        st.error("Passwords don't match")
                    else:
                        st.error("Please fill all fields")

            st.caption("Demo mode: any credentials work")
    else:
        st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
        if st.sidebar.button("Logout", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

# Run authentication
login_form()

# ─────────────────────────────────────────────
# Session state defaults for options inputs
# ─────────────────────────────────────────────

if "symbol" not in st.session_state:
    st.session_state.symbol = "AAPL"
if "outlook_key" not in st.session_state:
    st.session_state.outlook_key = "bullish"
if "risk_tol_key" not in st.session_state:
    st.session_state.risk_tol_key = "conservative"
if "portfolio_value" not in st.session_state:
    st.session_state.portfolio_value = 25000
if "risk_per_trade" not in st.session_state:
    st.session_state.risk_per_trade = 0.02
if "risk_free_rate" not in st.session_state:
    st.session_state.risk_free_rate = 0.045

symbol = st.session_state.symbol
outlook_key = st.session_state.outlook_key
risk_tol_key = st.session_state.risk_tol_key
portfolio_value = st.session_state.portfolio_value
risk_per_trade = st.session_state.risk_per_trade
risk_free_rate = st.session_state.risk_free_rate

# ─────────────────────────────────────────────
# Sidebar — AI Investment Advisor Chat
# ─────────────────────────────────────────────

st.sidebar.markdown(
    "<h2 style='text-align:center;margin-bottom:0;font-family:\"Georgia\",\"Times New Roman\",serif;"
    "font-style:italic;color:#B8B5FF;'>RCR Portfolio</h2>"
    "<p style='text-align:center;color:#888;font-size:0.85em;margin-top:2px;'>AI Investment Advisor</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

if "sidebar_chat_messages" not in st.session_state:
    st.session_state.sidebar_chat_messages = []

chat_container = st.sidebar.container(height=400)
with chat_container:
    if not st.session_state.sidebar_chat_messages:
        st.markdown(
            '<div style="text-align:center;padding:25px 15px;color:#888;">'
            '<h4 style="color:#B8B5FF;margin-bottom:6px;font-weight:600;">Welcome to RCR AI</h4>'
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
            if msg["role"] == "user":
                st.markdown(f'<div style="font-size:0.7em;color:#666;margin-bottom:2px;padding-left:4px;font-weight:600;text-transform:uppercase;">You</div><div style="background:linear-gradient(135deg,#6C63FF 0%,#5A4FE0 100%);color:white;padding:10px 14px;border-radius:16px 16px 4px 16px;margin:6px 0 6px 15%;font-size:0.88em;line-height:1.5;">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-size:0.7em;color:#666;margin-bottom:2px;padding-left:4px;font-weight:600;text-transform:uppercase;">Advisor</div><div style="background-color:#1A1D26;color:#E8E8EC;padding:10px 14px;border-radius:16px 16px 16px 4px;margin:6px 15% 6px 0;font-size:0.88em;line-height:1.5;border:1px solid rgba(108,99,255,0.15);">{msg["content"]}</div>', unsafe_allow_html=True)

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
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except (KeyError, FileNotFoundError, Exception):
            groq_key = ""
    if not groq_key or groq_key == "your-api-key-here":
        st.sidebar.warning(
            "AI Advisor requires a Groq API key.\n\n"
            "**Streamlit Cloud:** Go to your app Settings → Secrets and add:\n"
            "```\nGROQ_API_KEY = \"your-key-here\"\n```\n"
            "**Local:** Add it to `app/.env`"
        )
    else:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            context = (f"The user is analyzing {symbol}. Outlook: {outlook_key}, "
                       f"risk tolerance: {risk_tol_key}, portfolio: ${portfolio_value:,}, "
                       f"max risk/trade: {risk_per_trade*100:.1f}%.")
            system_prompt = (
                f"You are an expert AI investment advisor in RCR Portfolio Trackers. "
                f"Help with investment strategies, stock analysis, options, portfolio diversification. "
                f"Be concise (sidebar). Use bullet points. Keep under 200 words. "
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

# ─────────────────────────────────────────────
# Load Data (lazy — don't block the whole app)
# ─────────────────────────────────────────────

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
            info = _cached_ticker_info(symbol)
            expirations = _cached_expirations(symbol)
    except Exception:
        info = None
        expirations = []

    st.markdown(
        "<h1 style='font-family:\"Georgia\",\"Times New Roman\",serif;font-style:italic;"
        "color:#B8B5FF;margin-bottom:0;'>RCR Portfolio</h1>",
        unsafe_allow_html=True,
    )

    if info:
        spot = info["spot"]

        # IV and Entropy are loaded lazily when the Options/Entropy tab renders
        # to minimize upfront API calls and avoid rate limits

        for exp in expirations:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
            exp_with_dte.append(f"{exp}  ({dte}d)")

# ─────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────

tab_dashboard, tab_portfolio, tab_valuation, tab_cash, tab_calendar, tab_options, tab_entropy = st.tabs([
    "📊 Dashboard",
    "💼 Portfolio Manager",
    "📋 Valuation & Analysis",
    "💵 Cash & Money Market",
    "📅 Calendar & News",
    "📈 Options Trading",
    "🔬 Entropy Analysis",
])

# ═════════════════════════════════════════════
# TAB: Dashboard
# ═════════════════════════════════════════════

with tab_dashboard:
    st.subheader("Dashboard")

    # Initialize holdings if needed — load demo portfolio on first visit
    if "holdings" not in st.session_state:
        st.session_state.holdings = []
    if "cash_balance" not in st.session_state:
        st.session_state.cash_balance = 0.0

    if not st.session_state.holdings and not st.session_state.get("_demo_dismissed"):
        # Load a demo portfolio so new users see a populated dashboard
        if not st.session_state.get("_demo_loaded"):
            DEMO_PORTFOLIO = [
                {"symbol": "AAPL", "shares": 15, "cost": 175.00},
                {"symbol": "MSFT", "shares": 10, "cost": 380.00},
                {"symbol": "GOOGL", "shares": 8, "cost": 140.00},
                {"symbol": "AMZN", "shares": 12, "cost": 178.00},
                {"symbol": "VOO", "shares": 20, "cost": 440.00},
                {"symbol": "SCHD", "shares": 30, "cost": 76.00},
                {"symbol": "O", "shares": 25, "cost": 58.00},
                {"symbol": "JPM", "shares": 10, "cost": 195.00},
            ]
            with st.spinner("Loading demo portfolio..."):
                demo_holdings = create_holdings_batch(DEMO_PORTFOLIO)
                if demo_holdings:
                    st.session_state.holdings = demo_holdings
                    st.session_state._demo_loaded = True
                    st.rerun()
        if not st.session_state.holdings:
            show_styled_empty_state(
                icon="📊",
                title="Welcome to RCR Portfolio Trackers",
                message="Your dashboard will come alive once you add holdings.",
                steps=[
                    "Go to the <b>Portfolio Manager</b> tab",
                    "Upload a CSV or add holdings manually",
                    "Return here to see analytics, dividends & performance",
                ],
            )
            show_skeleton(num_cards=3, height=60)
    else:
        # Show demo banner if viewing sample data
        if st.session_state.get("_demo_loaded") and not st.session_state.get("_demo_dismissed"):
            demo_col1, demo_col2 = st.columns([5, 1])
            with demo_col1:
                st.info("You're viewing a **demo portfolio**. Go to Portfolio Manager to import your own holdings.")
            with demo_col2:
                if st.button("Clear Demo", key="dismiss_demo"):
                    st.session_state.holdings = []
                    st.session_state._demo_dismissed = True
                    st.session_state._demo_loaded = False
                    st.rerun()

        # Calculate portfolio metrics
        with st.spinner("Loading dashboard..."):
            summary = calculate_portfolio_summary(st.session_state.holdings)
            total_invested = summary["total_value"]
            total_with_cash = total_invested + st.session_state.cash_balance

        # ── Top Metrics Row ──
        st.markdown("### Portfolio Overview")
        metric_cols = st.columns(6)
        with metric_cols[0]:
            st.metric("Total Value", f"${total_with_cash:,.2f}",
                     delta=f"{summary['total_pnl_pct']:+.2f}%")
        with metric_cols[1]:
            st.metric("Invested", f"${total_invested:,.2f}")
        with metric_cols[2]:
            st.metric("Cash", f"${st.session_state.cash_balance:,.2f}")
        with metric_cols[3]:
            st.metric("Annual Income", f"${summary['annual_income']:,.2f}")
        with metric_cols[4]:
            st.metric("Yield", f"{summary['portfolio_yield']:.2f}%")
        with metric_cols[5]:
            st.metric("Holdings", f"{len(st.session_state.holdings)}")

        # ── Dashboard Sub-tabs ──
        dash_tab1, dash_tab2, dash_tab3, dash_tab4, dash_tab5 = st.tabs([
            "📈 Performance",
            "⚠️ Risk Analytics",
            "⚖️ Rebalancing",
            "💰 Dividends",
            "🤖 AI Insights"
        ])

        # ══════════════════════════════════════════════
        # PERFORMANCE TAB
        # ══════════════════════════════════════════════
        with dash_tab1:
            st.markdown("#### Portfolio Performance")

            # ── Sparkline Cards ──
            st.markdown("**Holdings Overview (7-Day Trend)**")

            # Cache sparkline data in session
            if "sparkline_cache" not in st.session_state:
                st.session_state.sparkline_cache = {}

            # Load sparklines for holdings not in cache
            for h in st.session_state.holdings:
                if h.symbol not in st.session_state.sparkline_cache:
                    prices = get_sparkline_data(h.symbol, days=7)
                    st.session_state.sparkline_cache[h.symbol] = prices

            # Display cards in grid
            cols_per_row = 4
            sorted_holdings = sorted(st.session_state.holdings, key=lambda x: x.market_value, reverse=True)

            for i in range(0, min(len(sorted_holdings), 8), cols_per_row):  # Show top 8
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(sorted_holdings) and i + j < 8:
                        h = sorted_holdings[i + j]

                        # Determine card color based on P&L
                        if h.unrealized_pnl_pct > 10:
                            glow_color = "0, 200, 83"
                            border_color = "#00C853"
                        elif h.unrealized_pnl_pct > 0:
                            glow_color = "76, 175, 80"
                            border_color = "#4CAF50"
                        elif h.unrealized_pnl_pct > -10:
                            glow_color = "255, 82, 82"
                            border_color = "#FF5252"
                        else:
                            glow_color = "211, 47, 47"
                            border_color = "#D32F2F"

                        # Get sparkline
                        prices = st.session_state.sparkline_cache.get(h.symbol, [])
                        sparkline_svg = create_sparkline_svg(prices, width=60, height=20) if prices else ""

                        # Create compact card
                        pnl_sign = "+" if h.unrealized_pnl >= 0 else ""
                        card_html = f"""
                        <div style='
                            background: linear-gradient(145deg, #2E2E2E, #252525);
                            border-radius: 10px;
                            padding: 12px;
                            margin: 4px 0;
                            border-left: 3px solid {border_color};
                            box-shadow: 0 0 12px rgba({glow_color}, 0.25);
                        '>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <span style='font-size: 1.1em; font-weight: bold; color: white;'>{h.symbol}</span>
                                {sparkline_svg}
                            </div>
                            <div style='margin-top: 6px; display: flex; justify-content: space-between; align-items: baseline;'>
                                <span style='color: white; font-size: 0.95em;'>${h.market_value:,.0f}</span>
                                <span style='color: {border_color}; font-weight: bold;'>{pnl_sign}{h.unrealized_pnl_pct:.1f}%</span>
                            </div>
                        </div>
                        """
                        with col:
                            st.markdown(card_html, unsafe_allow_html=True)

            if len(st.session_state.holdings) > 8:
                st.caption(f"Showing top 8 of {len(st.session_state.holdings)} holdings by value")

            st.markdown("---")

            # Top/Bottom Performers
            perf_col1, perf_col2 = st.columns(2)

            sorted_by_gain = sorted(st.session_state.holdings,
                                   key=lambda h: h.unrealized_pnl_pct, reverse=True)

            with perf_col1:
                st.markdown("**🚀 Top Performers**")
                for h in sorted_by_gain[:5]:
                    color = "#00C853" if h.unrealized_pnl_pct >= 0 else "#FF5252"
                    st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; padding: 8px;
                                background: #2E2E2E; border-radius: 5px; margin: 4px 0;
                                border-left: 3px solid {color};'>
                        <span><b>{h.symbol}</b> - {h.name[:20]}</span>
                        <span style='color: {color};'>{h.unrealized_pnl_pct:+.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            with perf_col2:
                st.markdown("**📉 Bottom Performers**")
                for h in sorted_by_gain[-5:][::-1]:
                    color = "#00C853" if h.unrealized_pnl_pct >= 0 else "#FF5252"
                    st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; padding: 8px;
                                background: #2E2E2E; border-radius: 5px; margin: 4px 0;
                                border-left: 3px solid {color};'>
                        <span><b>{h.symbol}</b> - {h.name[:20]}</span>
                        <span style='color: {color};'>{h.unrealized_pnl_pct:+.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Holdings by Value Chart
            st.markdown("---")
            st.markdown("#### Holdings by Value")
            holdings_sorted = sorted(st.session_state.holdings,
                                    key=lambda h: h.market_value, reverse=True)

            fig_bar = go.Figure(data=[
                go.Bar(
                    x=[h.symbol for h in holdings_sorted[:15]],
                    y=[h.market_value for h in holdings_sorted[:15]],
                    marker_color=[get_company_color(h.symbol) for h in holdings_sorted[:15]],
                    hovertemplate='<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>'
                )
            ])
            fig_bar.update_layout(
                paper_bgcolor=CHART_BG_COLOR,
                plot_bgcolor=CHART_FACE_COLOR,
                font=dict(color='white'),
                height=350,
                xaxis_title="Symbol",
                yaxis_title="Market Value ($)",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ══════════════════════════════════════════════
        # RISK ANALYTICS TAB
        # ══════════════════════════════════════════════
        with dash_tab2:
            st.markdown("#### Risk Analytics")

            # Calculate risk metrics
            returns = []
            betas = []
            weights = []
            for h in st.session_state.holdings:
                weight = h.market_value / total_invested if total_invested > 0 else 0
                weights.append(weight)
                betas.append(h.beta if h.beta else 1.0)
                # Estimate daily return from gain/loss (simplified)
                if h.total_cost > 0:
                    total_return = h.unrealized_pnl_pct / 100
                    # Assume held for average 1 year, estimate daily
                    returns.append(total_return)

            # Portfolio Beta
            portfolio_beta = sum(w * b for w, b in zip(weights, betas))

            # Simple risk metrics
            if returns:
                avg_return = np.mean(returns)
                volatility = np.std(returns) if len(returns) > 1 else 0.15
                # Sharpe Ratio (assuming 4.5% risk-free rate)
                risk_free = 0.045
                sharpe = (avg_return - risk_free) / volatility if volatility > 0 else 0

                # Max Drawdown (simplified - based on individual holdings)
                max_dd = min(h.unrealized_pnl_pct for h in st.session_state.holdings)

                # Value at Risk (95% confidence, simplified)
                var_95 = total_invested * volatility * 1.645
            else:
                avg_return = 0
                volatility = 0
                sharpe = 0
                max_dd = 0
                var_95 = 0

            # Display Risk Metrics
            risk_cols = st.columns(5)
            with risk_cols[0]:
                beta_color = "inverse" if portfolio_beta > 1.2 else "normal" if portfolio_beta > 0.8 else "off"
                st.metric("Portfolio Beta", f"{portfolio_beta:.2f}",
                         delta="High Risk" if portfolio_beta > 1.2 else "Low Risk" if portfolio_beta < 0.8 else "Moderate",
                         delta_color=beta_color)
            with risk_cols[1]:
                sharpe_color = "normal" if sharpe > 1 else "off" if sharpe > 0 else "inverse"
                st.metric("Sharpe Ratio", f"{sharpe:.2f}",
                         delta="Good" if sharpe > 1 else "Fair" if sharpe > 0 else "Poor",
                         delta_color=sharpe_color)
            with risk_cols[2]:
                st.metric("Volatility", f"{volatility*100:.1f}%")
            with risk_cols[3]:
                st.metric("Max Drawdown", f"{max_dd:.1f}%",
                         delta_color="inverse" if max_dd < -20 else "normal")
            with risk_cols[4]:
                st.metric("VaR (95%)", f"${var_95:,.0f}",
                         help="Maximum expected loss in a day with 95% confidence")

            # ── Detailed Risk Explanations ──
            st.markdown("---")
            st.markdown("#### Understanding Your Risk Metrics")

            with st.expander("📊 What These Numbers Mean", expanded=True):
                explanation_cols = st.columns(2)

                with explanation_cols[0]:
                    # Beta explanation
                    st.markdown("**Portfolio Beta**")
                    if portfolio_beta > 1.5:
                        st.error(f"""
                        🔴 **Very High Risk (Beta: {portfolio_beta:.2f})**

                        Your portfolio is **{((portfolio_beta - 1) * 100):.0f}% more volatile** than the market.

                        **What this means:**
                        - If S&P 500 drops 10%, expect ~{portfolio_beta * 10:.0f}% loss
                        - If S&P 500 rises 10%, expect ~{portfolio_beta * 10:.0f}% gain
                        - High potential reward, but high potential pain

                        **Why:** Heavy in growth/tech stocks, speculative positions, or leveraged ETFs
                        """)
                    elif portfolio_beta > 1.2:
                        st.warning(f"""
                        🟠 **Elevated Risk (Beta: {portfolio_beta:.2f})**

                        Your portfolio is **{((portfolio_beta - 1) * 100):.0f}% more volatile** than the market.

                        **What this means:**
                        - Amplified gains in bull markets
                        - Amplified losses in bear markets
                        - Suitable if you have long time horizon

                        **Why:** Growth-tilted portfolio, tech-heavy, or cyclical stocks
                        """)
                    elif portfolio_beta > 0.8:
                        st.info(f"""
                        🟢 **Moderate Risk (Beta: {portfolio_beta:.2f})**

                        Your portfolio moves **roughly in line** with the market.

                        **What this means:**
                        - Balanced risk/reward profile
                        - Good diversification across sectors
                        - Suitable for most investors

                        **Why:** Mix of growth and defensive positions
                        """)
                    else:
                        st.success(f"""
                        🛡️ **Defensive Portfolio (Beta: {portfolio_beta:.2f})**

                        Your portfolio is **{((1 - portfolio_beta) * 100):.0f}% less volatile** than the market.

                        **What this means:**
                        - More stable during market drops
                        - May lag during strong bull markets
                        - Capital preservation focus

                        **Why:** Utilities, consumer staples, bonds, or dividend stocks
                        """)

                with explanation_cols[1]:
                    # Sharpe explanation
                    st.markdown("**Sharpe Ratio**")
                    if sharpe > 2:
                        st.success(f"""
                        🏆 **Excellent Risk-Adjusted Returns (Sharpe: {sharpe:.2f})**

                        You're getting **exceptional returns** for the risk taken.

                        **Interpretation:**
                        - Top-tier performance
                        - Efficient portfolio construction
                        - Every unit of risk is well-rewarded
                        """)
                    elif sharpe > 1:
                        st.info(f"""
                        ✅ **Good Risk-Adjusted Returns (Sharpe: {sharpe:.2f})**

                        Returns are **adequately compensating** for risk.

                        **Interpretation:**
                        - Acceptable for most strategies
                        - Returns exceed risk-free rate meaningfully
                        - Room for optimization but solid
                        """)
                    elif sharpe > 0:
                        st.warning(f"""
                        ⚠️ **Marginal Returns (Sharpe: {sharpe:.2f})**

                        Returns are **barely beating** risk-free investments.

                        **Interpretation:**
                        - Consider if the risk is worth it
                        - Treasury bills might offer similar returns with no risk
                        - Review underperforming positions
                        """)
                    else:
                        st.error(f"""
                        🔴 **Negative Risk-Adjusted Returns (Sharpe: {sharpe:.2f})**

                        You'd be better off in **Treasury bills**.

                        **Interpretation:**
                        - Taking on risk without adequate reward
                        - Losses or returns below risk-free rate
                        - Urgent need to review strategy
                        """)

            # VaR and Max Drawdown explanations
            with st.expander("📉 Downside Risk Analysis"):
                dd_cols = st.columns(2)

                with dd_cols[0]:
                    st.markdown("**Value at Risk (VaR)**")
                    st.markdown(f"""
                    Your VaR (95%) is **${var_95:,.0f}**

                    **What this means:**
                    - On 95% of trading days, you won't lose more than ${var_95:,.0f}
                    - But on 5% of days (about 12-13 days/year), losses could exceed this
                    - This is a "normal conditions" estimate

                    **In perspective:**
                    - As % of portfolio: **{(var_95/total_invested*100) if total_invested > 0 else 0:.1f}%**
                    - Monthly worst case (approx): **${var_95 * 4.5:,.0f}**
                    """)

                with dd_cols[1]:
                    st.markdown("**Maximum Drawdown**")
                    if max_dd < -30:
                        st.error(f"""
                        🔴 **Severe Drawdown: {max_dd:.1f}%**

                        One or more positions has lost over 30%.

                        **Considerations:**
                        - Is the thesis still intact?
                        - Tax-loss harvesting opportunity?
                        - Average down or cut losses?
                        """)
                    elif max_dd < -15:
                        st.warning(f"""
                        🟠 **Significant Drawdown: {max_dd:.1f}%**

                        Notable loss in some positions.

                        **Considerations:**
                        - Review the losing positions
                        - Check if sector-wide or company-specific
                        - Rebalancing may help
                        """)
                    else:
                        st.success(f"""
                        🟢 **Contained Drawdown: {max_dd:.1f}%**

                        Losses are within normal range.

                        **Good signs:**
                        - Diversification is working
                        - No catastrophic single-stock risk
                        - Portfolio is resilient
                        """)

            st.markdown("---")

            # Concentration Risk
            st.markdown("#### Concentration Risk")
            top_5_weight = sum(sorted(weights, reverse=True)[:5]) * 100

            conc_col1, conc_col2 = st.columns(2)
            with conc_col1:
                st.metric("Top 5 Holdings Weight", f"{top_5_weight:.1f}%",
                         delta="Concentrated" if top_5_weight > 50 else "Diversified",
                         delta_color="inverse" if top_5_weight > 50 else "normal")

            with conc_col2:
                # Sector concentration
                sectors = {}
                for h in st.session_state.holdings:
                    sector = h.sector or "Unknown"
                    weight = h.market_value / total_invested * 100 if total_invested > 0 else 0
                    sectors[sector] = sectors.get(sector, 0) + weight

                if sectors:
                    top_sector = max(sectors.items(), key=lambda x: x[1])
                    st.metric(f"Top Sector: {top_sector[0]}", f"{top_sector[1]:.1f}%")

            # Risk gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=portfolio_beta * 50,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Level", 'font': {'color': 'white'}},
                delta={'reference': 50, 'increasing': {'color': "#FF5252"}, 'decreasing': {'color': "#00C853"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': "#4da6ff"},
                    'bgcolor': CHART_FACE_COLOR,
                    'steps': [
                        {'range': [0, 33], 'color': '#00C853'},
                        {'range': [33, 66], 'color': '#FFB300'},
                        {'range': [66, 100], 'color': '#FF5252'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': portfolio_beta * 50
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor=CHART_BG_COLOR,
                font=dict(color='white'),
                height=300
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Individual Stock Risk Lookup ──
            st.markdown("---")
            st.markdown("#### Individual Stock Risk Lookup")
            st.caption("Search for any stock to see its risk metrics")

            stock_search = st.text_input("Enter Stock Symbol", placeholder="TSLA", key="risk_stock_search").upper().strip()

            if stock_search:
                # Show golden dollar spinner while loading
                loading_placeholder = st.empty()
                loading_placeholder.markdown(show_dollar_spinner(f"Analyzing {stock_search}..."), unsafe_allow_html=True)

                try:
                    stock_data = fetch_security_data(stock_search)
                    # Clear the loading animation
                    loading_placeholder.empty()

                    if stock_data:
                        stock_risk_cols = st.columns(5)
                        with stock_risk_cols[0]:
                            st.metric("Symbol", stock_search)
                        with stock_risk_cols[1]:
                            st.metric("Price", f"${stock_data['current_price']:,.2f}")
                        with stock_risk_cols[2]:
                            beta_val = stock_data.get('beta', 1.0) or 1.0
                            st.metric("Beta", f"{beta_val:.2f}",
                                     delta="High" if beta_val > 1.2 else "Low" if beta_val < 0.8 else "Moderate")
                        with stock_risk_cols[3]:
                            high_52 = stock_data.get('fifty_two_week_high', 0)
                            low_52 = stock_data.get('fifty_two_week_low', 0)
                            if high_52 and low_52:
                                volatility_est = ((high_52 - low_52) / low_52) * 100 if low_52 > 0 else 0
                                st.metric("52W Range", f"{volatility_est:.1f}%")
                            else:
                                st.metric("52W Range", "N/A")
                        with stock_risk_cols[4]:
                            div_yield = stock_data.get('dividend_yield', 0) or 0
                            st.metric("Yield", f"{div_yield*100:.2f}%")

                        # Additional info
                        st.markdown(f"**{stock_data.get('name', stock_search)}** | Sector: {stock_data.get('sector', 'N/A')}")

                        st.markdown("---")

                        # Comprehensive Risk Assessment
                        beta_val = stock_data.get('beta', 1.0) or 1.0
                        high_52 = stock_data.get('fifty_two_week_high', 0) or 0
                        low_52 = stock_data.get('fifty_two_week_low', 0) or 0
                        current_price = stock_data.get('current_price', 0)
                        div_yield = stock_data.get('dividend_yield', 0) or 0
                        sector = stock_data.get('sector', 'Unknown')
                        market_cap = stock_data.get('market_cap', 0) or 0

                        # Calculate additional metrics
                        range_52w = ((high_52 - low_52) / low_52 * 100) if low_52 > 0 else 0
                        distance_from_high = ((high_52 - current_price) / high_52 * 100) if high_52 > 0 else 0
                        distance_from_low = ((current_price - low_52) / low_52 * 100) if low_52 > 0 else 0

                        st.markdown("#### Detailed Risk Analysis")

                        # Risk factors with explanations
                        risk_factors = []
                        positive_factors = []

                        # Beta analysis
                        if beta_val > 1.5:
                            risk_factors.append({
                                "factor": "High Beta",
                                "severity": "high",
                                "explanation": f"Beta of {beta_val:.2f} means this stock is {((beta_val-1)*100):.0f}% more volatile than the market. When the S&P 500 moves 1%, this stock typically moves {beta_val:.1f}%.",
                                "implication": "Expect significant swings during market volatility. Not suitable for conservative investors."
                            })
                        elif beta_val > 1.2:
                            risk_factors.append({
                                "factor": "Elevated Beta",
                                "severity": "medium",
                                "explanation": f"Beta of {beta_val:.2f} indicates above-average market sensitivity.",
                                "implication": "Will amplify both gains and losses relative to the market."
                            })
                        elif beta_val < 0.6:
                            positive_factors.append({
                                "factor": "Low Beta (Defensive)",
                                "explanation": f"Beta of {beta_val:.2f} means this stock is {((1-beta_val)*100):.0f}% less volatile than the market.",
                                "implication": "Good for capital preservation, but may lag in bull markets."
                            })

                        # 52-week range analysis
                        if range_52w > 80:
                            risk_factors.append({
                                "factor": "Extreme Price Swings",
                                "severity": "high",
                                "explanation": f"The stock has swung {range_52w:.0f}% between its 52-week high (${high_52:.2f}) and low (${low_52:.2f}).",
                                "implication": "Highly volatile - could experience dramatic moves in either direction."
                            })
                        elif range_52w > 50:
                            risk_factors.append({
                                "factor": "Wide Trading Range",
                                "severity": "medium",
                                "explanation": f"52-week range of {range_52w:.0f}% indicates significant price movement.",
                                "implication": "Moderate volatility - typical for growth stocks."
                            })

                        # Distance from 52-week high/low
                        if distance_from_high > 30:
                            risk_factors.append({
                                "factor": "Far Below 52W High",
                                "severity": "medium",
                                "explanation": f"Currently {distance_from_high:.0f}% below its 52-week high of ${high_52:.2f}.",
                                "implication": "Could be undervalued or in a downtrend. Research why it's down before buying."
                            })
                        elif distance_from_high < 5:
                            positive_factors.append({
                                "factor": "Near 52W High",
                                "explanation": f"Trading within 5% of its 52-week high - showing strength.",
                                "implication": "Momentum is positive, but may face resistance at the high."
                            })

                        # Sector-specific risks
                        sector_risks = {
                            "Technology": "Tech stocks are sensitive to interest rates and growth expectations. High valuations can lead to sharp corrections.",
                            "Consumer Cyclical": "Highly sensitive to economic cycles. Performs well in expansions, poorly in recessions.",
                            "Financial Services": "Sensitive to interest rates, credit quality, and regulatory changes.",
                            "Energy": "Dependent on oil/gas prices, geopolitical events, and energy transition policies.",
                            "Healthcare": "Regulatory risks, drug approval uncertainty, and political scrutiny on pricing.",
                            "Real Estate": "Interest rate sensitive. Rising rates typically pressure REIT valuations.",
                            "Communication Services": "Competitive pressure, content costs, and regulatory scrutiny.",
                            "Consumer Defensive": "Generally stable but may lag in bull markets. Inflation can pressure margins.",
                            "Utilities": "Interest rate sensitive but provides stable income. Limited growth potential.",
                            "Industrials": "Economically sensitive. Trade policies and supply chain issues are key risks.",
                            "Basic Materials": "Commodity price dependent. Cyclical and economically sensitive.",
                        }

                        if sector in sector_risks:
                            risk_factors.append({
                                "factor": f"{sector} Sector Risk",
                                "severity": "info",
                                "explanation": sector_risks[sector],
                                "implication": "Consider sector concentration in your portfolio."
                            })

                        # Dividend analysis
                        if div_yield > 0.05:  # 5%+
                            risk_factors.append({
                                "factor": "Very High Dividend Yield",
                                "severity": "medium",
                                "explanation": f"Yield of {div_yield*100:.1f}% is unusually high.",
                                "implication": "Could indicate market expects dividend cut, or stock price has fallen significantly. Verify dividend sustainability."
                            })
                        elif div_yield > 0.02:
                            positive_factors.append({
                                "factor": "Dividend Income",
                                "explanation": f"Pays a {div_yield*100:.1f}% dividend yield.",
                                "implication": "Provides income while you hold. Check dividend growth history."
                            })
                        elif div_yield == 0:
                            risk_factors.append({
                                "factor": "No Dividend",
                                "severity": "info",
                                "explanation": "This stock doesn't pay dividends.",
                                "implication": "Returns come entirely from price appreciation. Common for growth stocks."
                            })

                        # Market cap analysis
                        if market_cap > 0:
                            if market_cap < 2_000_000_000:  # < $2B
                                risk_factors.append({
                                    "factor": "Small Cap",
                                    "severity": "medium",
                                    "explanation": f"Market cap of ${market_cap/1_000_000_000:.1f}B is considered small cap.",
                                    "implication": "Higher volatility, less analyst coverage, potential liquidity issues. But also higher growth potential."
                                })
                            elif market_cap > 200_000_000_000:  # > $200B
                                positive_factors.append({
                                    "factor": "Mega Cap",
                                    "explanation": f"Market cap of ${market_cap/1_000_000_000:.0f}B indicates a large, established company.",
                                    "implication": "More stable, liquid, and well-covered by analysts."
                                })

                        # Display risk factors
                        if risk_factors:
                            st.markdown("**⚠️ Risk Factors**")
                            for rf in risk_factors:
                                if rf["severity"] == "high":
                                    st.error(f"""
                                    **{rf['factor']}**

                                    {rf['explanation']}

                                    *{rf['implication']}*
                                    """)
                                elif rf["severity"] == "medium":
                                    st.warning(f"""
                                    **{rf['factor']}**

                                    {rf['explanation']}

                                    *{rf['implication']}*
                                    """)
                                else:
                                    st.info(f"""
                                    **{rf['factor']}**

                                    {rf['explanation']}

                                    *{rf['implication']}*
                                    """)

                        # Display positive factors
                        if positive_factors:
                            st.markdown("**✅ Positive Factors**")
                            for pf in positive_factors:
                                st.success(f"""
                                **{pf['factor']}**

                                {pf['explanation']}

                                *{pf['implication']}*
                                """)

                        # Overall risk score
                        risk_score = 50  # Base
                        risk_score += (beta_val - 1) * 30  # Beta contribution
                        risk_score += (range_52w - 40) * 0.3  # Range contribution
                        if div_yield > 0.02:
                            risk_score -= 10  # Dividend reduces risk score
                        risk_score = max(0, min(100, risk_score))

                        st.markdown("---")
                        st.markdown(f"**Overall Risk Score: {risk_score:.0f}/100**")
                        if risk_score > 70:
                            st.markdown("🔴 **High Risk** - Suitable for aggressive investors with long time horizons")
                        elif risk_score > 40:
                            st.markdown("🟡 **Moderate Risk** - Suitable for balanced portfolios")
                        else:
                            st.markdown("🟢 **Lower Risk** - Suitable for conservative investors")

                    else:
                        st.error(f"Could not find {stock_search}")
                except Exception as e:
                    loading_placeholder.empty()
                    st.error(f"Error fetching data: {e}")

        # ══════════════════════════════════════════════
        # REBALANCING TAB
        # ══════════════════════════════════════════════
        with dash_tab3:
            st.markdown("#### Portfolio Rebalancing Tool")
            st.caption("Set target allocations and see suggested trades to rebalance")

            # Initialize target allocations
            if "target_allocations" not in st.session_state:
                st.session_state.target_allocations = {}

            # Current allocations
            current_alloc = {}
            for h in st.session_state.holdings:
                current_alloc[h.symbol] = {
                    "weight": h.market_value / total_invested * 100 if total_invested > 0 else 0,
                    "value": h.market_value,
                    "price": h.current_price
                }

            # Set targets
            st.markdown("**Set Target Allocations (%)**")
            target_cols = st.columns(4)
            new_targets = {}
            for i, h in enumerate(st.session_state.holdings):
                with target_cols[i % 4]:
                    default_target = st.session_state.target_allocations.get(h.symbol, current_alloc[h.symbol]["weight"])
                    new_targets[h.symbol] = st.number_input(
                        f"{h.symbol}",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(default_target),
                        step=1.0,
                        key=f"target_{h.symbol}"
                    )

            total_target = sum(new_targets.values())
            if abs(total_target - 100) > 0.1:
                st.warning(f"Target allocations sum to {total_target:.1f}% (should be 100%)")
            else:
                st.success("Target allocations sum to 100%")

            if st.button("Calculate Rebalancing Trades", key="calc_rebal"):
                st.session_state.target_allocations = new_targets

                st.markdown("---")
                st.markdown("**Suggested Trades:**")

                trades = []
                for symbol, target_pct in new_targets.items():
                    current = current_alloc[symbol]
                    diff_pct = target_pct - current["weight"]
                    diff_value = (diff_pct / 100) * total_invested
                    shares_to_trade = diff_value / current["price"] if current["price"] > 0 else 0

                    if abs(diff_pct) > 0.5:  # Only show if difference > 0.5%
                        action = "BUY" if diff_pct > 0 else "SELL"
                        color = "#00C853" if diff_pct > 0 else "#FF5252"
                        trades.append({
                            "symbol": symbol,
                            "action": action,
                            "shares": abs(shares_to_trade),
                            "value": abs(diff_value),
                            "diff_pct": diff_pct,
                            "color": color
                        })

                if trades:
                    for trade in sorted(trades, key=lambda x: abs(x["diff_pct"]), reverse=True):
                        st.markdown(f"""
                        <div style='display: flex; justify-content: space-between; padding: 12px;
                                    background: #2E2E2E; border-radius: 5px; margin: 6px 0;
                                    border-left: 4px solid {trade["color"]};'>
                            <span><b>{trade["action"]}</b> {trade["shares"]:.2f} shares of <b>{trade["symbol"]}</b></span>
                            <span style='color: {trade["color"]};'>${trade["value"]:,.2f} ({trade["diff_pct"]:+.1f}%)</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("Your portfolio is already balanced!")

        # ══════════════════════════════════════════════
        # DIVIDEND TRACKER TAB
        # ══════════════════════════════════════════════
        with dash_tab4:
            st.markdown("#### Dividend Tracker")

            # Monthly income projection
            monthly_income = summary["annual_income"] / 12

            div_metric_cols = st.columns(4)
            with div_metric_cols[0]:
                st.metric("Annual Dividend Income", f"${summary['annual_income']:,.2f}")
            with div_metric_cols[1]:
                st.metric("Monthly Income", f"${monthly_income:,.2f}")
            with div_metric_cols[2]:
                st.metric("Portfolio Yield", f"{summary['portfolio_yield']:.2f}%")
            with div_metric_cols[3]:
                dividend_payers = sum(1 for h in st.session_state.holdings if h.dividend_yield > 0)
                st.metric("Dividend Payers", f"{dividend_payers}/{len(st.session_state.holdings)}")

            st.markdown("---")

            # Dividend by holding
            st.markdown("**Dividend Income by Holding**")
            div_data = []
            for h in st.session_state.holdings:
                annual_div = h.market_value * h.dividend_yield
                div_data.append({
                    "Symbol": h.symbol,
                    "Name": h.name[:25],
                    "Yield": f"{h.dividend_yield*100:.2f}%",
                    "Annual Income": f"${annual_div:,.2f}",
                    "Monthly": f"${annual_div/12:,.2f}",
                    "value": annual_div
                })

            div_data_sorted = sorted(div_data, key=lambda x: x["value"], reverse=True)

            # Top dividend payers chart
            top_div = [d for d in div_data_sorted if d["value"] > 0][:10]
            if top_div:
                fig_div = go.Figure(data=[
                    go.Bar(
                        x=[d["Symbol"] for d in top_div],
                        y=[d["value"] for d in top_div],
                        marker_color='#4da6ff',
                        hovertemplate='<b>%{x}</b><br>Annual: $%{y:,.2f}<extra></extra>'
                    )
                ])
                fig_div.update_layout(
                    title="Top Dividend Contributors",
                    paper_bgcolor=CHART_BG_COLOR,
                    plot_bgcolor=CHART_FACE_COLOR,
                    font=dict(color='white'),
                    height=300,
                    xaxis_title="Symbol",
                    yaxis_title="Annual Dividend ($)"
                )
                st.plotly_chart(fig_div, use_container_width=True)

            # Dividend table
            st.dataframe(
                pd.DataFrame(div_data_sorted)[["Symbol", "Name", "Yield", "Annual Income", "Monthly"]],
                use_container_width=True,
                hide_index=True
            )

        # ══════════════════════════════════════════════
        # AI INSIGHTS TAB
        # ══════════════════════════════════════════════
        with dash_tab5:
            st.markdown("#### AI Portfolio Insights")
            st.caption("Smart analysis of your portfolio health and opportunities")

            # Generate insights
            insights = []

            # Portfolio Health Score (0-100)
            health_score = 70  # Base score

            # Diversification check
            num_holdings = len(st.session_state.holdings)
            if num_holdings < 5:
                insights.append({
                    "type": "warning",
                    "title": "Low Diversification",
                    "text": f"You only have {num_holdings} holdings. Consider adding more positions for better diversification.",
                    "impact": -10
                })
            elif num_holdings >= 15:
                insights.append({
                    "type": "positive",
                    "title": "Well Diversified",
                    "text": f"Your portfolio has {num_holdings} holdings, providing good diversification.",
                    "impact": 10
                })

            # Concentration check
            if weights and max(weights) > 0.25:
                top_holding = max(st.session_state.holdings, key=lambda h: h.market_value)
                insights.append({
                    "type": "warning",
                    "title": "High Concentration",
                    "text": f"{top_holding.symbol} represents {max(weights)*100:.1f}% of your portfolio. Consider rebalancing.",
                    "impact": -10
                })

            # Yield analysis
            if summary['portfolio_yield'] > 4:
                insights.append({
                    "type": "positive",
                    "title": "Strong Income Generation",
                    "text": f"Your {summary['portfolio_yield']:.2f}% yield provides excellent passive income.",
                    "impact": 10
                })
            elif summary['portfolio_yield'] < 1:
                insights.append({
                    "type": "info",
                    "title": "Low Dividend Focus",
                    "text": "Your portfolio is growth-focused with minimal dividend income.",
                    "impact": 0
                })

            # Beta/Risk analysis
            if portfolio_beta > 1.3:
                insights.append({
                    "type": "warning",
                    "title": "High Market Risk",
                    "text": f"Portfolio beta of {portfolio_beta:.2f} means higher volatility than the market.",
                    "impact": -5
                })
            elif portfolio_beta < 0.7:
                insights.append({
                    "type": "positive",
                    "title": "Defensive Positioning",
                    "text": f"Low beta of {portfolio_beta:.2f} provides protection in market downturns.",
                    "impact": 5
                })

            # Winners/Losers analysis
            big_winners = [h for h in st.session_state.holdings if h.unrealized_pnl_pct > 50]
            big_losers = [h for h in st.session_state.holdings if h.unrealized_pnl_pct < -20]

            if big_winners:
                insights.append({
                    "type": "positive",
                    "title": "Strong Performers",
                    "text": f"You have {len(big_winners)} holding(s) up more than 50%. Consider taking some profits.",
                    "impact": 5
                })

            if big_losers:
                insights.append({
                    "type": "negative",
                    "title": "Underperformers",
                    "text": f"You have {len(big_losers)} holding(s) down more than 20%. Review your thesis on these positions.",
                    "impact": -5
                })

            # Cash position
            cash_pct = st.session_state.cash_balance / total_with_cash * 100 if total_with_cash > 0 else 0
            if cash_pct > 20:
                insights.append({
                    "type": "info",
                    "title": "High Cash Position",
                    "text": f"You have {cash_pct:.1f}% in cash. Consider deploying capital or keeping as dry powder.",
                    "impact": 0
                })
            elif cash_pct < 5:
                insights.append({
                    "type": "info",
                    "title": "Fully Invested",
                    "text": "Nearly all capital is deployed. Keep some cash for opportunities.",
                    "impact": 0
                })

            # Calculate final health score
            for insight in insights:
                health_score += insight.get("impact", 0)
            health_score = max(0, min(100, health_score))

            # Display Health Score
            if health_score >= 80:
                health_class = "health-excellent"
                health_label = "Excellent"
            elif health_score >= 60:
                health_class = "health-good"
                health_label = "Good"
            elif health_score >= 40:
                health_class = "health-fair"
                health_label = "Fair"
            else:
                health_class = "health-poor"
                health_label = "Needs Attention"

            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background: #2E2E2E; border-radius: 10px; margin-bottom: 20px;'>
                <h2 class='{health_class}' style='margin: 0;'>Portfolio Health: {health_score}/100</h2>
                <p style='color: #888; margin: 5px 0;'>{health_label}</p>
            </div>
            """, unsafe_allow_html=True)

            # Display Insights
            st.markdown("**Key Insights:**")
            for insight in insights:
                icon = {"positive": "✅", "warning": "⚠️", "negative": "❌", "info": "ℹ️"}.get(insight["type"], "•")
                css_class = f"insight-{insight['type']}"
                st.markdown(f"""
                <div class='insight-card {css_class}'>
                    <b>{icon} {insight["title"]}</b><br>
                    <span style='color: #ccc;'>{insight["text"]}</span>
                </div>
                """, unsafe_allow_html=True)

            # Recommendations
            st.markdown("---")
            st.markdown("**Recommendations:**")

            recommendations = []
            if num_holdings < 10:
                recommendations.append("📈 Consider adding more positions for diversification")
            if summary['portfolio_yield'] < 2 and cash_pct < 10:
                recommendations.append("💰 Add dividend stocks for passive income")
            if portfolio_beta > 1.2:
                recommendations.append("🛡️ Add defensive stocks (utilities, consumer staples) to reduce risk")
            if big_losers:
                recommendations.append("🔍 Review underperforming positions - cut losses or average down")
            if cash_pct > 25:
                recommendations.append("💵 Consider deploying excess cash into quality stocks")

            if recommendations:
                for rec in recommendations:
                    st.markdown(f"• {rec}")
            else:
                st.success("Your portfolio looks well-balanced! Keep monitoring and stay diversified.")

            # ── Suggested Stocks ──
            st.markdown("---")
            st.markdown("#### Suggested Stocks to Consider")
            st.caption("Based on your portfolio composition and market conditions")

            # Generate suggestions based on portfolio analysis
            suggested_stocks = []

            # Get current holdings symbols
            current_symbols = [h.symbol for h in st.session_state.holdings]

            # Dividend suggestions if yield is low
            if summary['portfolio_yield'] < 3:
                dividend_picks = [
                    {"symbol": "SCHD", "name": "Schwab US Dividend Equity ETF", "reason": "High-quality dividend stocks", "yield": "3.5%"},
                    {"symbol": "VYM", "name": "Vanguard High Dividend Yield ETF", "reason": "Diversified dividend exposure", "yield": "3.0%"},
                    {"symbol": "O", "name": "Realty Income Corp", "reason": "Monthly dividend REIT", "yield": "5.5%"},
                    {"symbol": "JNJ", "name": "Johnson & Johnson", "reason": "Dividend aristocrat, defensive", "yield": "3.0%"},
                ]
                for pick in dividend_picks:
                    if pick["symbol"] not in current_symbols:
                        suggested_stocks.append({**pick, "category": "💰 Dividend"})

            # Defensive suggestions if beta is high
            if portfolio_beta > 1.1:
                defensive_picks = [
                    {"symbol": "XLU", "name": "Utilities Select Sector ETF", "reason": "Low beta, stable sector", "yield": "3.2%"},
                    {"symbol": "PG", "name": "Procter & Gamble", "reason": "Consumer staples, recession-resistant", "yield": "2.4%"},
                    {"symbol": "KO", "name": "Coca-Cola", "reason": "Defensive dividend stock", "yield": "3.1%"},
                    {"symbol": "WMT", "name": "Walmart", "reason": "Retail staple, low beta", "yield": "1.4%"},
                ]
                for pick in defensive_picks:
                    if pick["symbol"] not in current_symbols:
                        suggested_stocks.append({**pick, "category": "🛡️ Defensive"})

            # Growth suggestions if portfolio is conservative
            if portfolio_beta < 0.9:
                growth_picks = [
                    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "reason": "Tech exposure, growth potential", "yield": "0.5%"},
                    {"symbol": "NVDA", "name": "NVIDIA", "reason": "AI leader, high growth", "yield": "0.03%"},
                    {"symbol": "MSFT", "name": "Microsoft", "reason": "Quality tech, AI exposure", "yield": "0.7%"},
                    {"symbol": "GOOGL", "name": "Alphabet", "reason": "Search & cloud leader", "yield": "0%"},
                ]
                for pick in growth_picks:
                    if pick["symbol"] not in current_symbols:
                        suggested_stocks.append({**pick, "category": "📈 Growth"})

            # Diversification suggestions
            current_sectors = set(h.sector for h in st.session_state.holdings if h.sector)
            if "Healthcare" not in current_sectors:
                suggested_stocks.append({"symbol": "XLV", "name": "Health Care Select Sector ETF", "reason": "Healthcare sector exposure", "yield": "1.5%", "category": "🏥 Healthcare"})
            if "Financial Services" not in current_sectors and "Financials" not in current_sectors:
                suggested_stocks.append({"symbol": "XLF", "name": "Financial Select Sector ETF", "reason": "Financials sector exposure", "yield": "1.8%", "category": "🏦 Financials"})

            # Display suggestions
            if suggested_stocks:
                for stock in suggested_stocks[:8]:  # Show top 8 suggestions
                    st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; padding: 12px;
                                background: #2E2E2E; border-radius: 8px; margin: 6px 0;
                                border-left: 4px solid #4da6ff;'>
                        <div>
                            <span style='color: #888; font-size: 0.85em;'>{stock['category']}</span><br>
                            <b>{stock['symbol']}</b> - {stock['name']}<br>
                            <span style='color: #aaa; font-size: 0.9em;'>{stock['reason']}</span>
                        </div>
                        <div style='text-align: right;'>
                            <span style='color: #4da6ff;'>Yield: {stock['yield']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Your portfolio is well-diversified! No specific suggestions at this time.")


# ═════════════════════════════════════════════
# TAB: Portfolio Manager
# ═════════════════════════════════════════════

with tab_portfolio:
    st.subheader("Portfolio Manager")
    st.caption("Track your stocks, ETFs, bonds, and REITs with live prices and dividend yields.")

    # Initialize session state for holdings
    if "holdings" not in st.session_state:
        st.session_state.holdings = []
    if "editing_holding" not in st.session_state:
        st.session_state.editing_holding = None
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []

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

                # Create interactive Plotly pie chart
                colors = [h["color"] for h in holdings_for_pie]

                fig_hold = go.Figure(data=[go.Pie(
                    labels=[h["symbol"] for h in holdings_for_pie],
                    values=[h["weight"] for h in holdings_for_pie],
                    marker=dict(colors=colors),
                    hovertemplate="<b>%{label}</b><br>" +
                                  "%{customdata}<br>" +
                                  "Weight: %{percent}<br>" +
                                  "Value: $%{value:,.2f}<extra></extra>",
                    customdata=[h["name"] for h in holdings_for_pie],
                    textinfo="label+percent",
                    textposition="outside",
                    textfont=dict(color="white", size=11),
                    hole=0.3,  # Donut style
                )])

                fig_hold.update_layout(
                    title=dict(text="Portfolio Holdings", font=dict(color="white", size=14)),
                    paper_bgcolor=CHART_BG_COLOR,
                    plot_bgcolor=CHART_BG_COLOR,
                    font=dict(color="white"),
                    showlegend=False,
                    height=400,
                    margin=dict(t=50, b=20, l=20, r=20),
                )

                st.plotly_chart(fig_hold, use_container_width=True)

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

        # ── Holdings with Sparkline Cards ──
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

        # ── Holdings Table ──
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

                        # Interactive Price chart with Plotly
                        hist = perf_data["history"]
                        color = "#00ff00" if perf_data['total_return'] >= 0 else "#ff4444"

                        fig_stock = go.Figure()

                        # Main price line
                        fig_stock.add_trace(go.Scatter(
                            x=hist["Date"],
                            y=hist["Close"],
                            mode='lines',
                            name='Price',
                            line=dict(color=color, width=2),
                            fill='tozeroy',
                            fillcolor=color.replace('#', 'rgba(') + ', 0.1)' if color.startswith('#') else 'rgba(0,255,0,0.1)',
                            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Price: $%{y:.2f}<extra></extra>'
                        ))

                        # Add cost basis line if available
                        if selected_holding.avg_cost > 0:
                            fig_stock.add_hline(
                                y=selected_holding.avg_cost,
                                line_dash="dash",
                                line_color="#ffaa00",
                                annotation_text=f"Avg Cost: ${selected_holding.avg_cost:.2f}",
                                annotation_position="right"
                            )

                        fig_stock.update_layout(
                            title=f"{selected_holding.name} ({selected_sym}) - {perf_period}",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template="plotly_dark",
                            paper_bgcolor=CHART_BG_COLOR,
                            plot_bgcolor=CHART_FACE_COLOR,
                            hovermode='x unified',
                            height=400,
                            showlegend=False,
                        )

                        st.plotly_chart(fig_stock, use_container_width=True)

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

                # Interactive Portfolio performance chart with Plotly
                hist = port_perf["history"]
                port_color = "#00ff00" if port_perf['total_return'] >= 0 else "#ff4444"

                fig_port = go.Figure()

                # Portfolio line
                fig_port.add_trace(go.Scatter(
                    x=hist["Date"],
                    y=hist["Portfolio_Return"],
                    mode='lines',
                    name=f"Your Portfolio ({port_perf['total_return']:+.2f}%)",
                    line=dict(color=port_color, width=2),
                    hovertemplate='<b>%{x|%b %d, %Y}</b><br>Portfolio: %{y:+.2f}%<extra></extra>'
                ))

                # Add SPY benchmark
                if "SPY_Return" in hist.columns:
                    spy_final = hist["SPY_Return"].iloc[-1]
                    fig_port.add_trace(go.Scatter(
                        x=hist["Date"],
                        y=hist["SPY_Return"],
                        mode='lines',
                        name=f"S&P 500 ({spy_final:+.2f}%)",
                        line=dict(color="#888888", width=2, dash='dash'),
                        hovertemplate='<b>%{x|%b %d, %Y}</b><br>S&P 500: %{y:+.2f}%<extra></extra>'
                    ))

                fig_port.add_hline(y=0, line_color="white", line_width=0.5, opacity=0.5)

                fig_port.update_layout(
                    title=f"Portfolio Performance - {port_period}",
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    template="plotly_dark",
                    paper_bgcolor=CHART_BG_COLOR,
                    plot_bgcolor=CHART_FACE_COLOR,
                    hovermode='x unified',
                    height=400,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                )

                st.plotly_chart(fig_port, use_container_width=True)
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
        # DIVIDEND HISTORY & DRIP CALCULATOR
        # ══════════════════════════════════════════════
        st.markdown("### Dividend Income Tools")

        div_tool_tab1, div_tool_tab2 = st.tabs(["Dividend History", "DRIP Calculator"])

        with div_tool_tab1:
            st.caption("Actual dividend payments received based on your holdings and purchase dates")

            with st.spinner("Loading dividend history..."):
                div_history = get_dividend_payment_history(st.session_state.holdings, years=3)

            if not div_history.empty:
                # Monthly totals chart (Interactive Plotly)
                monthly_divs = get_monthly_dividend_totals(st.session_state.holdings, years=2)

                if not monthly_divs.empty:
                    fig_div_hist = go.Figure()
                    fig_div_hist.add_trace(go.Bar(
                        x=monthly_divs["Month"],
                        y=monthly_divs["Total"],
                        marker_color="#28a745",
                        text=[f"${v:,.0f}" for v in monthly_divs["Total"]],
                        textposition='outside',
                        textfont=dict(color='white', size=9),
                        hovertemplate='<b>%{x}</b><br>Dividends: $%{y:,.2f}<extra></extra>',
                    ))
                    fig_div_hist.update_layout(
                        title="Historical Dividend Income",
                        xaxis_title="Month",
                        yaxis_title="Dividends Received ($)",
                        plot_bgcolor=CHART_FACE_COLOR,
                        paper_bgcolor=CHART_BG_COLOR,
                        font=dict(color='white'),
                        xaxis=dict(tickangle=45, gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
                        height=400,
                        margin=dict(t=50, b=80),
                    )
                    st.plotly_chart(fig_div_hist, use_container_width=True)

                # Summary metrics
                total_received = div_history["Total"].sum()
                avg_monthly = total_received / max(len(monthly_divs), 1)
                st.success(f"**Total Dividends Received: ${total_received:,.2f}** | Average: ${avg_monthly:,.2f}/month")

                # Recent payments table
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

            # Select a holding for DRIP simulation
            drip_col1, drip_col2 = st.columns(2)

            with drip_col1:
                drip_symbols = [h.symbol for h in st.session_state.holdings if h.dividend_yield > 0]
                if drip_symbols:
                    drip_symbol = st.selectbox("Select Holding", drip_symbols, key="drip_symbol")
                    drip_holding = next((h for h in st.session_state.holdings if h.symbol == drip_symbol), None)
                else:
                    st.warning("No dividend-paying holdings in portfolio")
                    drip_holding = None

            with drip_col2:
                drip_years = st.slider("Projection Years", 5, 30, 10, key="drip_years")

            if drip_holding:
                drip_col3, drip_col4 = st.columns(2)
                with drip_col3:
                    price_growth = st.slider("Annual Price Growth %", 0, 15, 5, key="drip_price_growth") / 100
                with drip_col4:
                    div_growth = st.slider("Annual Dividend Growth %", 0, 15, 3, key="drip_div_growth") / 100

                # Calculate DRIP projection
                initial_value = drip_holding.market_value
                drip_result = calculate_drip_vs_no_drip(
                    initial_investment=initial_value,
                    share_price=drip_holding.current_price,
                    dividend_yield=drip_holding.dividend_yield,
                    years=drip_years,
                    annual_price_growth=price_growth,
                    annual_dividend_growth=div_growth,
                )

                # Summary
                st.markdown(f"**{drip_symbol} DRIP Projection ({drip_years} years)**")
                drip_sum_cols = st.columns(4)
                with drip_sum_cols[0]:
                    st.metric("Starting Value", f"${initial_value:,.0f}")
                with drip_sum_cols[1]:
                    st.metric("With DRIP", f"${drip_result['drip_final_value']:,.0f}")
                with drip_sum_cols[2]:
                    st.metric("Without DRIP", f"${drip_result['no_drip_final_value']:,.0f}")
                with drip_sum_cols[3]:
                    st.metric(
                        "DRIP Advantage",
                        f"${drip_result['drip_advantage']:,.0f}",
                        delta=f"+{drip_result['drip_advantage_pct']:.1f}%",
                        delta_color="normal"
                    )

                # Chart (Interactive Plotly)
                drip_df = drip_result["drip"]
                no_drip_df = drip_result["no_drip"]
                no_drip_total = no_drip_df["Portfolio Value"] + no_drip_df["Cumulative Dividends"]

                fig_drip = go.Figure()

                # With DRIP line
                fig_drip.add_trace(go.Scatter(
                    x=drip_df["Year"],
                    y=drip_df["Portfolio Value"],
                    mode='lines',
                    name='With DRIP',
                    line=dict(color='#00ff00', width=3),
                    fill='tonexty',
                    fillcolor='rgba(0, 255, 0, 0.1)',
                    hovertemplate='<b>Year %{x}</b><br>With DRIP: $%{y:,.0f}<br>Shares: %{customdata:,.2f}<extra></extra>',
                    customdata=drip_df["Shares"],
                ))

                # Without DRIP line (add first so fill works correctly)
                fig_drip.add_trace(go.Scatter(
                    x=no_drip_df["Year"],
                    y=no_drip_total,
                    mode='lines',
                    name='Without DRIP',
                    line=dict(color='#ff6666', width=3, dash='dash'),
                    hovertemplate='<b>Year %{x}</b><br>Without DRIP: $%{y:,.0f}<br>Cash Dividends: $%{customdata:,.0f}<extra></extra>',
                    customdata=no_drip_df["Cumulative Dividends"],
                ))

                fig_drip.update_layout(
                    title=f"{drip_symbol} - DRIP vs No DRIP Projection",
                    xaxis_title="Years",
                    yaxis_title="Total Value ($)",
                    plot_bgcolor=CHART_FACE_COLOR,
                    paper_bgcolor=CHART_BG_COLOR,
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.2)', tickformat='$,.0f'),
                    height=450,
                    legend=dict(bgcolor=CHART_FACE_COLOR, font=dict(color='white')),
                    hovermode='x unified',
                )
                st.plotly_chart(fig_drip, use_container_width=True)

                # Shares growth
                st.caption(f"Shares: {drip_holding.shares:,.2f} → {drip_result['drip_final_shares']:,.2f} with DRIP")

    # ══════════════════════════════════════════════
    # CASH & MONEY MARKET
    # ══════════════════════════════════════════════
    # ══════════════════════════════════════════════
    # IMPORT / EXPORT
    # ══════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Save / Import / Export")

    # Save Portfolio Button (for logged-in users)
    if st.session_state.get("authenticated") and st.session_state.get("username"):
        save_col1, save_col2 = st.columns([1, 3])
        with save_col1:
            if st.button("💾 Save Portfolio", key="save_portfolio", type="primary"):
                try:
                    save_user_data(
                        st.session_state.username,
                        st.session_state.holdings,
                        st.session_state.get("cash_balance", 0),
                        st.session_state.get("watchlist", [])
                    )
                    st.success("Portfolio saved!")
                except Exception as e:
                    st.error(f"Error saving: {e}")
        with save_col2:
            st.caption("Your portfolio will be automatically loaded next time you log in.")
    else:
        st.info("💡 Log in to save your portfolio and have it load automatically next time.")

    st.markdown("---")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("**Export Portfolio**")
        if st.session_state.holdings:
            csv_data = export_holdings_to_csv(st.session_state.holdings)
            st.download_button(
                "Download Holdings CSV",
                csv_data,
                file_name="portfolio_holdings.csv",
                mime="text/csv",
                key="export_holdings"
            )
        else:
            st.caption("No holdings to export")

        if st.session_state.watchlist:
            watch_csv = export_watchlist_to_csv(st.session_state.watchlist)
            st.download_button(
                "Download Watchlist CSV",
                watch_csv,
                file_name="watchlist.csv",
                mime="text/csv",
                key="export_watchlist"
            )

    with exp_col2:
        st.markdown("**Import Portfolio**")
        st.caption("Format: Symbol,Shares,Cost,Date (e.g., AAPL,10,150.00,2024-01-15)")

        # Initialize parsed holdings cache
        if "parsed_csv_holdings" not in st.session_state:
            st.session_state.parsed_csv_holdings = None

        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="import_file",
                                          help="Upload a CSV or Excel file with columns: Symbol, Shares, Cost, Date, Notes")

        if uploaded_file:
            file_name = uploaded_file.name.lower()
            parsed = None

            if file_name.endswith(".csv"):
                try:
                    csv_content = uploaded_file.read().decode("utf-8")
                    st.text_area("CSV Preview", csv_content[:500], height=100, disabled=True)
                    parsed = import_holdings_from_csv(csv_content)
                    if parsed:
                        st.session_state.parsed_csv_holdings = parsed
                        st.success(f"Parsed {len(parsed)} holdings from CSV")
                    else:
                        st.error("Could not parse any holdings from CSV")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

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
                        parsed_list = []
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
                            parsed_list.append({"symbol": sym, "shares": shares, "cost": cost, "purchase_date": date_val, "notes": notes})
                        st.session_state.parsed_csv_holdings = parsed_list
                    else:
                        st.warning("Could not auto-detect columns. Need **Symbol** (or Ticker) and **Shares** (or Quantity).")
                except ImportError:
                    st.error("Excel support requires openpyxl. Run: `pip install openpyxl`")
                except Exception as e:
                    st.error(f"Error reading Excel file: {e}")

        # Show import options if we have parsed data
        if st.session_state.parsed_csv_holdings:
            parsed = st.session_state.parsed_csv_holdings
            st.success(f"Found {len(parsed)} holdings in CSV")

            # Show preview
            preview_data = []
            for item in parsed[:5]:
                preview_data.append({
                    "Symbol": item["symbol"],
                    "Shares": item["shares"],
                    "Cost": f"${item['cost']:.2f}"
                })
            if preview_data:
                st.dataframe(pd.DataFrame(preview_data), hide_index=True)
                if len(parsed) > 5:
                    st.caption(f"...and {len(parsed) - 5} more")

            if st.button("Import All Holdings", key="do_import", type="primary"):
                progress_bar = st.progress(0, text="Fetching live prices...")
                holdings_list = create_holdings_batch(parsed)
                progress_bar.progress(80, text="Building portfolio...")
                success = 0
                # Clear demo data if present
                if st.session_state.get("_demo_loaded"):
                    st.session_state.holdings = []
                    st.session_state._demo_loaded = False
                    st.session_state._demo_dismissed = True
                for holding in holdings_list:
                    existing_idx = next(
                        (i for i, h in enumerate(st.session_state.holdings) if h.symbol == holding.symbol),
                        None
                    )
                    if existing_idx is not None:
                        st.session_state.holdings[existing_idx] = holding
                    else:
                        st.session_state.holdings.append(holding)
                    success += 1
                progress_bar.progress(100, text="Done!")
                st.session_state.parsed_csv_holdings = None  # Clear cache
                failed = len(parsed) - success
                # Auto-save for logged-in users
                if st.session_state.get("authenticated") and st.session_state.get("username"):
                    try:
                        save_user_data(
                            st.session_state.username,
                            st.session_state.holdings,
                            st.session_state.get("cash_balance", 0),
                            st.session_state.get("watchlist", [])
                        )
                    except Exception:
                        pass
                if failed > 0:
                    st.warning(f"Imported {success} holdings ({failed} failed — possibly invalid tickers or rate limited). Switch to the **Dashboard** tab to see your portfolio.")
                else:
                    st.success(f"Imported {success} holdings! Switch to the **Dashboard** tab to see your portfolio.")
                st.rerun()

            if st.button("Clear", key="clear_csv"):
                st.session_state.parsed_csv_holdings = None
                st.rerun()
        elif uploaded_file:
            st.warning("Could not parse CSV. Use format: Symbol,Shares,Cost,Date,Notes")

    # ═══════════════════════════════════════════
    # Smart Tools Section
    # ═══════════════════════════════════════════
    if st.session_state.holdings:
        st.markdown("---")
        st.markdown("### Smart Tools")

        smart_tab1, smart_tab2, smart_tab3, smart_tab4 = st.tabs([
            "🔮 What-If Simulator",
            "💸 Tax-Loss Harvesting",
            "🎯 Similar Investors Bought",
            "📤 Share Portfolio"
        ])

        # ── What-If Simulator ──
        with smart_tab1:
            st.markdown("#### What-If Simulator")
            st.caption("See how buying or selling affects your portfolio")

            sim_col1, sim_col2, sim_col3 = st.columns([1.5, 1, 1])

            with sim_col1:
                sim_options = ["Buy New Stock"] + [f"{h.symbol} - {h.name[:20]}" for h in st.session_state.holdings]
                sim_selection = st.selectbox("Select Stock", sim_options, key="sim_stock")

            with sim_col2:
                sim_action = st.radio("Action", ["Buy", "Sell"], horizontal=True, key="sim_action")

            with sim_col3:
                sim_shares = st.number_input("Shares", min_value=1, value=10, step=1, key="sim_shares")

            # New stock input
            if sim_selection == "Buy New Stock":
                sim_new_symbol = st.text_input("Enter Ticker Symbol", key="sim_new_symbol").upper().strip()
                if sim_new_symbol:
                    sim_data = fetch_security_data(sim_new_symbol)
                    if sim_data:
                        sim_price = sim_data.get("current_price", 0)
                        sim_cost = sim_shares * sim_price

                        st.markdown(f"**{sim_new_symbol}** - {sim_data.get('name', 'Unknown')}")
                        st.markdown(f"Current Price: **${sim_price:,.2f}**")
                        st.markdown(f"Cost to Buy {sim_shares} shares: **${sim_cost:,.2f}**")

                        # Portfolio impact
                        current_total = sum(h.market_value for h in st.session_state.holdings)
                        new_total = current_total + sim_cost
                        new_weight = (sim_cost / new_total) * 100

                        st.markdown("---")
                        st.markdown("**Portfolio Impact:**")
                        impact_cols = st.columns(3)
                        with impact_cols[0]:
                            st.metric("Current Value", f"${current_total:,.0f}")
                        with impact_cols[1]:
                            st.metric("New Value", f"${new_total:,.0f}", delta=f"+${sim_cost:,.0f}")
                        with impact_cols[2]:
                            st.metric("New Position Weight", f"{new_weight:.1f}%")
            else:
                # Existing holding
                sim_symbol = sim_selection.split(" - ")[0]
                sim_holding = next((h for h in st.session_state.holdings if h.symbol == sim_symbol), None)

                if sim_holding:
                    sim_price = sim_holding.current_price
                    sim_value_change = sim_shares * sim_price

                    if sim_action == "Buy":
                        new_shares = sim_holding.shares + sim_shares
                        new_cost = ((sim_holding.avg_cost * sim_holding.shares) + (sim_price * sim_shares)) / new_shares
                        new_value = new_shares * sim_price

                        st.markdown(f"**Buying {sim_shares} more shares of {sim_symbol}**")

                        what_if_cols = st.columns(4)
                        with what_if_cols[0]:
                            st.metric("Current Shares", f"{sim_holding.shares:,.2f}")
                        with what_if_cols[1]:
                            st.metric("New Shares", f"{new_shares:,.2f}", delta=f"+{sim_shares}")
                        with what_if_cols[2]:
                            st.metric("New Avg Cost", f"${new_cost:,.2f}", delta=f"${new_cost - sim_holding.avg_cost:+,.2f}")
                        with what_if_cols[3]:
                            st.metric("Cost", f"${sim_value_change:,.0f}")

                    else:  # Sell
                        if sim_shares > sim_holding.shares:
                            st.error(f"You only have {sim_holding.shares:.2f} shares to sell")
                        else:
                            new_shares = sim_holding.shares - sim_shares
                            realized_pnl = sim_shares * (sim_price - sim_holding.avg_cost)
                            proceeds = sim_shares * sim_price

                            st.markdown(f"**Selling {sim_shares} shares of {sim_symbol}**")

                            what_if_cols = st.columns(4)
                            with what_if_cols[0]:
                                st.metric("Current Shares", f"{sim_holding.shares:,.2f}")
                            with what_if_cols[1]:
                                st.metric("Remaining", f"{new_shares:,.2f}", delta=f"-{sim_shares}")
                            with what_if_cols[2]:
                                st.metric("Proceeds", f"${proceeds:,.0f}")
                            with what_if_cols[3]:
                                pnl_color = "normal" if realized_pnl >= 0 else "inverse"
                                st.metric("Realized P&L", f"${realized_pnl:+,.0f}", delta_color=pnl_color)

                            if realized_pnl < 0:
                                tax_savings = abs(realized_pnl) * 0.25
                                st.info(f"💡 This loss could save ~${tax_savings:,.0f} in taxes (at 25% rate)")

        # ── Tax-Loss Harvesting ──
        with smart_tab2:
            st.markdown("#### Tax-Loss Harvesting Opportunities")
            st.caption("Identify positions to sell for tax benefits")

            opportunities = get_tax_loss_opportunities(st.session_state.holdings)

            if opportunities:
                total_losses = sum(o["loss"] for o in opportunities)
                total_savings = sum(o["tax_savings"] for o in opportunities)

                summary_cols = st.columns(3)
                with summary_cols[0]:
                    st.metric("Harvestable Losses", f"${abs(total_losses):,.0f}")
                with summary_cols[1]:
                    st.metric("Potential Tax Savings", f"${total_savings:,.0f}", delta="at 25% rate")
                with summary_cols[2]:
                    st.metric("Positions with Losses", len(opportunities))

                st.markdown("---")

                for opp in opportunities:
                    loss_severity = "🔴" if opp["loss_pct"] < -20 else "🟠" if opp["loss_pct"] < -10 else "🟡"

                    with st.expander(f"{loss_severity} {opp['symbol']} — Loss: ${abs(opp['loss']):,.0f} ({opp['loss_pct']:.1f}%)"):
                        opp_cols = st.columns([2, 1, 1])

                        with opp_cols[0]:
                            st.markdown(f"**{opp['name']}**")
                            st.markdown(f"Sector: {opp['sector'] or 'Unknown'}")
                            st.markdown(f"Shares: {opp['shares']:,.2f}")

                        with opp_cols[1]:
                            st.markdown("**Cost Basis**")
                            st.markdown(f"${opp['avg_cost']:,.2f}/share")
                            st.markdown(f"Total: ${opp['avg_cost'] * opp['shares']:,.0f}")

                        with opp_cols[2]:
                            st.markdown("**Current Value**")
                            st.markdown(f"${opp['current_price']:,.2f}/share")
                            st.markdown(f"Tax Savings: **${opp['tax_savings']:,.0f}**")

                        if opp["replacements"]:
                            st.markdown("---")
                            st.markdown("**Similar Stocks to Maintain Exposure** (avoid wash sale):")
                            st.markdown(", ".join([f"`{r}`" for r in opp["replacements"]]))
                            st.caption("⚠️ Wait 31 days before repurchasing the same stock to avoid wash sale rules")
            else:
                st.success("🎉 No tax-loss harvesting opportunities — all positions are profitable!")

        # ── Similar Investors Bought ──
        with smart_tab3:
            st.markdown("#### Stocks Similar Investors Bought")
            st.caption("Based on your current holdings, here are stocks that complement your portfolio")

            recommendations = get_similar_stocks_recommendations(st.session_state.holdings)

            if recommendations:
                # Group by type
                cluster_recs = [r for r in recommendations if r["type"] == "cluster"]
                sector_recs = [r for r in recommendations if r["type"] == "sector"]
                diversify_recs = [r for r in recommendations if r["type"] == "diversify"]

                if cluster_recs:
                    st.markdown("**🔗 Often Held Together**")
                    for rec in cluster_recs[:4]:
                        rec_data = fetch_security_data(rec["symbol"])
                        if rec_data:
                            price = rec_data.get("current_price", 0)
                            name = rec_data.get("name", rec["symbol"])

                            st.markdown(f"""
                            <div style='background: #2E2E2E; padding: 12px; border-radius: 8px; margin: 8px 0;
                                        border-left: 3px solid #4da6ff;'>
                                <b>{rec['symbol']}</b> — {name[:30]}<br>
                                <span style='color: #888;'>Because you own {rec['based_on']}</span><br>
                                <span style='color: #4da6ff;'>${price:,.2f}</span>
                            </div>
                            """, unsafe_allow_html=True)

                if sector_recs:
                    st.markdown("**📊 Sector Leaders**")
                    for rec in sector_recs[:3]:
                        rec_data = fetch_security_data(rec["symbol"])
                        if rec_data:
                            price = rec_data.get("current_price", 0)
                            name = rec_data.get("name", rec["symbol"])

                            st.markdown(f"""
                            <div style='background: #2E2E2E; padding: 12px; border-radius: 8px; margin: 8px 0;
                                        border-left: 3px solid #00C853;'>
                                <b>{rec['symbol']}</b> — {name[:30]}<br>
                                <span style='color: #888;'>{rec['reason']}</span><br>
                                <span style='color: #00C853;'>${price:,.2f}</span>
                            </div>
                            """, unsafe_allow_html=True)

                if diversify_recs:
                    st.markdown("**🌐 Diversification Ideas**")
                    for rec in diversify_recs[:3]:
                        rec_data = fetch_security_data(rec["symbol"])
                        if rec_data:
                            price = rec_data.get("current_price", 0)
                            name = rec_data.get("name", rec["symbol"])

                            st.markdown(f"""
                            <div style='background: #2E2E2E; padding: 12px; border-radius: 8px; margin: 8px 0;
                                        border-left: 3px solid #FFB300;'>
                                <b>{rec['symbol']}</b> — {name[:30]}<br>
                                <span style='color: #888;'>{rec['reason']}</span><br>
                                <span style='color: #FFB300;'>${price:,.2f}</span>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("Add more holdings to get personalized recommendations")

        # ── Share Portfolio ──
        with smart_tab4:
            st.markdown("#### Share Your Portfolio")
            st.caption("Generate a shareable snapshot of your portfolio")

            summary = calculate_portfolio_summary(st.session_state.holdings)
            username = st.session_state.get("username", "Investor")

            # Generate snapshot
            snapshot_text = generate_portfolio_snapshot(st.session_state.holdings, summary, username)

            # Display preview
            st.markdown("**Preview:**")
            st.code(snapshot_text, language=None)

            # Copy button (download as text file)
            share_col1, share_col2 = st.columns(2)

            with share_col1:
                st.download_button(
                    "📥 Download Snapshot",
                    snapshot_text,
                    file_name=f"portfolio_snapshot_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    key="download_snapshot"
                )

            with share_col2:
                # Generate a simple shareable stats card
                total_value = summary.get("total_value", 0)
                total_pnl_pct = summary.get("total_pnl_pct", 0)
                num_holdings = len(st.session_state.holdings)

                share_card = f"""
Portfolio Stats:
💰 ${total_value:,.0f} total value
📈 {total_pnl_pct:+.1f}% all-time return
📊 {num_holdings} holdings

#RCRPortfolioTrackers #Investing
                """

                st.download_button(
                    "🐦 Copy for Social",
                    share_card.strip(),
                    file_name="portfolio_share.txt",
                    mime="text/plain",
                    key="share_social"
                )

            st.markdown("---")
            st.markdown("**Privacy Note:** This snapshot does not include specific share counts or cost basis. Only symbols, values, and percentage returns are shown.")


# ═════════════════════════════════════════════
# TAB 1: Cash & Money Market
# ═════════════════════════════════════════════

with tab_cash:
    st.subheader("Cash & Money Market")
    st.caption("Track cash holdings and calculate money market growth")

    # Initialize cash balance
    if "cash_balance" not in st.session_state:
        st.session_state.cash_balance = 0.0

    cash_tab1, cash_tab2, cash_tab3 = st.tabs(["Cash Holdings", "Money Market Calculator", "Best Yields"])

    with cash_tab1:
        st.markdown("### Cash Holdings")
        st.caption("Track your cash position in the portfolio")

        cash_col1, cash_col2, cash_col3 = st.columns([2, 1, 1])

        with cash_col1:
            new_cash = st.number_input(
                "Cash Balance ($)",
                min_value=0.0,
                value=st.session_state.cash_balance,
                step=100.0,
                key="cash_input",
                help="Enter your total cash holdings"
            )

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

        # Show cash in portfolio context
        if st.session_state.cash_balance > 0 or st.session_state.holdings:
            total_invested = sum(h.market_value for h in st.session_state.holdings) if st.session_state.holdings else 0
            total_portfolio = total_invested + st.session_state.cash_balance
            cash_pct = (st.session_state.cash_balance / total_portfolio * 100) if total_portfolio > 0 else 0

            st.markdown("---")
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
        st.markdown("### Money Market Calculator")
        st.caption("Calculate money market account growth with interest and contributions")

        mm_col1, mm_col2 = st.columns(2)

        with mm_col1:
            mm_initial = st.number_input(
                "Initial Deposit ($)",
                min_value=0.0,
                value=10000.0,
                step=1000.0,
                key="mm_initial"
            )
            mm_rate = st.slider(
                "Annual Interest Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=4.5,
                step=0.1,
                key="mm_rate",
                help="Current money market rates are typically 4-5%"
            )

        with mm_col2:
            mm_contribution = st.number_input(
                "Monthly Contribution ($)",
                min_value=0.0,
                value=500.0,
                step=100.0,
                key="mm_contribution"
            )
            mm_years = st.slider(
                "Time Period (Years)",
                min_value=1,
                max_value=30,
                value=5,
                key="mm_years"
            )

        if st.button("Calculate Growth", key="calc_mm"):
            # Calculate money market growth
            monthly_rate = mm_rate / 100 / 12
            months = mm_years * 12

            # Calculate future value with contributions
            balance_history = [mm_initial]
            balance = mm_initial
            total_contributions = mm_initial
            total_interest = 0

            for month in range(1, months + 1):
                interest = balance * monthly_rate
                total_interest += interest
                balance = balance + interest + mm_contribution
                total_contributions += mm_contribution
                if month % 12 == 0:  # Record yearly
                    balance_history.append(balance)

            # Results
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

            # Growth chart
            years_list = list(range(mm_years + 1))

            fig_mm = go.Figure()
            fig_mm.add_trace(go.Scatter(
                x=years_list,
                y=balance_history,
                mode='lines+markers',
                name='Balance',
                line=dict(color='#4da6ff', width=3),
                fill='tozeroy',
                fillcolor='rgba(77, 166, 255, 0.2)',
                hovertemplate='<b>Year %{x}</b><br>Balance: $%{y:,.2f}<extra></extra>'
            ))

            # Add contribution line for reference
            contribution_line = [mm_initial + (mm_contribution * 12 * y) for y in years_list]
            fig_mm.add_trace(go.Scatter(
                x=years_list,
                y=contribution_line,
                mode='lines',
                name='Contributions Only',
                line=dict(color='#888888', width=2, dash='dash'),
                hovertemplate='<b>Year %{x}</b><br>Contributions: $%{y:,.2f}<extra></extra>'
            ))

            fig_mm.update_layout(
                title=f"Money Market Growth ({mm_rate}% APY)",
                xaxis_title="Years",
                yaxis_title="Balance ($)",
                paper_bgcolor=CHART_BG_COLOR,
                plot_bgcolor=CHART_FACE_COLOR,
                font=dict(color='white'),
                height=400,
                legend=dict(bgcolor=CHART_FACE_COLOR, font=dict(color='white')),
                hovermode='x unified',
            )
            st.plotly_chart(fig_mm, use_container_width=True)

            st.info(f"At {mm_rate}% APY with ${mm_contribution:,.0f}/month contributions, "
                    f"your ${mm_initial:,.0f} grows to **${balance:,.0f}** in {mm_years} years. "
                    f"Interest earned: **${total_interest:,.0f}**")

    with cash_tab3:
        st.markdown("### Best Money Market & Savings Yields")
        st.caption("Current top yields from banks and brokerages (rates as of January 2026)")

        # Best yields data - Updated January 2026 from findbanks.com
        best_yields = [
            {"name": "SoFi Checking & Savings", "type": "High-Yield Savings", "apy": 4.00, "min": "$0", "notes": "Up to $300 bonus, FDIC insured up to $3M"},
            {"name": "Valley Bank", "type": "High-Yield Savings", "apy": 3.90, "min": "$1", "notes": "Up to $1,500 bonus with code HEADSTART"},
            {"name": "Western Alliance Bank", "type": "High-Yield Savings", "apy": 3.80, "min": "$0", "notes": "No maintenance fees, FDIC insured"},
            {"name": "E*TRADE Premium Savings", "type": "High-Yield Savings", "apy": 3.75, "min": "$0", "notes": "6-month promo rate, up to $2,000 bonus"},
            {"name": "CIT Bank Platinum Savings", "type": "High-Yield Savings", "apy": 3.75, "min": "$100", "notes": "Tiered rates, FDIC insured"},
            {"name": "American Express Savings", "type": "High-Yield Savings", "apy": 3.30, "min": "$0", "notes": "No monthly fees, daily compounding"},
            {"name": "Vanguard Federal Money Market (VMFXX)", "type": "Money Market Fund", "apy": 4.20, "min": "$3,000", "notes": "Brokerage account required"},
            {"name": "Fidelity Money Market (SPAXX)", "type": "Money Market Fund", "apy": 4.00, "min": "$0", "notes": "Fidelity account required"},
            {"name": "Schwab Value Advantage Money (SWVXX)", "type": "Money Market Fund", "apy": 4.10, "min": "$0", "notes": "Schwab account required"},
            {"name": "Treasury Bills (4-Week)", "type": "Government", "apy": 4.25, "min": "$100", "notes": "State tax exempt"},
        ]

        # Sort by APY
        best_yields.sort(key=lambda x: x["apy"], reverse=True)

        # Display best yields
        st.markdown("#### Top Yields Available")

        for i, item in enumerate(best_yields):
            if i < 3:
                badge = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            else:
                badge = ""

            color = "#00C853" if item["apy"] >= 5.0 else "#4da6ff" if item["apy"] >= 4.5 else "#FFB300"

            st.markdown(f"""
            <div style='display: flex; justify-content: space-between; padding: 14px;
                        background: #2E2E2E; border-radius: 8px; margin: 8px 0;
                        border-left: 4px solid {color};'>
                <div>
                    <b>{badge} {item['name']}</b><br>
                    <span style='color: #888; font-size: 0.9em;'>{item['type']} • Min: {item['min']}</span><br>
                    <span style='color: #aaa; font-size: 0.85em;'>{item['notes']}</span>
                </div>
                <div style='text-align: right;'>
                    <span style='color: {color}; font-size: 1.4em; font-weight: bold;'>{item['apy']:.2f}%</span><br>
                    <span style='color: #888; font-size: 0.85em;'>APY</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Calculator with best rate
        st.markdown("#### Quick Calculator with Best Rate")
        calc_col1, calc_col2 = st.columns(2)

        with calc_col1:
            calc_amount = st.number_input("Amount to Invest ($)", min_value=0.0, value=10000.0, step=1000.0, key="best_yield_calc")

        with calc_col2:
            best_rate = best_yields[0]["apy"]
            st.metric("Best Current Rate", f"{best_rate:.2f}%", delta=best_yields[0]["name"][:30])

        if calc_amount > 0:
            annual_earnings = calc_amount * (best_rate / 100)
            monthly_earnings = annual_earnings / 12

            earn_cols = st.columns(3)
            with earn_cols[0]:
                st.metric("Monthly Earnings", f"${monthly_earnings:,.2f}")
            with earn_cols[1]:
                st.metric("Annual Earnings", f"${annual_earnings:,.2f}")
            with earn_cols[2]:
                five_year = calc_amount * ((1 + best_rate/100) ** 5) - calc_amount
                st.metric("5-Year Earnings", f"${five_year:,.2f}")

        st.caption("⚠️ Rates change frequently. Always verify current rates before opening accounts.")


# ═════════════════════════════════════════════
# TAB 2: Calendar & News
# ═════════════════════════════════════════════

with tab_calendar:
    st.subheader("Calendar & News")
    st.caption("Today's top finance news, economic events, and portfolio calendar")

    # ══════════════════════════════════════════════
    # TODAY'S TOP FINANCE NEWS (always visible, no holdings needed)
    # ══════════════════════════════════════════════
    st.markdown("### Today's Finance News")
    st.markdown(f"**{datetime.now().strftime('%A, %B %d, %Y')}**")

    @st.cache_data(ttl=900, show_spinner=False)
    def fetch_market_news():
        """Fetch market news from major tickers. Cached for 15 minutes."""
        import yfinance as yf
        results = []
        seen = set()
        source_map = {"SPY": "S&P 500", "QQQ": "Nasdaq", "DIA": "Dow Jones"}
        for mt in ["SPY", "QQQ", "DIA"]:
            try:
                t = yf.Ticker(mt)
                try:
                    raw = t.news
                except Exception:
                    raw = None
                if not raw:
                    continue
                # Handle both list and dict formats from different yfinance versions
                articles = raw if isinstance(raw, list) else raw.get("news", raw.get("items", []))
                for article in articles[:6]:
                    try:
                        if isinstance(article, dict):
                            # New yfinance format: article has "content" wrapper
                            content = article.get("content", article)
                        else:
                            content = article
                        if isinstance(content, dict):
                            title = content.get("title", "")
                        else:
                            title = getattr(content, "title", "")
                        if not title or title in seen:
                            continue
                        seen.add(title)
                        if isinstance(content, dict):
                            prov = content.get("provider", {})
                            publisher = prov.get("displayName", "") if isinstance(prov, dict) else str(prov)
                            link = ""
                            if "canonicalUrl" in content and isinstance(content["canonicalUrl"], dict):
                                link = content["canonicalUrl"].get("url", "")
                            elif "clickThroughUrl" in content and isinstance(content["clickThroughUrl"], dict):
                                link = content["clickThroughUrl"].get("url", "")
                            else:
                                link = content.get("link", content.get("url", ""))
                            pub_str = content.get("pubDate", "")
                            pub_time = datetime.now()
                            if pub_str:
                                try:
                                    pub_time = datetime.fromisoformat(str(pub_str).replace("Z", "+00:00")).replace(tzinfo=None)
                                except Exception:
                                    pass
                            elif isinstance(article, dict) and article.get("providerPublishTime"):
                                try:
                                    pub_time = datetime.fromtimestamp(article["providerPublishTime"])
                                except Exception:
                                    pass
                        else:
                            publisher = getattr(content, "publisher", "")
                            link = getattr(content, "link", getattr(content, "url", ""))
                            pub_time = datetime.now()
                        results.append({
                            "title": title,
                            "publisher": publisher or "Financial News",
                            "link": link or "#",
                            "published": pub_time,
                            "symbol": source_map.get(mt, "Markets"),
                        })
                    except Exception:
                        continue
            except Exception:
                continue
        results.sort(key=lambda x: x["published"], reverse=True)
        return results

    with st.spinner("Loading market news..."):
        try:
            market_news = fetch_market_news()
            if market_news:
                save_fallback("market_news", market_news)
        except Exception:
            market_news = []
        if not market_news:
            cached_news, age_min = load_fallback("market_news", max_age_hours=6)
            if cached_news:
                market_news = cached_news
                st.warning(f"Showing cached news from ~{age_min} minutes ago (live fetch failed).")

    if market_news:
        for news in market_news[:12]:
            time_ago = datetime.now() - news["published"]
            if time_ago.days > 0:
                time_str = f"{time_ago.days}d ago"
            elif time_ago.seconds > 3600:
                time_str = f"{time_ago.seconds // 3600}h ago"
            else:
                time_str = f"{max(1, time_ago.seconds // 60)}m ago"
            cat_colors = {"S&P 500": "#28a745", "Nasdaq": "#6C63FF", "Dow Jones": "#ff6b6b"}
            badge_color = cat_colors.get(news["symbol"], "#4da6ff")
            st.markdown(
                f"<div style='padding:12px;margin:8px 0;background:{CHART_FACE_COLOR};border-radius:8px;border-left:4px solid {badge_color};'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
                f"<span style='background:{badge_color};padding:3px 10px;border-radius:4px;font-size:0.85em;font-weight:bold;color:white;'>{news['symbol']}</span>"
                f"<span style='color:#888;font-size:0.8em;'>{time_str} • {news['publisher']}</span></div>"
                f"<a href='{news['link']}' target='_blank' style='color:#4da6ff;text-decoration:none;font-weight:500;font-size:1.05em;'>{news['title']} ↗</a></div>",
                unsafe_allow_html=True
            )
    elif not market_news:
        show_styled_empty_state(
            icon="📰",
            title="No Market News Available",
            message="News could not be loaded right now. This may be due to rate limiting — try refreshing in a minute.",
        )

    st.markdown("---")

    # ══════════════════════════════════════════════
    # PORTFOLIO CALENDAR (if holdings exist)
    # ══════════════════════════════════════════════
    if st.session_state.holdings:
        st.markdown("### Dividend & Earnings Calendar")

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
            if st.button("◀ Prev", key="cal_prev_month"):
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
            st.markdown(
                f"<h3 style='text-align: center; margin: 0;'>{month_name} {st.session_state.cal_year}</h3>",
                unsafe_allow_html=True
            )

        with nav_col5:
            if st.button("Next ▶", key="cal_next_month"):
                if st.session_state.cal_month == 12:
                    st.session_state.cal_month = 1
                    st.session_state.cal_year += 1
                else:
                    st.session_state.cal_month += 1
                st.rerun()

        # Build calendar data
        with st.spinner("Loading calendar..."):
            cal_data = build_combined_calendar(
                st.session_state.holdings,
                st.session_state.cal_year,
                st.session_state.cal_month
            )

        # Month summary
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            if cal_data["dividend_events"]:
                total_div = sum(e.expected_income for e in cal_data["dividend_events"])
                st.success(
                    f"**Dividends: ${total_div:,.2f}** from "
                    f"{len(cal_data['dividend_events'])} payment(s)"
                )
            else:
                st.info("No dividends this month")
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
                                div_details = [f"{e.symbol}: ${e.expected_income:.0f}" for e in div_events[:2]]
                                if len(div_events) > 2:
                                    div_details.append(f"+{len(div_events)-2} more")
                                lines.append(f"💵 {', '.join(div_details)}")
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
                            # Dividend only - green - show per-stock amounts
                            total_income = sum(e.expected_income for e in div_events)
                            div_lines = [f"{e.symbol}: ${e.expected_income:.2f}" for e in div_events[:3]]
                            if len(div_events) > 3:
                                div_lines.append(f"+{len(div_events)-3} more")
                            div_content = "<br>".join(
                                f"<span style='font-size: 0.65em; color: #000000;'>{l}</span>"
                                for l in div_lines
                            )
                            st.markdown(
                                f"<div style='background-color: #d4edda; padding: 4px; "
                                f"border-radius: 4px; text-align: center; min-height: 70px; color: #000000;'>"
                                f"<strong style='color: #000000;'>{day_num}</strong><br>"
                                f"<span style='font-size: 0.7em; color: #000000; font-weight: bold;'>💵 ${total_income:,.0f}</span><br>"
                                f"{div_content}"
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

        # Detailed events - Fed/Economic with clickable links
        if cal_data.get("fed_events"):
            st.markdown("#### Fed & Economic Events")

            for event in cal_data["fed_events"]:
                importance_badge = "🔴" if event.importance == "HIGH" else "🟡" if event.importance == "MEDIUM" else "🟢"

                col_date, col_event, col_link = st.columns([2, 4, 1])
                with col_date:
                    st.markdown(f"**{event.event_date.strftime('%b %d, %Y')}**")
                with col_event:
                    st.markdown(f"{importance_badge} **{event.event_type}** - {event.description}")
                with col_link:
                    if event.url:
                        st.markdown(f"[📎 Details]({event.url})")

            st.caption(
                "Fed/Economic events can cause significant market volatility. "
                "Consider adjusting positions before high-importance events."
            )

        # 12-Month Projection Chart (Interactive Plotly)
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
                fig_proj = go.Figure()
                fig_proj.add_trace(go.Bar(
                    x=proj_df["Month"],
                    y=proj_df["Income"],
                    marker_color="#28a745",
                    text=[f"${v:,.0f}" for v in proj_df["Income"]],
                    textposition='outside',
                    textfont=dict(color='white', size=10),
                    hovertemplate='<b>%{x}</b><br>Expected Income: $%{y:,.2f}<br>Dividend Payments: %{customdata}<extra></extra>',
                    customdata=proj_df["Payments"],
                ))
                fig_proj.update_layout(
                    title="12-Month Dividend Income Projection",
                    xaxis_title="Month",
                    yaxis_title="Expected Income ($)",
                    plot_bgcolor=CHART_FACE_COLOR,
                    paper_bgcolor=CHART_BG_COLOR,
                    font=dict(color='white'),
                    xaxis=dict(tickangle=45, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
                    height=400,
                    margin=dict(t=50, b=80),
                )
                st.plotly_chart(fig_proj, use_container_width=True)

                # Annual summary
                annual_total = proj_df["Income"].sum()
                st.info(
                    f"**Projected Annual Dividend Income: ${annual_total:,.2f}** "
                    f"(${annual_total/12:,.2f}/month average)"
                )

        st.markdown("---")

        # ══════════════════════════════════════════════
        # PORTFOLIO NEWS
        # ══════════════════════════════════════════════
        st.markdown("### Portfolio News")
        st.caption("Recent headlines specific to your holdings")

        with st.spinner("Loading news..."):
            news_items = get_portfolio_news(st.session_state.holdings, limit_per_stock=3)

        if news_items:
            # Filter selector
            news_symbols = ["All"] + sorted(list(set(n.symbol for n in news_items)))
            news_filter = st.selectbox("Filter by Stock", news_symbols, key="cal_news_filter")

            filtered_news = news_items if news_filter == "All" else [n for n in news_items if n.symbol == news_filter]

            for i, news in enumerate(filtered_news[:20]):
                time_ago = datetime.now() - news.published
                if time_ago.days > 0:
                    time_str = f"{time_ago.days}d ago"
                elif time_ago.seconds > 3600:
                    time_str = f"{time_ago.seconds // 3600}h ago"
                else:
                    time_str = f"{time_ago.seconds // 60}m ago"

                # Get company color for the badge
                badge_color = get_company_color(news.symbol)

                st.markdown(
                    f"""
                    <div style='padding: 12px; margin: 8px 0; background-color: {CHART_FACE_COLOR}; border-radius: 8px; border-left: 4px solid {badge_color};'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                            <span style='background-color: {badge_color}; padding: 3px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; color: white;'>{news.symbol}</span>
                            <span style='color: #888; font-size: 0.8em;'>{time_str} • {news.publisher}</span>
                        </div>
                        <a href='{news.link}' target='_blank' style='color: #4da6ff; text-decoration: none; font-weight: 500; font-size: 1.05em; display: block;'>
                            {news.title} ↗
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No recent news found for your holdings")

    # ══════════════════════════════════════════════
    # WATCHLIST
    # ══════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Watchlist")
    st.caption("Track stocks you're interested in but don't own yet")

    # Add to watchlist
    watch_col1, watch_col2, watch_col3, watch_col4 = st.columns([1.5, 1, 2, 1])
    with watch_col1:
        watch_symbol = st.text_input("Symbol", placeholder="TSLA", key="cal_watch_symbol").upper().strip()
    with watch_col2:
        watch_target = st.number_input("Target Price ($)", min_value=0.0, value=0.0, key="cal_watch_target",
                                        help="Optional buy target price")
    with watch_col3:
        watch_notes = st.text_input("Notes", placeholder="Waiting for pullback...", key="cal_watch_notes")
    with watch_col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add to Watchlist", key="cal_add_watch"):
            if watch_symbol:
                with st.spinner(f"Adding {watch_symbol}..."):
                    item = create_watchlist_item(
                        watch_symbol,
                        watch_target if watch_target > 0 else None,
                        watch_notes if watch_notes else None
                    )
                    if item:
                        st.session_state.watchlist.append(item)
                        st.success(f"Added {watch_symbol} to watchlist")
                        st.rerun()
                    else:
                        st.error(f"Could not find {watch_symbol}")

    # Display watchlist
    if st.session_state.watchlist:
        # Refresh button
        if st.button("Refresh Prices", key="cal_refresh_watchlist"):
            with st.spinner("Refreshing..."):
                st.session_state.watchlist = [refresh_watchlist_item(w) for w in st.session_state.watchlist]
                st.rerun()

        watch_data = []
        for w in st.session_state.watchlist:
            target_str = f"${w.target_price:.2f}" if w.target_price else "-"
            at_target = "✓" if w.target_price and w.current_price <= w.target_price else ""
            watch_data.append({
                "Symbol": w.symbol,
                "Name": w.name[:25] + "..." if len(w.name) > 25 else w.name,
                "Price": f"${w.current_price:.2f}",
                "Target": target_str,
                "At Target": at_target,
                "Change": f"{w.change_since_add_pct:+.1f}%",
                "Yield": f"{w.dividend_yield*100:.2f}%",
                "52W High": f"${w.fifty_two_week_high:.2f}" if w.fifty_two_week_high else "-",
                "52W Low": f"${w.fifty_two_week_low:.2f}" if w.fifty_two_week_low else "-",
                "Notes": w.notes or "-",
            })

        st.dataframe(pd.DataFrame(watch_data), use_container_width=True, hide_index=True)

        # Move to portfolio / Remove buttons
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
                    if st.button(f"❌", key=f"cal_del_watch_{w.symbol}_{i}"):
                        st.session_state.watchlist.pop(i)
                        st.rerun()

        # ══════════════════════════════════════════════
        # NEWS FOR WATCHLIST ITEMS
        # ══════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### Watchlist News")
        st.caption("Recent news for stocks on your watchlist (last 30 days)")

        with st.spinner("Loading watchlist news..."):
            watchlist_news = []
            for w in st.session_state.watchlist:
                news_items = get_stock_news(w.symbol, limit=5)
                # Filter to last 30 days
                thirty_days_ago = datetime.now() - relativedelta(days=30)
                for n in news_items:
                    if n.published >= thirty_days_ago:
                        watchlist_news.append(n)

            # Sort by date, most recent first
            watchlist_news.sort(key=lambda x: x.published, reverse=True)

        if watchlist_news:
            # Filter selector
            watch_news_symbols = ["All"] + sorted(list(set(n.symbol for n in watchlist_news)))
            watch_news_filter = st.selectbox("Filter by Stock", watch_news_symbols, key="watchlist_news_filter")

            filtered_watch_news = watchlist_news if watch_news_filter == "All" else [n for n in watchlist_news if n.symbol == watch_news_filter]

            # Get ticker names for display
            ticker_names = {w.symbol: w.name for w in st.session_state.watchlist}

            for i, news in enumerate(filtered_watch_news[:20]):
                time_ago = datetime.now() - news.published
                if time_ago.days > 0:
                    time_str = f"{time_ago.days}d ago"
                elif time_ago.seconds > 3600:
                    time_str = f"{time_ago.seconds // 3600}h ago"
                else:
                    time_str = f"{time_ago.seconds // 60}m ago"

                # Get company color for the badge
                badge_color = get_company_color(news.symbol)
                ticker_name = ticker_names.get(news.symbol, news.symbol)

                st.markdown(
                    f"""
                    <div style='padding: 12px; margin: 8px 0; background-color: {CHART_FACE_COLOR}; border-radius: 8px; border-left: 4px solid {badge_color};'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                            <span style='background-color: {badge_color}; padding: 3px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; color: white;'>{news.symbol} - {ticker_name[:20]}</span>
                            <span style='color: #888; font-size: 0.8em;'>{time_str} • {news.publisher}</span>
                        </div>
                        <a href='{news.link}' target='_blank' style='color: #4da6ff; text-decoration: none; font-weight: 500; font-size: 1.05em; display: block;'>
                            {news.title} ↗
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No recent news found for your watchlist stocks")
    else:
        st.info("Your watchlist is empty. Add stocks you're watching above.")


# ═════════════════════════════════════════════
# TAB 2: Options Trading (Strategy + Chain + Greeks)
# ═════════════════════════════════════════════

with tab_options:
    if info and spot:
        opt_title_col, opt_price_col = st.columns([3, 1])
        with opt_title_col:
            st.subheader(f"Options Trading — {info['name']} ({symbol})")
        with opt_price_col:
            st.metric("Spot Price", f"${spot:,.2f}")
    else:
        st.subheader("Options Trading")

    # ── Options Settings (moved from sidebar) ──
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
        show_styled_empty_state(
            icon="⚡",
            title="Options Trading",
            message="Enter a valid ticker symbol above to load the options chain, IV analysis, and strategy recommendations.",
        )
        show_skeleton(num_cards=4, height=50)
    elif info and symbol:
        # Load IV and Entropy lazily — only when Options tab is rendered
        with st.spinner("Loading IV & market data..."):
            try:
                iv_data = _cached_iv_percentile(symbol)
            except Exception:
                iv_data = None
            try:
                entropy_signal = _cached_entropy()
            except Exception:
                entropy_signal = None
            if iv_data:
                iv_percentile = iv_data["iv_percentile"]

    st.markdown("---")

    # ── IV Percentile Banner ──
    if iv_data:
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

    # ── Entropy Signal ──
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

    if not expirations:
        show_styled_empty_state(
            icon="📅",
            title="No Expirations Available",
            message="Could not load options expirations. Check the ticker symbol or try again in a moment.",
        )
        selected_exp = None
        selected_idx = None
    else:
        # ── Expiration Selector ──
        selected_idx = st.selectbox(
            "Expiration Date",
            range(len(expirations)),
            format_func=lambda i: exp_with_dte[i],
            index=min(2, len(expirations) - 1),
            key="options_exp_selector"
        )
        selected_exp = expirations[selected_idx]

    if selected_exp:
        selected_dte = (datetime.strptime(selected_exp, "%Y-%m-%d") - datetime.now()).days

        # ── Fetch Chain (cached, reuse spot to avoid extra API call) ──
        chain_df = None
        with st.spinner(f"Loading options chain for {selected_exp}..."):
            try:
                chain_df, _ = _cached_enriched_chain(symbol, selected_exp, risk_free_rate, spot)
            except Exception as e:
                st.error(f"Could not load options chain: {e}")

        if chain_df is None or chain_df.empty:
            st.warning("Options chain data unavailable. Yahoo Finance may be rate-limiting. Try again in a minute.")

        st.markdown("---")

        # Create subtabs for options trading
        options_subtab1, options_subtab2, options_subtab3 = st.tabs([
            "Strategy Recommender",
            "Options Chain",
            "Greeks & IV"
        ])

        with options_subtab1:
            st.markdown("#### Recommended Strategies")

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

                # Compute delta-based POP from the chain
                pop = _compute_pop_from_delta(strat.name, rec.concrete_legs, chain_df)

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
                    estimated_pop=pop,
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
                        pop_source = "delta-derived" if pop != 0.50 else "default"
                        if card.kelly_fraction > 0:
                            st.caption(
                                f"POP: {pop:.0%} ({pop_source}) | "
                                f"Half-Kelly suggests {card.kelly_contracts} contracts "
                                f"({card.kelly_fraction:.1%} of portfolio). "
                                f"Risk-rule sizing: {card.recommended_contracts} contracts."
                            )
                        elif card.kelly_fraction < 0:
                            st.caption(
                                f"POP: {pop:.0%} ({pop_source}) | "
                                f"Kelly criterion is negative ({card.kelly_fraction:.3f}) — "
                                f"risk/reward may not justify this trade."
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
        # SUBTAB 2: Options Chain Explorer
        # ═════════════════════════════════════════════

        with options_subtab2:
            st.markdown("#### Options Chain Explorer")

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
        # SUBTAB 3: Greeks & IV
        # ═════════════════════════════════════════════

        with options_subtab3:
            st.markdown("#### Greeks Across Strikes")

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
# TAB: Valuation & Analysis
# ═════════════════════════════════════════════

with tab_valuation:
    st.subheader("Stock Valuation & Fundamental Analysis")
    st.caption("DCF, P/E, EV/EBITDA valuations with comprehensive fundamental metrics")

    # Stock input
    with st.form("valuation_form"):
        val_col1, val_col2 = st.columns([2, 1])
        with val_col1:
            val_symbol = st.text_input("Enter Ticker Symbol", value="AAPL", key="val_ticker").upper().strip()
        with val_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_valuation = st.form_submit_button("Run Valuation Analysis", type="primary")

    @st.cache_data(ttl=600, show_spinner=False)
    def _fetch_valuation_data(sym):
        """Fetch all valuation data for a ticker. Cached 10 min to avoid rate limits.
        Uses retry logic and splits calls to reduce rate limit risk."""
        import yfinance as yf
        import time as _time

        t = yf.Ticker(sym)

        # Step 1: Get info with retry
        val_info = {}
        for attempt in range(3):
            try:
                val_info = t.info
                if val_info and val_info.get('currentPrice'):
                    break
            except Exception:
                pass
            _time.sleep(1.5 * (attempt + 1))  # 1.5s, 3s, 4.5s backoff

        # If info failed, try to build from fast_info
        if not val_info or not val_info.get('currentPrice'):
            try:
                fi = t.fast_info
                val_info = {
                    'currentPrice': fi.get('lastPrice'),
                    'marketCap': fi.get('marketCap'),
                    'sharesOutstanding': fi.get('shares'),
                    'previousClose': fi.get('previousClose'),
                    'fiftyDayAverage': fi.get('fiftyDayAverage'),
                    'twoHundredDayAverage': fi.get('twoHundredDayAverage'),
                }
            except Exception:
                pass

        _time.sleep(0.5)

        # Step 2: Get financial statements (these use a different endpoint, less rate-limited)
        financials = balance_sheet = cashflow = None
        q_fin = q_bs = q_cf = None
        try:
            financials = t.financials
            balance_sheet = t.balance_sheet
            cashflow = t.cashflow
        except Exception:
            pass

        _time.sleep(0.5)

        try:
            q_fin = t.quarterly_financials
            q_bs = t.quarterly_balance_sheet
            q_cf = t.quarterly_cashflow
        except Exception:
            pass

        return {
            "info": val_info or {},
            "financials": financials if financials is not None else pd.DataFrame(),
            "balance_sheet": balance_sheet if balance_sheet is not None else pd.DataFrame(),
            "cashflow": cashflow if cashflow is not None else pd.DataFrame(),
            "quarterly_financials": q_fin if q_fin is not None else pd.DataFrame(),
            "quarterly_balance_sheet": q_bs if q_bs is not None else pd.DataFrame(),
            "quarterly_cashflow": q_cf if q_cf is not None else pd.DataFrame(),
        }

    if val_symbol and run_valuation:
        # Show loading animation
        loading_placeholder = st.empty()
        loading_placeholder.markdown(show_dollar_spinner(f"Analyzing {val_symbol}..."), unsafe_allow_html=True)

        try:
            val_data = _fetch_valuation_data(val_symbol)
            val_info = val_data["info"]
            financials = val_data["financials"]
            balance_sheet = val_data["balance_sheet"]
            cash_flow = val_data["cashflow"]
            quarterly_financials = val_data["quarterly_financials"]
            quarterly_balance_sheet = val_data["quarterly_balance_sheet"]
            quarterly_cashflow = val_data["quarterly_cashflow"]

            # Save to fallback cache on success
            if val_info.get('currentPrice'):
                save_fallback(f"valuation_{val_symbol}", val_data)

            loading_placeholder.empty()

            if not val_info.get('currentPrice'):
                # Suggest common corrections
                common_typos = {"APPL": "AAPL", "GOGL": "GOOGL", "AMZN ": "AMZN", "TSLA ": "TSLA", "MSFT ": "MSFT"}
                suggestion = common_typos.get(val_symbol, "")
                # Also try search_tickers for a suggestion
                if not suggestion:
                    try:
                        matches = search_tickers(val_symbol)
                        if matches:
                            suggestion = matches[0]["symbol"]
                    except Exception:
                        pass
                if suggestion:
                    st.error(f"Could not find data for **{val_symbol}**. Did you mean **{suggestion}**?")
                else:
                    st.error(f"Could not find data for **{val_symbol}**. Check the ticker symbol and try again.")
            else:
                # Extract key data
                current_price = val_info.get('currentPrice', 0)
                market_cap = val_info.get('marketCap', 0)
                enterprise_value = val_info.get('enterpriseValue', 0)
                shares_outstanding = val_info.get('sharesOutstanding', 0)

                # Valuation metrics
                pe_ratio = val_info.get('trailingPE', 0) or 0
                forward_pe = val_info.get('forwardPE', 0) or 0
                peg_ratio = val_info.get('pegRatio', 0) or 0
                pb_ratio = val_info.get('priceToBook', 0) or 0
                ps_ratio = val_info.get('priceToSalesTrailing12Months', 0) or 0
                ev_ebitda = val_info.get('enterpriseToEbitda', 0) or 0
                ev_revenue = val_info.get('enterpriseToRevenue', 0) or 0

                # Profitability
                profit_margin = val_info.get('profitMargins', 0) or 0
                operating_margin = val_info.get('operatingMargins', 0) or 0
                gross_margin = val_info.get('grossMargins', 0) or 0
                roe = val_info.get('returnOnEquity', 0) or 0
                roa = val_info.get('returnOnAssets', 0) or 0

                # Growth
                revenue_growth = val_info.get('revenueGrowth', 0) or 0
                earnings_growth = val_info.get('earningsGrowth', 0) or 0

                # Financial health
                current_ratio = val_info.get('currentRatio', 0) or 0
                debt_to_equity = val_info.get('debtToEquity', 0) or 0
                total_debt = val_info.get('totalDebt', 0) or 0
                total_cash = val_info.get('totalCash', 0) or 0
                free_cash_flow = val_info.get('freeCashflow', 0) or 0

                # EPS data
                trailing_eps = val_info.get('trailingEps', 0) or 0
                forward_eps = val_info.get('forwardEps', 0) or 0

                # EBITDA
                ebitda = val_info.get('ebitda', 0) or 0
                total_revenue = val_info.get('totalRevenue', 0) or 0

                # Company info
                company_name = val_info.get('shortName', val_symbol)
                sector = val_info.get('sector', 'N/A')
                industry = val_info.get('industry', 'N/A')

                # ═══════════════════════════════════════════
                # HEADER - Company Overview
                # ═══════════════════════════════════════════
                st.markdown(f"## {company_name} ({val_symbol})")
                st.markdown(f"**Sector:** {sector} | **Industry:** {industry}")

                overview_cols = st.columns(5)
                with overview_cols[0]:
                    st.metric("Current Price", f"${current_price:,.2f}")
                with overview_cols[1]:
                    st.metric("Market Cap", f"${market_cap/1e9:,.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:,.0f}M")
                with overview_cols[2]:
                    st.metric("P/E Ratio", f"{pe_ratio:.1f}" if pe_ratio else "N/A")
                with overview_cols[3]:
                    st.metric("EV/EBITDA", f"{ev_ebitda:.1f}" if ev_ebitda else "N/A")
                with overview_cols[4]:
                    div_yield = val_info.get('trailingAnnualDividendYield', 0) or 0
                    st.metric("Div Yield", f"{div_yield*100:.2f}%")

                st.markdown("---")

                # ═══════════════════════════════════════════
                # VALUATION METHODS
                # ═══════════════════════════════════════════
                val_tab0, val_tab1, val_tab2, val_tab3, val_tab4 = st.tabs([
                    "📄 Financial Statements",
                    "📊 DCF Valuation",
                    "📈 Relative Valuation",
                    "📉 Fundamental Analysis",
                    "💰 Financial Health"
                ])

                # ── FINANCIAL STATEMENTS (10-K / Yahoo Finance style) ──
                with val_tab0:
                    st.markdown("### Financial Statements")

                    # Detect fiscal year end
                    fiscal_year_end = val_info.get('lastFiscalYearEnd', None)
                    fy_month_name = ""
                    if fiscal_year_end:
                        try:
                            from datetime import datetime as dt_cls
                            fy_date = dt_cls.fromtimestamp(fiscal_year_end)
                            fy_month_name = fy_date.strftime('%B')
                        except Exception:
                            pass
                    if fy_month_name:
                        st.caption(f"{company_name} fiscal year ends in **{fy_month_name}**. Data from SEC 10-K/10-Q filings via Yahoo Finance.")
                    else:
                        st.caption(f"Annual and quarterly data from {company_name}'s SEC filings via Yahoo Finance")

                    has_statements = (
                        financials is not None and not financials.empty and
                        balance_sheet is not None and not balance_sheet.empty and
                        cash_flow is not None and not cash_flow.empty
                    )

                    if not has_statements:
                        st.warning("Financial statements not available for this ticker.")
                    else:
                        import re as re_module

                        def fmt_stmt(df, is_annual=True):
                            """Format a financial statement: FY/quarter labels, readable values."""
                            display = df.copy()
                            new_cols = []
                            for col in display.columns:
                                if hasattr(col, 'strftime'):
                                    if is_annual:
                                        new_cols.append(f"FY {col.strftime('%Y')} ({col.strftime('%b')})")
                                    else:
                                        q = (col.month - 1) // 3 + 1
                                        new_cols.append(f"Q{q} {col.strftime('%Y')} ({col.strftime('%b')})")
                                else:
                                    new_cols.append(str(col))
                            display.columns = new_cols
                            for c in display.columns:
                                display[c] = display[c].apply(
                                    lambda x: f"${x/1e9:,.2f}B" if pd.notna(x) and isinstance(x, (int, float)) and abs(x) >= 1e9
                                    else (f"${x/1e6:,.0f}M" if pd.notna(x) and isinstance(x, (int, float)) and abs(x) >= 1e6
                                    else (f"${x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) and abs(x) >= 1
                                    else ("—" if not pd.notna(x) else f"{x}")))
                                )
                            display.index = [re_module.sub(r'(?<=[a-z])(?=[A-Z])', ' ', str(idx)) for idx in display.index]
                            return display

                        period_toggle = st.radio("Period", ["Annual", "Quarterly"], horizontal=True, key="stmt_period")
                        is_annual = period_toggle == "Annual"

                        if fy_month_name and fy_month_name != "December" and is_annual:
                            st.info(f"{company_name}'s fiscal year ends in **{fy_month_name}**, not December. Dates reflect the fiscal year-end period from the company's 10-K filing.")

                        stmt_tab_is, stmt_tab_bs, stmt_tab_cf = st.tabs(["Income Statement (P&L)", "Balance Sheet", "Cash Flow Statement"])

                        with stmt_tab_is:
                            is_data = financials if is_annual else quarterly_financials
                            if is_data is not None and not is_data.empty:
                                st.markdown(f"#### Income Statement \u2014 {period_toggle}")
                                st.dataframe(fmt_stmt(is_data, is_annual), use_container_width=True, height=600)
                            else:
                                st.info(f"No {period_toggle.lower()} income statement data available.")

                        with stmt_tab_bs:
                            bs_data = balance_sheet if is_annual else quarterly_balance_sheet
                            if bs_data is not None and not bs_data.empty:
                                st.markdown(f"#### Balance Sheet \u2014 {period_toggle}")
                                st.dataframe(fmt_stmt(bs_data, is_annual), use_container_width=True, height=600)
                            else:
                                st.info(f"No {period_toggle.lower()} balance sheet data available.")

                        with stmt_tab_cf:
                            cf_data = cash_flow if is_annual else quarterly_cashflow
                            if cf_data is not None and not cf_data.empty:
                                st.markdown(f"#### Cash Flow Statement \u2014 {period_toggle}")
                                st.dataframe(fmt_stmt(cf_data, is_annual), use_container_width=True, height=600)
                            else:
                                st.info(f"No {period_toggle.lower()} cash flow data available.")


                # ── DCF VALUATION ──
                with val_tab1:
                    st.markdown("### Discounted Cash Flow (DCF) Valuation")
                    st.caption("Fully derived from financial statements — no manual assumptions")

                    # Use annual statements for DCF
                    income_stmt = financials
                    bal_sheet = balance_sheet
                    cf_stmt = cash_flow

                    if not has_statements:
                        st.warning("Financial statements not available. DCF requires income statement, balance sheet, and cash flow data.")
                    else:
                        # ═══════════════════════════════════════════
                        # EXTRACT HISTORICAL FCF FROM CASH FLOW STMT
                        # ═══════════════════════════════════════════
                        # Get historical Free Cash Flow
                        hist_fcf = []
                        fcf_label = None
                        for label in ['Free Cash Flow', 'FreeCashFlow']:
                            if label in cf_stmt.index:
                                fcf_label = label
                                break

                        if fcf_label:
                            for col in reversed(cf_stmt.columns):  # oldest to newest
                                val = cf_stmt.loc[fcf_label, col]
                                if pd.notna(val):
                                    hist_fcf.append(float(val))

                        # Fallback: Operating Cash Flow - CapEx
                        if len(hist_fcf) < 2:
                            hist_fcf = []
                            op_cf_label = None
                            capex_label = None
                            for label in ['Operating Cash Flow', 'OperatingCashFlow', 'Total Cash From Operating Activities']:
                                if label in cf_stmt.index:
                                    op_cf_label = label
                                    break
                            for label in ['Capital Expenditure', 'CapitalExpenditure', 'Capital Expenditures']:
                                if label in cf_stmt.index:
                                    capex_label = label
                                    break
                            if op_cf_label and capex_label:
                                for col in reversed(cf_stmt.columns):
                                    op_val = cf_stmt.loc[op_cf_label, col]
                                    cap_val = cf_stmt.loc[capex_label, col]
                                    if pd.notna(op_val) and pd.notna(cap_val):
                                        hist_fcf.append(float(op_val) + float(cap_val))  # capex is negative

                        # Extract historical revenue from income statement
                        hist_revenue = []
                        rev_label = None
                        for label in ['Total Revenue', 'TotalRevenue', 'Revenue']:
                            if label in income_stmt.index:
                                rev_label = label
                                break
                        if rev_label:
                            for col in reversed(income_stmt.columns):
                                val = income_stmt.loc[rev_label, col]
                                if pd.notna(val):
                                    hist_revenue.append(float(val))

                        # Extract historical EBITDA
                        hist_ebitda = []
                        ebitda_label = None
                        for label in ['EBITDA', 'Ebitda']:
                            if label in income_stmt.index:
                                ebitda_label = label
                                break
                        if ebitda_label:
                            for col in reversed(income_stmt.columns):
                                val = income_stmt.loc[ebitda_label, col]
                                if pd.notna(val):
                                    hist_ebitda.append(float(val))

                        latest_fcf = hist_fcf[-1] if hist_fcf else free_cash_flow

                        if not latest_fcf or latest_fcf <= 0:
                            st.warning("DCF analysis requires positive free cash flow. This company has negative or zero FCF in the most recent year.")
                            if latest_fcf:
                                st.markdown(f"**Most Recent FCF:** ${latest_fcf/1e9:.2f}B")
                        else:
                            # ═══════════════════════════════════════════
                            # COMPUTE FCF CAGR FROM HISTORICAL DATA
                            # ═══════════════════════════════════════════
                            positive_fcf = [f for f in hist_fcf if f > 0]
                            if len(positive_fcf) >= 2:
                                n_years_hist = len(positive_fcf) - 1
                                fcf_cagr = (positive_fcf[-1] / positive_fcf[0]) ** (1 / n_years_hist) - 1
                            else:
                                fcf_cagr = revenue_growth if revenue_growth > 0 else 0.05

                            # Revenue CAGR
                            if len(hist_revenue) >= 2:
                                rev_cagr = (hist_revenue[-1] / hist_revenue[0]) ** (1 / (len(hist_revenue) - 1)) - 1
                            else:
                                rev_cagr = revenue_growth

                            # Clamp FCF growth to reasonable range
                            fcf_growth_rate = max(0.02, min(fcf_cagr, 0.30))
                            projection_years = 5

                            # ═══════════════════════════════════════════
                            # AUTO-CALCULATE WACC
                            # ═══════════════════════════════════════════
                            beta_val = val_info.get('beta', 1.0) or 1.0
                            risk_free = 0.043  # ~10Y Treasury yield
                            market_premium = 0.055  # Historical equity risk premium
                            cost_of_equity = risk_free + beta_val * market_premium

                            # Cost of Debt from income statement
                            interest_expense = 0
                            for ie_label in ['Interest Expense', 'InterestExpense']:
                                if ie_label in income_stmt.index:
                                    ie_val = income_stmt.loc[ie_label].iloc[0]
                                    if pd.notna(ie_val):
                                        interest_expense = abs(float(ie_val))
                                    break

                            cost_of_debt_pretax = (interest_expense / total_debt) if total_debt > 0 else 0.04
                            tax_rate_est = val_info.get('effectiveTaxRate', 0.21) or 0.21
                            if isinstance(tax_rate_est, (int, float)) and tax_rate_est > 1:
                                tax_rate_est = tax_rate_est / 100
                            cost_of_debt = cost_of_debt_pretax * (1 - tax_rate_est)

                            # Capital structure weights
                            equity_value_market = market_cap if market_cap > 0 else 1
                            total_capital = equity_value_market + total_debt
                            weight_equity = equity_value_market / total_capital if total_capital > 0 else 0.8
                            weight_debt = total_debt / total_capital if total_capital > 0 else 0.2

                            wacc_calculated = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt)
                            discount_rate = max(0.04, min(wacc_calculated, 0.25))

                            # Terminal growth = long-term GDP, capped below WACC
                            terminal_growth = min(0.025, discount_rate - 0.01)

                            # Exit multiple from current market data
                            current_ev_ebitda = ev_ebitda if ev_ebitda and ev_ebitda > 0 else 12.0

                            # ═══════════════════════════════════════════
                            # DISPLAY: WACC BREAKDOWN
                            # ═══════════════════════════════════════════
                            st.markdown("#### WACC — Derived from Financial Statements")
                            wacc_cols = st.columns(6)
                            with wacc_cols[0]:
                                st.metric("Beta", f"{beta_val:.2f}")
                            with wacc_cols[1]:
                                st.metric("Cost of Equity (CAPM)", f"{cost_of_equity:.1%}")
                            with wacc_cols[2]:
                                st.metric("Cost of Debt (after-tax)", f"{cost_of_debt:.1%}")
                            with wacc_cols[3]:
                                st.metric("Equity Weight", f"{weight_equity:.0%}")
                            with wacc_cols[4]:
                                st.metric("Debt Weight", f"{weight_debt:.0%}")
                            with wacc_cols[5]:
                                st.metric("**WACC**", f"{discount_rate:.2%}")
                            st.caption(f"Ke = {risk_free:.1%} + {beta_val:.2f} x {market_premium:.1%} = {cost_of_equity:.2%} | Kd = ${interest_expense/1e6:,.0f}M / ${total_debt/1e9:.1f}B x (1 - {tax_rate_est:.0%}) = {cost_of_debt:.2%} | WACC = {weight_equity:.0%} x {cost_of_equity:.2%} + {weight_debt:.0%} x {cost_of_debt:.2%} = **{discount_rate:.2%}**")

                            # ═══════════════════════════════════════════
                            # DISPLAY: GROWTH RATES FROM STATEMENTS
                            # ═══════════════════════════════════════════
                            st.markdown("---")
                            st.markdown("#### Growth Rates — Derived from Historical Data")
                            gr_cols = st.columns(4)
                            with gr_cols[0]:
                                st.metric("FCF CAGR", f"{fcf_cagr:.1%}", help=f"Compound annual growth of FCF over {len(positive_fcf)} years of data" if len(positive_fcf) >= 2 else "Estimated from revenue growth")
                            with gr_cols[1]:
                                st.metric("Revenue CAGR", f"{rev_cagr:.1%}" if len(hist_revenue) >= 2 else "N/A")
                            with gr_cols[2]:
                                st.metric("FCF Growth Used", f"{fcf_growth_rate:.1%}", help="FCF CAGR clamped to 2%-30%")
                            with gr_cols[3]:
                                st.metric("Terminal Growth", f"{terminal_growth:.1%}", help="Long-term GDP ~2.5%, capped below WACC")

                            # Show historical FCF trend
                            if len(hist_fcf) >= 2:
                                st.markdown("**Historical Free Cash Flow:**")
                                hist_years = list(range(len(hist_fcf)))
                                yr_labels = [f"Y-{len(hist_fcf)-1-i}" for i in range(len(hist_fcf)-1)] + ["Latest"]
                                hist_fcf_df = pd.DataFrame({
                                    "Year": yr_labels,
                                    "FCF ($B)": [f/1e9 for f in hist_fcf]
                                })
                                st.dataframe(hist_fcf_df.style.format({"FCF ($B)": "${:.2f}"}), hide_index=True)

                            # ═══════════════════════════════════════════
                            # DCF PROJECTION (5 years)
                            # ═══════════════════════════════════════════
                            st.markdown("---")
                            st.markdown("#### Terminal Value Method")
                            tv_method = st.radio("", ["Perpetuity Growth Model", "Exit Multiple (EV/EBITDA)"], horizontal=True, key="tv_method")

                            # Project FCF forward
                            projected_fcf = []
                            fcf_val = latest_fcf
                            for year in range(1, projection_years + 1):
                                fcf_val = fcf_val * (1 + fcf_growth_rate)
                                projected_fcf.append(fcf_val)

                            # Present value of projected FCF
                            pv_fcf = [f / ((1 + discount_rate) ** (i + 1)) for i, f in enumerate(projected_fcf)]

                            # Terminal value
                            if tv_method == "Perpetuity Growth Model":
                                terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
                                terminal_value = terminal_fcf / (discount_rate - terminal_growth)
                            else:
                                # Grow EBITDA forward at same rate
                                terminal_ebitda = hist_ebitda[-1] if hist_ebitda else (ebitda if ebitda and ebitda > 0 else projected_fcf[-1] * 1.5)
                                for _ in range(projection_years):
                                    terminal_ebitda = terminal_ebitda * (1 + fcf_growth_rate)
                                terminal_value = terminal_ebitda * current_ev_ebitda
                                terminal_fcf = projected_fcf[-1]

                            pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)

                            # Enterprise & equity value
                            dcf_enterprise_value = sum(pv_fcf) + pv_terminal
                            net_debt = total_debt - total_cash
                            equity_value = dcf_enterprise_value - net_debt
                            intrinsic_value = equity_value / shares_outstanding if shares_outstanding > 0 else 0
                            upside = ((intrinsic_value - current_price) / current_price) * 100 if current_price > 0 else 0

                            # ═══════════════════════════════════════════
                            # DISPLAY: TERMINAL VALUE BREAKDOWN
                            # ═══════════════════════════════════════════
                            st.markdown("---")
                            st.markdown("### DCF Results")

                            st.markdown("#### Terminal Value Breakdown")
                            if tv_method == "Perpetuity Growth Model":
                                tv_cols = st.columns(4)
                                with tv_cols[0]:
                                    st.metric("Last Projected FCF", f"${projected_fcf[-1]/1e9:.2f}B")
                                with tv_cols[1]:
                                    st.metric("Terminal FCF (FCFn x (1+g))", f"${terminal_fcf/1e9:.2f}B")
                                with tv_cols[2]:
                                    st.metric("Terminal Value", f"${terminal_value/1e9:.1f}B")
                                with tv_cols[3]:
                                    st.metric("PV of Terminal Value", f"${pv_terminal/1e9:.1f}B")
                                tv_pct = (pv_terminal / dcf_enterprise_value * 100) if dcf_enterprise_value > 0 else 0
                                st.caption(f"TV = ${projected_fcf[-1]/1e9:.2f}B x (1 + {terminal_growth:.1%}) / ({discount_rate:.2%} - {terminal_growth:.1%}) = **${terminal_value/1e9:.1f}B** | TV is **{tv_pct:.0f}%** of EV")
                            else:
                                tv_cols = st.columns(4)
                                with tv_cols[0]:
                                    st.metric("Terminal EBITDA", f"${terminal_ebitda/1e9:.2f}B")
                                with tv_cols[1]:
                                    st.metric("Exit Multiple", f"{current_ev_ebitda:.1f}x EV/EBITDA")
                                with tv_cols[2]:
                                    st.metric("Terminal Value", f"${terminal_value/1e9:.1f}B")
                                with tv_cols[3]:
                                    st.metric("PV of Terminal Value", f"${pv_terminal/1e9:.1f}B")
                                tv_pct = (pv_terminal / dcf_enterprise_value * 100) if dcf_enterprise_value > 0 else 0
                                st.caption(f"TV = ${terminal_ebitda/1e9:.2f}B x {current_ev_ebitda:.1f}x = **${terminal_value/1e9:.1f}B** | TV is **{tv_pct:.0f}%** of EV")

                            # Projected FCF table
                            fcf_df = pd.DataFrame({
                                "Year": [f"Year {i+1}" for i in range(projection_years)] + ["Terminal"],
                                "FCF ($B)": [f/1e9 for f in projected_fcf] + [terminal_fcf/1e9],
                                "PV ($B)": [f/1e9 for f in pv_fcf] + [pv_terminal/1e9]
                            })
                            st.dataframe(fcf_df.style.format({"FCF ($B)": "${:.2f}", "PV ($B)": "${:.2f}"}), hide_index=True)

                            # Results
                            result_cols = st.columns(4)
                            with result_cols[0]:
                                st.metric("Enterprise Value", f"${dcf_enterprise_value/1e9:.1f}B")
                            with result_cols[1]:
                                st.metric("Equity Value", f"${equity_value/1e9:.1f}B")
                            with result_cols[2]:
                                st.metric("Intrinsic Value/Share", f"${intrinsic_value:.2f}")
                            with result_cols[3]:
                                upside_color = "normal" if upside > 0 else "inverse"
                                st.metric("Upside/Downside", f"{upside:+.1f}%", delta_color=upside_color)

                            # Verdict
                            if upside > 20:
                                st.success(f"🟢 **UNDERVALUED** — The stock appears undervalued by {upside:.0f}% based on DCF analysis. Current price ${current_price:.2f} vs intrinsic value ${intrinsic_value:.2f}.")
                            elif upside > 0:
                                st.info(f"🟡 **FAIRLY VALUED** — The stock is slightly undervalued by {upside:.0f}%. Limited upside potential.")
                            elif upside > -20:
                                st.warning(f"🟠 **SLIGHTLY OVERVALUED** — The stock appears {abs(upside):.0f}% overvalued. Consider waiting for a better entry point.")
                            else:
                                st.error(f"🔴 **OVERVALUED** — The stock appears significantly overvalued by {abs(upside):.0f}%. DCF suggests intrinsic value of ${intrinsic_value:.2f}.")

                            # Sensitivity analysis
                            st.markdown("---")
                            st.markdown("### Sensitivity Analysis")
                            st.caption("How intrinsic value changes across WACC and growth rate scenarios")

                            growth_rates = [fcf_growth_rate - 0.03, fcf_growth_rate - 0.015, fcf_growth_rate, fcf_growth_rate + 0.015, fcf_growth_rate + 0.03]
                            discount_rates_sens = [discount_rate - 0.02, discount_rate - 0.01, discount_rate, discount_rate + 0.01, discount_rate + 0.02]

                            sensitivity_data = []
                            for gr in growth_rates:
                                row = {"FCF Growth": f"{gr*100:.1f}%"}
                                for dr in discount_rates_sens:
                                    temp_fcf_list = []
                                    temp_f = latest_fcf
                                    for _ in range(projection_years):
                                        temp_f = temp_f * (1 + gr)
                                        temp_fcf_list.append(temp_f)
                                    temp_pv = [f / ((1 + dr) ** (i + 1)) for i, f in enumerate(temp_fcf_list)]
                                    if tv_method == "Perpetuity Growth Model":
                                        temp_terminal = temp_fcf_list[-1] * (1 + terminal_growth) / (dr - terminal_growth) if dr > terminal_growth else 0
                                    else:
                                        temp_ebitda_s = hist_ebitda[-1] if hist_ebitda else (ebitda if ebitda and ebitda > 0 else temp_fcf_list[-1] * 1.5)
                                        for _ in range(projection_years):
                                            temp_ebitda_s = temp_ebitda_s * (1 + gr)
                                        temp_terminal = temp_ebitda_s * current_ev_ebitda
                                    temp_pv_terminal = temp_terminal / ((1 + dr) ** projection_years)
                                    temp_ev = sum(temp_pv) + temp_pv_terminal
                                    temp_equity = temp_ev - net_debt
                                    temp_iv = temp_equity / shares_outstanding if shares_outstanding > 0 else 0
                                    row[f"WACC {dr*100:.1f}%"] = f"${temp_iv:.0f}"
                                sensitivity_data.append(row)

                            sens_df = pd.DataFrame(sensitivity_data)
                            st.dataframe(sens_df, hide_index=True, use_container_width=True)

                # ── RELATIVE VALUATION ──
                with val_tab2:
                    st.markdown("### Relative Valuation")
                    st.caption("All multiples and implied values derived from available market data — no targets or assumptions")

                    # Current multiples from market data
                    st.markdown("#### Current Valuation Multiples")
                    mult_cols = st.columns(4)

                    with mult_cols[0]:
                        st.metric("P/E (TTM)", f"{pe_ratio:.1f}x" if pe_ratio else "N/A")
                        st.metric("Forward P/E", f"{forward_pe:.1f}x" if forward_pe else "N/A")

                    with mult_cols[1]:
                        st.metric("PEG Ratio", f"{peg_ratio:.2f}" if peg_ratio else "N/A")
                        st.metric("P/B Ratio", f"{pb_ratio:.1f}x" if pb_ratio else "N/A")

                    with mult_cols[2]:
                        st.metric("P/S Ratio", f"{ps_ratio:.1f}x" if ps_ratio else "N/A")
                        st.metric("EV/Revenue", f"{ev_revenue:.1f}x" if ev_revenue else "N/A")

                    with mult_cols[3]:
                        st.metric("EV/EBITDA", f"{ev_ebitda:.1f}x" if ev_ebitda else "N/A")
                        div_yield_val = val_info.get('trailingAnnualDividendYield', 0) or 0
                        st.metric("Dividend Yield", f"{div_yield_val*100:.2f}%" if div_yield_val else "N/A")

                    st.markdown("---")

                    # Implied valuations from actual data
                    st.markdown("#### Implied Valuations from Current Data")
                    st.caption("What the stock is worth based on each multiple applied to actual financials")

                    net_debt_rv = total_debt - total_cash

                    valuation_rows = []

                    # P/E implied: current P/E x EPS = current price (show Forward P/E implied)
                    if forward_pe and forward_pe > 0 and forward_eps and forward_eps > 0:
                        fwd_pe_implied = forward_eps * forward_pe
                        valuation_rows.append({"Method": "Forward P/E", "Multiple": f"{forward_pe:.1f}x", "Basis": f"Fwd EPS ${forward_eps:.2f}", "Implied Price": f"${fwd_pe_implied:.2f}", "vs Current": f"{((fwd_pe_implied/current_price)-1)*100:+.1f}%"})

                    if pe_ratio and pe_ratio > 0 and trailing_eps and trailing_eps > 0:
                        ttm_pe_implied = trailing_eps * pe_ratio
                        valuation_rows.append({"Method": "Trailing P/E", "Multiple": f"{pe_ratio:.1f}x", "Basis": f"TTM EPS ${trailing_eps:.2f}", "Implied Price": f"${ttm_pe_implied:.2f}", "vs Current": f"{((ttm_pe_implied/current_price)-1)*100:+.1f}%"})

                    # EV/EBITDA implied
                    if ev_ebitda and ev_ebitda > 0 and ebitda and ebitda > 0:
                        ev_implied = ebitda * ev_ebitda
                        equity_implied = ev_implied - net_debt_rv
                        ev_ebitda_price = equity_implied / shares_outstanding if shares_outstanding > 0 else 0
                        valuation_rows.append({"Method": "EV/EBITDA", "Multiple": f"{ev_ebitda:.1f}x", "Basis": f"EBITDA ${ebitda/1e9:.2f}B", "Implied Price": f"${ev_ebitda_price:.2f}", "vs Current": f"{((ev_ebitda_price/current_price)-1)*100:+.1f}%"})

                    # EV/Revenue implied
                    if ev_revenue and ev_revenue > 0 and total_revenue and total_revenue > 0:
                        ev_rev_implied = total_revenue * ev_revenue
                        equity_rev_implied = ev_rev_implied - net_debt_rv
                        ev_rev_price = equity_rev_implied / shares_outstanding if shares_outstanding > 0 else 0
                        valuation_rows.append({"Method": "EV/Revenue", "Multiple": f"{ev_revenue:.1f}x", "Basis": f"Revenue ${total_revenue/1e9:.2f}B", "Implied Price": f"${ev_rev_price:.2f}", "vs Current": f"{((ev_rev_price/current_price)-1)*100:+.1f}%"})

                    # P/B implied
                    book_value_per_share = val_info.get('bookValue', 0) or 0
                    if pb_ratio and pb_ratio > 0 and book_value_per_share and book_value_per_share > 0:
                        pb_implied = book_value_per_share * pb_ratio
                        valuation_rows.append({"Method": "P/B Ratio", "Multiple": f"{pb_ratio:.1f}x", "Basis": f"Book Value ${book_value_per_share:.2f}", "Implied Price": f"${pb_implied:.2f}", "vs Current": f"{((pb_implied/current_price)-1)*100:+.1f}%"})

                    # P/S implied
                    if ps_ratio and ps_ratio > 0 and total_revenue and total_revenue > 0 and shares_outstanding > 0:
                        rev_per_share = total_revenue / shares_outstanding
                        ps_implied = rev_per_share * ps_ratio
                        valuation_rows.append({"Method": "P/S Ratio", "Multiple": f"{ps_ratio:.1f}x", "Basis": f"Rev/Share ${rev_per_share:.2f}", "Implied Price": f"${ps_implied:.2f}", "vs Current": f"{((ps_implied/current_price)-1)*100:+.1f}%"})

                    if valuation_rows:
                        val_df = pd.DataFrame(valuation_rows)
                        st.dataframe(val_df, hide_index=True, use_container_width=True)

                        # Composite implied value (average of all methods)
                        implied_prices = []
                        for row in valuation_rows:
                            price_str = row["Implied Price"].replace("$", "").replace(",", "")
                            try:
                                implied_prices.append(float(price_str))
                            except ValueError:
                                pass
                        if implied_prices:
                            avg_implied = sum(implied_prices) / len(implied_prices)
                            avg_upside = ((avg_implied - current_price) / current_price) * 100

                            st.markdown("---")
                            st.markdown("#### Composite Valuation")
                            comp_cols = st.columns(3)
                            with comp_cols[0]:
                                st.metric("Average Implied Price", f"${avg_implied:.2f}")
                            with comp_cols[1]:
                                st.metric("Current Price", f"${current_price:.2f}")
                            with comp_cols[2]:
                                st.metric("Avg Upside/Downside", f"{avg_upside:+.1f}%",
                                         delta_color="normal" if avg_upside > 0 else "inverse")
                            st.caption(f"Average of {len(implied_prices)} valuation methods based on actual market multiples")
                    else:
                        st.warning("Insufficient data to compute relative valuations.")

                # ── FUNDAMENTAL ANALYSIS ──
                with val_tab3:
                    st.markdown("### Fundamental Analysis")
                    st.caption("Profitability, growth, and efficiency metrics")

                    # Profitability
                    st.markdown("#### Profitability Metrics")
                    prof_cols = st.columns(4)

                    with prof_cols[0]:
                        margin_color = "normal" if gross_margin and gross_margin > 0.4 else "inverse" if gross_margin and gross_margin < 0.2 else "off"
                        st.metric("Gross Margin", f"{gross_margin*100:.1f}%" if gross_margin else "N/A",
                                 delta="Strong" if gross_margin and gross_margin > 0.4 else "Weak" if gross_margin and gross_margin < 0.2 else None,
                                 delta_color=margin_color)

                    with prof_cols[1]:
                        st.metric("Operating Margin", f"{operating_margin*100:.1f}%" if operating_margin else "N/A",
                                 delta="Strong" if operating_margin and operating_margin > 0.2 else None)

                    with prof_cols[2]:
                        st.metric("Profit Margin", f"{profit_margin*100:.1f}%" if profit_margin else "N/A")

                    with prof_cols[3]:
                        st.metric("EBITDA Margin", f"{(ebitda/total_revenue)*100:.1f}%" if ebitda and total_revenue else "N/A")

                    st.markdown("---")

                    # Returns
                    st.markdown("#### Return Metrics")
                    ret_cols = st.columns(3)

                    with ret_cols[0]:
                        roe_color = "normal" if roe and roe > 0.15 else "inverse" if roe and roe < 0.05 else "off"
                        st.metric("Return on Equity (ROE)", f"{roe*100:.1f}%" if roe else "N/A",
                                 delta="Excellent" if roe and roe > 0.2 else "Good" if roe and roe > 0.15 else None,
                                 delta_color=roe_color)
                        if roe:
                            if roe > 0.20:
                                st.caption("🟢 Exceptional profitability")
                            elif roe > 0.15:
                                st.caption("🟢 Strong profitability")
                            elif roe > 0.10:
                                st.caption("🟡 Average profitability")
                            else:
                                st.caption("🔴 Below-average profitability")

                    with ret_cols[1]:
                        st.metric("Return on Assets (ROA)", f"{roa*100:.1f}%" if roa else "N/A")
                        if roa:
                            if roa > 0.10:
                                st.caption("🟢 Efficient asset utilization")
                            elif roa > 0.05:
                                st.caption("🟡 Average efficiency")
                            else:
                                st.caption("🔴 Low asset efficiency")

                    with ret_cols[2]:
                        # Calculate ROIC estimate
                        if ebitda and enterprise_value and enterprise_value > 0:
                            roic_est = ebitda / enterprise_value
                            st.metric("ROIC (Est)", f"{roic_est*100:.1f}%")
                        else:
                            st.metric("ROIC", "N/A")

                    st.markdown("---")

                    # Growth
                    st.markdown("#### Growth Metrics")
                    growth_cols = st.columns(3)

                    with growth_cols[0]:
                        rev_color = "normal" if revenue_growth and revenue_growth > 0.1 else "inverse" if revenue_growth and revenue_growth < 0 else "off"
                        st.metric("Revenue Growth", f"{revenue_growth*100:.1f}%" if revenue_growth else "N/A",
                                 delta="Strong" if revenue_growth and revenue_growth > 0.15 else "Declining" if revenue_growth and revenue_growth < 0 else None,
                                 delta_color=rev_color)

                    with growth_cols[1]:
                        earn_color = "normal" if earnings_growth and earnings_growth > 0.1 else "inverse" if earnings_growth and earnings_growth < 0 else "off"
                        st.metric("Earnings Growth", f"{earnings_growth*100:.1f}%" if earnings_growth else "N/A",
                                 delta_color=earn_color)

                    with growth_cols[2]:
                        # 5-year revenue CAGR if available
                        five_yr_growth = val_info.get('revenueGrowth', 0)
                        st.metric("5Y Revenue CAGR", f"{five_yr_growth*100:.1f}%" if five_yr_growth else "N/A")

                    st.markdown("---")

                    # Analyst Ratings
                    st.markdown("#### Analyst Estimates")
                    target_high = val_info.get('targetHighPrice', 0)
                    target_low = val_info.get('targetLowPrice', 0)
                    target_mean = val_info.get('targetMeanPrice', 0)
                    recommendation = val_info.get('recommendationKey', 'N/A')

                    if target_mean:
                        analyst_cols = st.columns(4)
                        with analyst_cols[0]:
                            st.metric("Target Low", f"${target_low:.2f}" if target_low else "N/A")
                        with analyst_cols[1]:
                            upside_mean = ((target_mean - current_price) / current_price) * 100
                            st.metric("Target Mean", f"${target_mean:.2f}",
                                     delta=f"{upside_mean:+.0f}%",
                                     delta_color="normal" if upside_mean > 0 else "inverse")
                        with analyst_cols[2]:
                            st.metric("Target High", f"${target_high:.2f}" if target_high else "N/A")
                        with analyst_cols[3]:
                            rec_emoji = "🟢" if recommendation in ['buy', 'strongBuy'] else "🟡" if recommendation == 'hold' else "🔴"
                            st.metric("Recommendation", f"{rec_emoji} {recommendation.upper()}")

                # ── FINANCIAL HEALTH ──
                with val_tab4:
                    st.markdown("### Financial Health")
                    st.caption("Balance sheet strength and credit metrics")

                    # Liquidity
                    st.markdown("#### Liquidity")
                    liq_cols = st.columns(3)

                    with liq_cols[0]:
                        cr_color = "normal" if current_ratio and current_ratio > 1.5 else "inverse" if current_ratio and current_ratio < 1 else "off"
                        st.metric("Current Ratio", f"{current_ratio:.2f}" if current_ratio else "N/A",
                                 delta="Strong" if current_ratio and current_ratio > 2 else "Weak" if current_ratio and current_ratio < 1 else None,
                                 delta_color=cr_color)
                        if current_ratio:
                            if current_ratio > 2:
                                st.caption("🟢 Excellent liquidity")
                            elif current_ratio > 1:
                                st.caption("🟡 Adequate liquidity")
                            else:
                                st.caption("🔴 Liquidity concerns")

                    with liq_cols[1]:
                        st.metric("Total Cash", f"${total_cash/1e9:.1f}B" if total_cash else "N/A")

                    with liq_cols[2]:
                        st.metric("Free Cash Flow", f"${free_cash_flow/1e9:.2f}B" if free_cash_flow else "N/A",
                                 delta="Positive" if free_cash_flow and free_cash_flow > 0 else "Negative",
                                 delta_color="normal" if free_cash_flow and free_cash_flow > 0 else "inverse")

                    st.markdown("---")

                    # Leverage
                    st.markdown("#### Leverage & Solvency")
                    debt_cols = st.columns(3)

                    with debt_cols[0]:
                        de_color = "normal" if debt_to_equity and debt_to_equity < 100 else "inverse" if debt_to_equity and debt_to_equity > 200 else "off"
                        st.metric("Debt/Equity", f"{debt_to_equity:.0f}%" if debt_to_equity else "N/A",
                                 delta="High" if debt_to_equity and debt_to_equity > 150 else "Low" if debt_to_equity and debt_to_equity < 50 else None,
                                 delta_color=de_color)

                    with debt_cols[1]:
                        st.metric("Total Debt", f"${total_debt/1e9:.1f}B" if total_debt else "N/A")

                    with debt_cols[2]:
                        net_debt_val = total_debt - total_cash if total_debt and total_cash else 0
                        nd_color = "normal" if net_debt_val < 0 else "off"
                        st.metric("Net Debt", f"${net_debt_val/1e9:.1f}B" if net_debt_val else "N/A",
                                 delta="Net Cash" if net_debt_val < 0 else None,
                                 delta_color=nd_color)

                    # Interest coverage
                    interest_expense = val_info.get('interestExpense', 0) or 0
                    if ebitda and interest_expense and interest_expense > 0:
                        interest_coverage = ebitda / interest_expense
                        st.markdown("---")
                        st.markdown("#### Interest Coverage")
                        ic_color = "normal" if interest_coverage > 5 else "inverse" if interest_coverage < 2 else "off"
                        st.metric("Interest Coverage Ratio", f"{interest_coverage:.1f}x",
                                 delta="Strong" if interest_coverage > 5 else "Weak" if interest_coverage < 2 else None,
                                 delta_color=ic_color)
                        if interest_coverage > 5:
                            st.caption("🟢 Company easily covers interest payments")
                        elif interest_coverage > 2:
                            st.caption("🟡 Adequate coverage")
                        else:
                            st.caption("🔴 May struggle to cover interest payments")

                    # Credit Risk Summary
                    st.markdown("---")
                    st.markdown("#### Credit Risk Assessment")

                    risk_score = 50  # Base
                    risk_factors = []

                    if current_ratio and current_ratio < 1:
                        risk_score += 20
                        risk_factors.append("Low current ratio")
                    if debt_to_equity and debt_to_equity > 200:
                        risk_score += 20
                        risk_factors.append("High debt/equity")
                    if free_cash_flow and free_cash_flow < 0:
                        risk_score += 15
                        risk_factors.append("Negative free cash flow")
                    if interest_expense and ebitda:
                        if ebitda / interest_expense < 2:
                            risk_score += 15
                            risk_factors.append("Low interest coverage")

                    if current_ratio and current_ratio > 2:
                        risk_score -= 10
                    if debt_to_equity and debt_to_equity < 50:
                        risk_score -= 10
                    if free_cash_flow and free_cash_flow > 0:
                        risk_score -= 10

                    risk_score = max(0, min(100, risk_score))

                    if risk_score < 30:
                        st.success(f"🟢 **Low Credit Risk** (Score: {risk_score}/100) — Strong balance sheet with healthy liquidity and manageable debt.")
                    elif risk_score < 60:
                        st.info(f"🟡 **Moderate Credit Risk** (Score: {risk_score}/100) — Adequate financial health with some areas to monitor.")
                    else:
                        st.error(f"🔴 **Elevated Credit Risk** (Score: {risk_score}/100) — Balance sheet concerns. Risk factors: {', '.join(risk_factors)}")

        except Exception as e:
            loading_placeholder.empty()
            # Try fallback cache before showing error
            cached_val, age_min = load_fallback(f"valuation_{val_symbol}", max_age_hours=24)
            if cached_val:
                st.warning(f"Live data unavailable — showing cached analysis from ~{age_min} minutes ago.")
                val_info = cached_val["info"]
                financials = cached_val["financials"]
                balance_sheet = cached_val["balance_sheet"]
                cash_flow = cached_val["cashflow"]
                quarterly_financials = cached_val["quarterly_financials"]
                quarterly_balance_sheet = cached_val["quarterly_balance_sheet"]
                quarterly_cashflow = cached_val["quarterly_cashflow"]
            else:
                st.error(f"Error analyzing {val_symbol}: {str(e)}")

    elif not run_valuation:
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;padding:60px 20px;'>"
            "<h3 style='color:#6C63FF;'>Enter a ticker symbol and click Run Valuation Analysis</h3>"
            "<p style='color:#888;font-size:1.1em;'>Get financial statements, DCF valuation, relative valuation, and more</p>"
            "</div>",
            unsafe_allow_html=True
        )


# ═════════════════════════════════════════════
# TAB 2: Entropy Analysis
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
            st.metric("Baseline (Equal-Weight)", f"{entropy_signal.baseline_entropy:.3f} bits")
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
        weights_src = "Live Market Caps" if entropy_signal.using_live_caps else "ETF Performance Estimate"
        ax4.set_title(f"S&P 500 Sector Weights ({weights_src})", color='white')
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
    "RCR Portfolio Trackers v2.0 | Data: yfinance (15-min delayed) | "
    "Greeks: Black-Scholes (European approx.) | "
    "Entropy: Sector ETF-based Shannon entropy | "
    "Portfolio: Live prices & dividend yields | "
    "Not financial advice. For educational purposes only."
)
