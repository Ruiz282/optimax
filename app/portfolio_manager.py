"""
Portfolio Manager — Track stocks, bonds, ETFs with prices and dividend yields.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
import hashlib


# ─────────────────────────────────────────────
# Company Brand Colors
# ─────────────────────────────────────────────

COMPANY_COLORS = {
    # Tech Giants
    "AAPL": "#555555",      # Apple - Gray/Silver
    "MSFT": "#00A4EF",      # Microsoft - Blue
    "GOOGL": "#4285F4",     # Google - Blue
    "GOOG": "#4285F4",      # Google - Blue
    "AMZN": "#FF9900",      # Amazon - Orange
    "META": "#0668E1",      # Meta - Blue
    "NVDA": "#76B900",      # NVIDIA - Green
    "TSLA": "#CC0000",      # Tesla - Red
    "NFLX": "#E50914",      # Netflix - Red
    "ADBE": "#FF0000",      # Adobe - Red
    "CRM": "#00A1E0",       # Salesforce - Blue
    "ORCL": "#F80000",      # Oracle - Red
    "IBM": "#0530AD",       # IBM - Blue
    "INTC": "#0071C5",      # Intel - Blue
    "AMD": "#ED1C24",       # AMD - Red
    "CSCO": "#049FD9",      # Cisco - Blue
    "QCOM": "#3253DC",      # Qualcomm - Blue
    "TXN": "#CC0000",       # Texas Instruments - Red
    "AVGO": "#CC092F",      # Broadcom - Red
    "NOW": "#81B5A1",       # ServiceNow - Green
    "INTU": "#365EBF",      # Intuit - Blue
    "SNOW": "#29B5E8",      # Snowflake - Blue
    "PLTR": "#101010",      # Palantir - Black
    "CRWD": "#FF0000",      # CrowdStrike - Red
    "PANW": "#FA582D",      # Palo Alto - Orange
    "ZS": "#0090D4",        # Zscaler - Blue
    "DDOG": "#632CA6",      # Datadog - Purple
    "NET": "#F38020",       # Cloudflare - Orange
    "MDB": "#00ED64",       # MongoDB - Green
    "SHOP": "#96BF48",      # Shopify - Green

    # Financials
    "JPM": "#117ACA",       # JPMorgan - Blue
    "BAC": "#012169",       # Bank of America - Blue
    "WFC": "#D71E28",       # Wells Fargo - Red
    "GS": "#7399C6",        # Goldman Sachs - Blue
    "MS": "#002D72",        # Morgan Stanley - Blue
    "C": "#003B70",         # Citigroup - Blue
    "V": "#1A1F71",         # Visa - Blue
    "MA": "#FF5F00",        # Mastercard - Orange
    "AXP": "#006FCF",       # American Express - Blue
    "PYPL": "#003087",      # PayPal - Blue
    "SQ": "#000000",        # Block/Square - Black
    "SCHW": "#00A0DF",      # Schwab - Blue
    "BLK": "#000000",       # BlackRock - Black
    "COF": "#004879",       # Capital One - Blue

    # Healthcare
    "JNJ": "#D51900",       # J&J - Red
    "UNH": "#002677",       # UnitedHealth - Blue
    "PFE": "#0093D0",       # Pfizer - Blue
    "ABBV": "#071D49",      # AbbVie - Blue
    "MRK": "#009A9A",       # Merck - Teal
    "LLY": "#D52B1E",       # Eli Lilly - Red
    "TMO": "#003A70",       # Thermo Fisher - Blue
    "ABT": "#0072CE",       # Abbott - Blue
    "BMY": "#BE1E2D",       # Bristol-Myers - Red
    "AMGN": "#0063BE",      # Amgen - Blue
    "GILD": "#C8102E",      # Gilead - Red
    "MRNA": "#00857C",      # Moderna - Teal
    "CVS": "#CC0000",       # CVS - Red

    # Consumer
    "WMT": "#0071CE",       # Walmart - Blue
    "COST": "#E31837",      # Costco - Red
    "HD": "#F96302",        # Home Depot - Orange
    "LOW": "#004990",       # Lowe's - Blue
    "TGT": "#CC0000",       # Target - Red
    "NKE": "#111111",       # Nike - Black
    "SBUX": "#00704A",      # Starbucks - Green
    "MCD": "#FFC72C",       # McDonald's - Yellow/Gold
    "DIS": "#006E99",       # Disney - Blue
    "KO": "#F40009",        # Coca-Cola - Red
    "PEP": "#004B93",       # PepsiCo - Blue
    "PG": "#003DA5",        # P&G - Blue
    "CL": "#E21836",        # Colgate - Red
    "KHC": "#F40009",       # Kraft Heinz - Red

    # Industrials
    "CAT": "#FFCD11",       # Caterpillar - Yellow
    "DE": "#367C2B",        # John Deere - Green
    "BA": "#0033A0",        # Boeing - Blue
    "LMT": "#003366",       # Lockheed Martin - Blue
    "RTX": "#00205B",       # RTX - Blue
    "GE": "#3B73B9",        # GE - Blue
    "HON": "#D0103A",       # Honeywell - Red
    "UPS": "#351C15",       # UPS - Brown
    "FDX": "#4D148C",       # FedEx - Purple

    # Energy
    "XOM": "#ED1B2D",       # Exxon - Red
    "CVX": "#0054A6",       # Chevron - Blue
    "COP": "#E31837",       # ConocoPhillips - Red
    "SLB": "#0072CE",       # Schlumberger - Blue
    "OXY": "#EE3124",       # Occidental - Red

    # Communications
    "T": "#00A8E0",         # AT&T - Blue
    "VZ": "#CD040B",        # Verizon - Red
    "TMUS": "#E20074",      # T-Mobile - Magenta
    "CMCSA": "#000000",     # Comcast - Black

    # ETFs
    "SPY": "#003399",       # SPDR - Blue
    "VOO": "#96151D",       # Vanguard - Maroon
    "VTI": "#96151D",       # Vanguard - Maroon
    "QQQ": "#00AEB3",       # Invesco - Teal
    "IWM": "#000000",       # iShares - Black
    "SCHD": "#00A0DF",      # Schwab - Blue
    "VYM": "#96151D",       # Vanguard - Maroon
    "VIG": "#96151D",       # Vanguard - Maroon
    "BND": "#96151D",       # Vanguard - Maroon
    "AGG": "#000000",       # iShares - Black
    "TLT": "#000000",       # iShares - Black
    "VNQ": "#96151D",       # Vanguard - Maroon
    "XLK": "#003399",       # SPDR - Blue
    "XLF": "#003399",       # SPDR - Blue
    "XLE": "#003399",       # SPDR - Blue
    "XLV": "#003399",       # SPDR - Blue
    "ARKK": "#FFFFFF",      # ARK - White (use dark text)

    # REITs
    "O": "#003768",         # Realty Income - Blue
    "AMT": "#009FDA",       # American Tower - Blue
    "PLD": "#0056A3",       # Prologis - Blue
    "SPG": "#000000",       # Simon Property - Black
    "CCI": "#D31245",       # Crown Castle - Red

    # Crypto-related
    "COIN": "#0052FF",      # Coinbase - Blue
    "MSTR": "#D9232E",      # MicroStrategy - Red

    # International
    "BABA": "#FF6A00",      # Alibaba - Orange
    "TSM": "#CC0000",       # TSMC - Red
    "NVO": "#003DA5",       # Novo Nordisk - Blue
    "ASML": "#00A3E0",      # ASML - Blue
    "TM": "#EB0A1E",        # Toyota - Red
    "SONY": "#000000",      # Sony - Black
}


def get_company_color(symbol: str) -> str:
    """Get brand color for a company, or generate a consistent color if not in database."""
    if symbol in COMPANY_COLORS:
        return COMPANY_COLORS[symbol]
    # Generate a consistent color based on symbol hash
    hash_obj = hashlib.md5(symbol.encode())
    hash_hex = hash_obj.hexdigest()[:6]
    return f"#{hash_hex}"


@dataclass
class DividendEvent:
    """A dividend payment event."""
    symbol: str
    name: str
    ex_date: datetime
    pay_date: Optional[datetime]
    amount: float
    shares: float
    expected_income: float
    frequency: str  # "Monthly", "Quarterly", "Semi-Annual", "Annual"


@dataclass
class EarningsEvent:
    """An earnings report event."""
    symbol: str
    name: str
    earnings_date: datetime
    time_of_day: str  # "BMO" (Before Market Open), "AMC" (After Market Close), "Unknown"
    eps_estimate: Optional[float]
    revenue_estimate: Optional[float]
    is_confirmed: bool  # True if date is confirmed, False if estimated


@dataclass
class Holding:
    """A single portfolio holding."""
    symbol: str
    name: str
    asset_type: str  # "Stock", "ETF", "Bond ETF", "REIT"
    shares: float
    avg_cost: float
    current_price: float
    dividend_yield: float
    annual_dividend: float
    market_value: float
    total_cost: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    annual_income: float
    sector: Optional[str]
    beta: Optional[float]
    pe_ratio: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]


def classify_asset_type(info: dict) -> str:
    """Classify security type based on yfinance info."""
    quote_type = info.get("quoteType", "").upper()
    long_name = info.get("longName", "").lower()
    short_name = info.get("shortName", "").lower()

    if quote_type == "ETF":
        # Check if it's a bond ETF
        bond_keywords = ["bond", "treasury", "fixed income", "aggregate", "corporate bond"]
        if any(kw in long_name or kw in short_name for kw in bond_keywords):
            return "Bond ETF"
        # Check if it's a REIT
        if "reit" in long_name or "real estate" in long_name:
            return "REIT"
        return "ETF"
    elif quote_type == "EQUITY":
        # Check if it's a REIT
        industry = info.get("industry", "").lower()
        if "reit" in industry or "real estate" in long_name:
            return "REIT"
        return "Stock"
    else:
        return "Stock"


def fetch_security_data(symbol: str) -> Optional[Dict]:
    """
    Fetch comprehensive data for a security.
    Returns dict with price, dividend, and fundamental data.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get current price
        try:
            current_price = ticker.fast_info.get("lastPrice")
            if current_price is None:
                current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("navPrice")
        except:
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        if current_price is None:
            return None

        # Get dividend info
        dividend_yield = info.get("dividendYield") or info.get("yield") or 0
        if dividend_yield and dividend_yield > 1:  # Sometimes returned as percentage
            dividend_yield = dividend_yield / 100

        dividend_rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate") or 0

        # If we have yield but no rate, calculate it
        if dividend_yield > 0 and dividend_rate == 0:
            dividend_rate = current_price * dividend_yield
        # If we have rate but no yield, calculate it
        elif dividend_rate > 0 and dividend_yield == 0:
            dividend_yield = dividend_rate / current_price

        return {
            "symbol": symbol.upper(),
            "name": info.get("shortName") or info.get("longName") or symbol,
            "asset_type": classify_asset_type(info),
            "current_price": current_price,
            "dividend_yield": dividend_yield,
            "dividend_rate": dividend_rate,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "beta": info.get("beta"),
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
            "description": info.get("longBusinessSummary", "")[:200] + "..." if info.get("longBusinessSummary") else "",
        }
    except Exception as e:
        return None


def create_holding(symbol: str, shares: float, avg_cost: float) -> Optional[Holding]:
    """Create a Holding object with live data."""
    data = fetch_security_data(symbol)
    if data is None:
        return None

    market_value = shares * data["current_price"]
    total_cost = shares * avg_cost
    unrealized_pnl = market_value - total_cost
    unrealized_pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
    annual_income = shares * data["dividend_rate"]

    return Holding(
        symbol=data["symbol"],
        name=data["name"],
        asset_type=data["asset_type"],
        shares=shares,
        avg_cost=avg_cost,
        current_price=data["current_price"],
        dividend_yield=data["dividend_yield"],
        annual_dividend=data["dividend_rate"],
        market_value=market_value,
        total_cost=total_cost,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        annual_income=annual_income,
        sector=data["sector"],
        beta=data["beta"],
        pe_ratio=data["pe_ratio"],
        fifty_two_week_high=data["fifty_two_week_high"],
        fifty_two_week_low=data["fifty_two_week_low"],
    )


def calculate_portfolio_summary(holdings: List[Holding]) -> Dict:
    """Calculate portfolio-level metrics."""
    if not holdings:
        return {
            "total_value": 0,
            "total_cost": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "annual_income": 0,
            "portfolio_yield": 0,
            "weighted_beta": 0,
            "asset_allocation": {},
            "sector_allocation": {},
        }

    total_value = sum(h.market_value for h in holdings)
    total_cost = sum(h.total_cost for h in holdings)
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    annual_income = sum(h.annual_income for h in holdings)
    portfolio_yield = (annual_income / total_value * 100) if total_value > 0 else 0

    # Weighted beta
    betas = [(h.beta or 1.0, h.market_value) for h in holdings]
    weighted_beta = sum(b * v for b, v in betas) / total_value if total_value > 0 else 1.0

    # Asset allocation
    asset_allocation = {}
    for h in holdings:
        asset_allocation[h.asset_type] = asset_allocation.get(h.asset_type, 0) + h.market_value
    for k in asset_allocation:
        asset_allocation[k] = asset_allocation[k] / total_value * 100 if total_value > 0 else 0

    # Sector allocation (for stocks only)
    sector_allocation = {}
    stock_value = sum(h.market_value for h in holdings if h.asset_type == "Stock")
    for h in holdings:
        if h.asset_type == "Stock" and h.sector:
            sector_allocation[h.sector] = sector_allocation.get(h.sector, 0) + h.market_value
    for k in sector_allocation:
        sector_allocation[k] = sector_allocation[k] / stock_value * 100 if stock_value > 0 else 0

    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "annual_income": annual_income,
        "portfolio_yield": portfolio_yield,
        "weighted_beta": weighted_beta,
        "asset_allocation": asset_allocation,
        "sector_allocation": sector_allocation,
    }


# ─────────────────────────────────────────────
# Popular Securities for Quick Add
# ─────────────────────────────────────────────

POPULAR_STOCKS = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft"),
    ("GOOGL", "Alphabet"),
    ("AMZN", "Amazon"),
    ("NVDA", "NVIDIA"),
    ("META", "Meta Platforms"),
    ("TSLA", "Tesla"),
    ("JPM", "JPMorgan Chase"),
    ("V", "Visa"),
    ("JNJ", "Johnson & Johnson"),
    ("UNH", "UnitedHealth"),
    ("HD", "Home Depot"),
    ("PG", "Procter & Gamble"),
    ("MA", "Mastercard"),
    ("DIS", "Disney"),
]

# ─────────────────────────────────────────────
# Comprehensive Ticker Database for Autocomplete
# ─────────────────────────────────────────────

TICKER_DATABASE = [
    # Mega Cap Tech
    ("AAPL", "Apple Inc.", "Stock"),
    ("MSFT", "Microsoft Corporation", "Stock"),
    ("GOOGL", "Alphabet Inc. Class A", "Stock"),
    ("GOOG", "Alphabet Inc. Class C", "Stock"),
    ("AMZN", "Amazon.com Inc.", "Stock"),
    ("NVDA", "NVIDIA Corporation", "Stock"),
    ("META", "Meta Platforms Inc.", "Stock"),
    ("TSLA", "Tesla Inc.", "Stock"),
    ("AVGO", "Broadcom Inc.", "Stock"),
    ("ORCL", "Oracle Corporation", "Stock"),
    ("ADBE", "Adobe Inc.", "Stock"),
    ("CRM", "Salesforce Inc.", "Stock"),
    ("CSCO", "Cisco Systems Inc.", "Stock"),
    ("ACN", "Accenture plc", "Stock"),
    ("IBM", "International Business Machines", "Stock"),
    ("INTC", "Intel Corporation", "Stock"),
    ("AMD", "Advanced Micro Devices", "Stock"),
    ("QCOM", "Qualcomm Inc.", "Stock"),
    ("TXN", "Texas Instruments", "Stock"),
    ("NOW", "ServiceNow Inc.", "Stock"),
    ("INTU", "Intuit Inc.", "Stock"),
    ("AMAT", "Applied Materials", "Stock"),
    ("MU", "Micron Technology", "Stock"),
    ("LRCX", "Lam Research", "Stock"),
    ("KLAC", "KLA Corporation", "Stock"),
    ("SNPS", "Synopsys Inc.", "Stock"),
    ("CDNS", "Cadence Design Systems", "Stock"),
    ("PANW", "Palo Alto Networks", "Stock"),
    ("CRWD", "CrowdStrike Holdings", "Stock"),
    ("SNOW", "Snowflake Inc.", "Stock"),
    ("PLTR", "Palantir Technologies", "Stock"),
    ("NET", "Cloudflare Inc.", "Stock"),
    ("DDOG", "Datadog Inc.", "Stock"),
    ("ZS", "Zscaler Inc.", "Stock"),
    ("TEAM", "Atlassian Corporation", "Stock"),
    ("WDAY", "Workday Inc.", "Stock"),
    ("SPLK", "Splunk Inc.", "Stock"),
    ("FTNT", "Fortinet Inc.", "Stock"),

    # Financials
    ("JPM", "JPMorgan Chase & Co.", "Stock"),
    ("BAC", "Bank of America Corp.", "Stock"),
    ("WFC", "Wells Fargo & Co.", "Stock"),
    ("C", "Citigroup Inc.", "Stock"),
    ("GS", "Goldman Sachs Group", "Stock"),
    ("MS", "Morgan Stanley", "Stock"),
    ("BLK", "BlackRock Inc.", "Stock"),
    ("SCHW", "Charles Schwab Corp.", "Stock"),
    ("AXP", "American Express Co.", "Stock"),
    ("V", "Visa Inc.", "Stock"),
    ("MA", "Mastercard Inc.", "Stock"),
    ("PYPL", "PayPal Holdings Inc.", "Stock"),
    ("SQ", "Block Inc.", "Stock"),
    ("COF", "Capital One Financial", "Stock"),
    ("USB", "U.S. Bancorp", "Stock"),
    ("PNC", "PNC Financial Services", "Stock"),
    ("TFC", "Truist Financial Corp.", "Stock"),
    ("BK", "Bank of New York Mellon", "Stock"),
    ("AIG", "American International Group", "Stock"),
    ("MET", "MetLife Inc.", "Stock"),
    ("PRU", "Prudential Financial", "Stock"),
    ("ALL", "Allstate Corporation", "Stock"),
    ("TRV", "Travelers Companies", "Stock"),
    ("CB", "Chubb Limited", "Stock"),
    ("AFL", "Aflac Inc.", "Stock"),
    ("ICE", "Intercontinental Exchange", "Stock"),
    ("CME", "CME Group Inc.", "Stock"),
    ("SPGI", "S&P Global Inc.", "Stock"),
    ("MCO", "Moody's Corporation", "Stock"),
    ("MSCI", "MSCI Inc.", "Stock"),

    # Healthcare
    ("UNH", "UnitedHealth Group", "Stock"),
    ("JNJ", "Johnson & Johnson", "Stock"),
    ("LLY", "Eli Lilly and Company", "Stock"),
    ("PFE", "Pfizer Inc.", "Stock"),
    ("ABBV", "AbbVie Inc.", "Stock"),
    ("MRK", "Merck & Co. Inc.", "Stock"),
    ("TMO", "Thermo Fisher Scientific", "Stock"),
    ("ABT", "Abbott Laboratories", "Stock"),
    ("DHR", "Danaher Corporation", "Stock"),
    ("BMY", "Bristol-Myers Squibb", "Stock"),
    ("AMGN", "Amgen Inc.", "Stock"),
    ("GILD", "Gilead Sciences Inc.", "Stock"),
    ("ISRG", "Intuitive Surgical", "Stock"),
    ("VRTX", "Vertex Pharmaceuticals", "Stock"),
    ("REGN", "Regeneron Pharmaceuticals", "Stock"),
    ("MDT", "Medtronic plc", "Stock"),
    ("SYK", "Stryker Corporation", "Stock"),
    ("BSX", "Boston Scientific", "Stock"),
    ("ZBH", "Zimmer Biomet Holdings", "Stock"),
    ("EW", "Edwards Lifesciences", "Stock"),
    ("CI", "Cigna Group", "Stock"),
    ("ELV", "Elevance Health", "Stock"),
    ("HUM", "Humana Inc.", "Stock"),
    ("CVS", "CVS Health Corporation", "Stock"),
    ("MCK", "McKesson Corporation", "Stock"),
    ("CAH", "Cardinal Health Inc.", "Stock"),
    ("MRNA", "Moderna Inc.", "Stock"),
    ("BIIB", "Biogen Inc.", "Stock"),

    # Consumer
    ("WMT", "Walmart Inc.", "Stock"),
    ("COST", "Costco Wholesale", "Stock"),
    ("HD", "Home Depot Inc.", "Stock"),
    ("LOW", "Lowe's Companies", "Stock"),
    ("TGT", "Target Corporation", "Stock"),
    ("AMZN", "Amazon.com Inc.", "Stock"),
    ("NKE", "Nike Inc.", "Stock"),
    ("SBUX", "Starbucks Corporation", "Stock"),
    ("MCD", "McDonald's Corporation", "Stock"),
    ("DIS", "Walt Disney Company", "Stock"),
    ("NFLX", "Netflix Inc.", "Stock"),
    ("CMCSA", "Comcast Corporation", "Stock"),
    ("PG", "Procter & Gamble", "Stock"),
    ("KO", "Coca-Cola Company", "Stock"),
    ("PEP", "PepsiCo Inc.", "Stock"),
    ("PM", "Philip Morris International", "Stock"),
    ("MO", "Altria Group Inc.", "Stock"),
    ("CL", "Colgate-Palmolive", "Stock"),
    ("EL", "Estee Lauder Companies", "Stock"),
    ("KMB", "Kimberly-Clark Corp.", "Stock"),
    ("GIS", "General Mills Inc.", "Stock"),
    ("K", "Kellanova", "Stock"),
    ("KHC", "Kraft Heinz Company", "Stock"),
    ("MDLZ", "Mondelez International", "Stock"),
    ("HSY", "Hershey Company", "Stock"),
    ("STZ", "Constellation Brands", "Stock"),
    ("TAP", "Molson Coors Beverage", "Stock"),
    ("BUD", "Anheuser-Busch InBev", "Stock"),
    ("TJX", "TJX Companies Inc.", "Stock"),
    ("ROST", "Ross Stores Inc.", "Stock"),
    ("BURL", "Burlington Stores", "Stock"),
    ("LULU", "Lululemon Athletica", "Stock"),
    ("GPS", "Gap Inc.", "Stock"),
    ("ANF", "Abercrombie & Fitch", "Stock"),
    ("F", "Ford Motor Company", "Stock"),
    ("GM", "General Motors Company", "Stock"),
    ("RIVN", "Rivian Automotive", "Stock"),
    ("LCID", "Lucid Group Inc.", "Stock"),

    # Industrials
    ("CAT", "Caterpillar Inc.", "Stock"),
    ("DE", "Deere & Company", "Stock"),
    ("BA", "Boeing Company", "Stock"),
    ("LMT", "Lockheed Martin", "Stock"),
    ("RTX", "RTX Corporation", "Stock"),
    ("NOC", "Northrop Grumman", "Stock"),
    ("GD", "General Dynamics", "Stock"),
    ("GE", "GE Aerospace", "Stock"),
    ("HON", "Honeywell International", "Stock"),
    ("MMM", "3M Company", "Stock"),
    ("UPS", "United Parcel Service", "Stock"),
    ("FDX", "FedEx Corporation", "Stock"),
    ("UNP", "Union Pacific Corp.", "Stock"),
    ("CSX", "CSX Corporation", "Stock"),
    ("NSC", "Norfolk Southern", "Stock"),
    ("DAL", "Delta Air Lines", "Stock"),
    ("UAL", "United Airlines Holdings", "Stock"),
    ("AAL", "American Airlines Group", "Stock"),
    ("LUV", "Southwest Airlines", "Stock"),
    ("WM", "Waste Management Inc.", "Stock"),
    ("RSG", "Republic Services", "Stock"),
    ("EMR", "Emerson Electric", "Stock"),
    ("ETN", "Eaton Corporation", "Stock"),
    ("ITW", "Illinois Tool Works", "Stock"),
    ("PH", "Parker-Hannifin", "Stock"),
    ("ROK", "Rockwell Automation", "Stock"),

    # Energy
    ("XOM", "Exxon Mobil Corp.", "Stock"),
    ("CVX", "Chevron Corporation", "Stock"),
    ("COP", "ConocoPhillips", "Stock"),
    ("EOG", "EOG Resources Inc.", "Stock"),
    ("SLB", "Schlumberger Limited", "Stock"),
    ("OXY", "Occidental Petroleum", "Stock"),
    ("MPC", "Marathon Petroleum", "Stock"),
    ("VLO", "Valero Energy Corp.", "Stock"),
    ("PSX", "Phillips 66", "Stock"),
    ("PXD", "Pioneer Natural Resources", "Stock"),
    ("DVN", "Devon Energy Corp.", "Stock"),
    ("HAL", "Halliburton Company", "Stock"),
    ("BKR", "Baker Hughes Company", "Stock"),
    ("KMI", "Kinder Morgan Inc.", "Stock"),
    ("WMB", "Williams Companies", "Stock"),
    ("OKE", "ONEOK Inc.", "Stock"),
    ("ET", "Energy Transfer LP", "Stock"),
    ("EPD", "Enterprise Products Partners", "Stock"),

    # Utilities & Real Estate
    ("NEE", "NextEra Energy Inc.", "Stock"),
    ("DUK", "Duke Energy Corp.", "Stock"),
    ("SO", "Southern Company", "Stock"),
    ("D", "Dominion Energy Inc.", "Stock"),
    ("AEP", "American Electric Power", "Stock"),
    ("EXC", "Exelon Corporation", "Stock"),
    ("SRE", "Sempra Energy", "Stock"),
    ("XEL", "Xcel Energy Inc.", "Stock"),
    ("ED", "Consolidated Edison", "Stock"),
    ("WEC", "WEC Energy Group", "Stock"),
    ("AWK", "American Water Works", "Stock"),
    ("AMT", "American Tower Corp.", "REIT"),
    ("PLD", "Prologis Inc.", "REIT"),
    ("CCI", "Crown Castle Inc.", "REIT"),
    ("EQIX", "Equinix Inc.", "REIT"),
    ("PSA", "Public Storage", "REIT"),
    ("O", "Realty Income Corp.", "REIT"),
    ("WELL", "Welltower Inc.", "REIT"),
    ("DLR", "Digital Realty Trust", "REIT"),
    ("SPG", "Simon Property Group", "REIT"),
    ("AVB", "AvalonBay Communities", "REIT"),
    ("EQR", "Equity Residential", "REIT"),
    ("VTR", "Ventas Inc.", "REIT"),
    ("ARE", "Alexandria Real Estate", "REIT"),
    ("MAA", "Mid-America Apartment", "REIT"),
    ("UDR", "UDR Inc.", "REIT"),
    ("ESS", "Essex Property Trust", "REIT"),
    ("INVH", "Invitation Homes", "REIT"),
    ("SUI", "Sun Communities Inc.", "REIT"),
    ("ELS", "Equity LifeStyle Props", "REIT"),

    # Communications
    ("T", "AT&T Inc.", "Stock"),
    ("VZ", "Verizon Communications", "Stock"),
    ("TMUS", "T-Mobile US Inc.", "Stock"),
    ("CHTR", "Charter Communications", "Stock"),
    ("CMCSA", "Comcast Corporation", "Stock"),

    # Materials
    ("LIN", "Linde plc", "Stock"),
    ("APD", "Air Products & Chemicals", "Stock"),
    ("SHW", "Sherwin-Williams", "Stock"),
    ("ECL", "Ecolab Inc.", "Stock"),
    ("DD", "DuPont de Nemours", "Stock"),
    ("DOW", "Dow Inc.", "Stock"),
    ("NEM", "Newmont Corporation", "Stock"),
    ("FCX", "Freeport-McMoRan", "Stock"),
    ("NUE", "Nucor Corporation", "Stock"),
    ("STLD", "Steel Dynamics Inc.", "Stock"),
    ("CLF", "Cleveland-Cliffs Inc.", "Stock"),
    ("X", "United States Steel", "Stock"),
    ("AA", "Alcoa Corporation", "Stock"),

    # Popular ETFs
    ("SPY", "SPDR S&P 500 ETF", "ETF"),
    ("VOO", "Vanguard S&P 500 ETF", "ETF"),
    ("IVV", "iShares Core S&P 500", "ETF"),
    ("QQQ", "Invesco QQQ Trust", "ETF"),
    ("VTI", "Vanguard Total Stock Market", "ETF"),
    ("IWM", "iShares Russell 2000", "ETF"),
    ("IWF", "iShares Russell 1000 Growth", "ETF"),
    ("IWD", "iShares Russell 1000 Value", "ETF"),
    ("VUG", "Vanguard Growth ETF", "ETF"),
    ("VTV", "Vanguard Value ETF", "ETF"),
    ("VIG", "Vanguard Dividend Appreciation", "ETF"),
    ("VYM", "Vanguard High Dividend Yield", "ETF"),
    ("SCHD", "Schwab US Dividend Equity", "ETF"),
    ("DVY", "iShares Select Dividend", "ETF"),
    ("HDV", "iShares Core High Dividend", "ETF"),
    ("DGRO", "iShares Core Dividend Growth", "ETF"),
    ("NOBL", "ProShares S&P 500 Dividend", "ETF"),
    ("XLK", "Technology Select Sector", "ETF"),
    ("XLF", "Financial Select Sector", "ETF"),
    ("XLV", "Health Care Select Sector", "ETF"),
    ("XLE", "Energy Select Sector", "ETF"),
    ("XLI", "Industrial Select Sector", "ETF"),
    ("XLY", "Consumer Discretionary Select", "ETF"),
    ("XLP", "Consumer Staples Select", "ETF"),
    ("XLU", "Utilities Select Sector", "ETF"),
    ("XLB", "Materials Select Sector", "ETF"),
    ("XLRE", "Real Estate Select Sector", "ETF"),
    ("XLC", "Communication Services Select", "ETF"),
    ("VNQ", "Vanguard Real Estate ETF", "ETF"),
    ("SCHH", "Schwab US REIT ETF", "ETF"),
    ("IYR", "iShares US Real Estate", "ETF"),
    ("ARKK", "ARK Innovation ETF", "ETF"),
    ("ARKW", "ARK Next Gen Internet", "ETF"),
    ("ARKF", "ARK Fintech Innovation", "ETF"),
    ("ARKG", "ARK Genomic Revolution", "ETF"),
    ("ARKQ", "ARK Autonomous Tech", "ETF"),
    ("EEM", "iShares MSCI Emerging Markets", "ETF"),
    ("VWO", "Vanguard FTSE Emerging Markets", "ETF"),
    ("EFA", "iShares MSCI EAFE", "ETF"),
    ("VEA", "Vanguard FTSE Developed Markets", "ETF"),
    ("VXUS", "Vanguard Total International", "ETF"),
    ("IEMG", "iShares Core MSCI EM", "ETF"),

    # Bond ETFs
    ("BND", "Vanguard Total Bond Market", "Bond ETF"),
    ("AGG", "iShares Core US Aggregate", "Bond ETF"),
    ("TLT", "iShares 20+ Year Treasury", "Bond ETF"),
    ("IEF", "iShares 7-10 Year Treasury", "Bond ETF"),
    ("SHY", "iShares 1-3 Year Treasury", "Bond ETF"),
    ("GOVT", "iShares US Treasury Bond", "Bond ETF"),
    ("TIP", "iShares TIPS Bond", "Bond ETF"),
    ("LQD", "iShares Investment Grade Corp", "Bond ETF"),
    ("VCIT", "Vanguard Intermediate Corp", "Bond ETF"),
    ("VCSH", "Vanguard Short-Term Corp", "Bond ETF"),
    ("HYG", "iShares High Yield Corporate", "Bond ETF"),
    ("JNK", "SPDR Bloomberg High Yield", "Bond ETF"),
    ("MUB", "iShares National Muni Bond", "Bond ETF"),
    ("VTEB", "Vanguard Tax-Exempt Bond", "Bond ETF"),
    ("MBB", "iShares MBS ETF", "Bond ETF"),
    ("VMBS", "Vanguard Mortgage-Backed", "Bond ETF"),
    ("EMB", "iShares JP Morgan EM Bond", "Bond ETF"),
    ("BNDX", "Vanguard Total Intl Bond", "Bond ETF"),
    ("IGIB", "iShares Intermediate Corp", "Bond ETF"),
    ("SCHO", "Schwab Short-Term US Treasury", "Bond ETF"),
    ("SCHR", "Schwab Intermediate US Treasury", "Bond ETF"),
    ("SCHZ", "Schwab US Aggregate Bond", "Bond ETF"),

    # Leveraged & Inverse ETFs
    ("TQQQ", "ProShares UltraPro QQQ", "ETF"),
    ("SQQQ", "ProShares UltraPro Short QQQ", "ETF"),
    ("SPXL", "Direxion Daily S&P 500 Bull 3X", "ETF"),
    ("SPXS", "Direxion Daily S&P 500 Bear 3X", "ETF"),
    ("UPRO", "ProShares UltraPro S&P 500", "ETF"),
    ("SOXL", "Direxion Daily Semiconductor Bull 3X", "ETF"),
    ("SOXS", "Direxion Daily Semiconductor Bear 3X", "ETF"),

    # Crypto-related
    ("COIN", "Coinbase Global Inc.", "Stock"),
    ("MSTR", "MicroStrategy Inc.", "Stock"),
    ("MARA", "Marathon Digital Holdings", "Stock"),
    ("RIOT", "Riot Platforms Inc.", "Stock"),
    ("CLSK", "CleanSpark Inc.", "Stock"),
    ("HUT", "Hut 8 Mining Corp.", "Stock"),
    ("BITF", "Bitfarms Ltd.", "Stock"),

    # Other Popular
    ("BRKB", "Berkshire Hathaway B", "Stock"),
    ("BRK-B", "Berkshire Hathaway B", "Stock"),
    ("BRK.B", "Berkshire Hathaway B", "Stock"),
    ("UBER", "Uber Technologies", "Stock"),
    ("LYFT", "Lyft Inc.", "Stock"),
    ("ABNB", "Airbnb Inc.", "Stock"),
    ("DASH", "DoorDash Inc.", "Stock"),
    ("ZM", "Zoom Video Communications", "Stock"),
    ("DOCU", "DocuSign Inc.", "Stock"),
    ("TWLO", "Twilio Inc.", "Stock"),
    ("OKTA", "Okta Inc.", "Stock"),
    ("MDB", "MongoDB Inc.", "Stock"),
    ("TTD", "Trade Desk Inc.", "Stock"),
    ("ROKU", "Roku Inc.", "Stock"),
    ("SNAP", "Snap Inc.", "Stock"),
    ("PINS", "Pinterest Inc.", "Stock"),
    ("SPOT", "Spotify Technology", "Stock"),
    ("SQ", "Block Inc.", "Stock"),
    ("SHOP", "Shopify Inc.", "Stock"),
    ("SE", "Sea Limited", "Stock"),
    ("BABA", "Alibaba Group", "Stock"),
    ("JD", "JD.com Inc.", "Stock"),
    ("PDD", "PDD Holdings Inc.", "Stock"),
    ("BIDU", "Baidu Inc.", "Stock"),
    ("NIO", "NIO Inc.", "Stock"),
    ("XPEV", "XPeng Inc.", "Stock"),
    ("LI", "Li Auto Inc.", "Stock"),
    ("TSM", "Taiwan Semiconductor", "Stock"),
    ("ASML", "ASML Holding NV", "Stock"),
    ("SAP", "SAP SE", "Stock"),
    ("TM", "Toyota Motor Corp.", "Stock"),
    ("SONY", "Sony Group Corporation", "Stock"),
    ("NVO", "Novo Nordisk A/S", "Stock"),
    ("AZN", "AstraZeneca PLC", "Stock"),
    ("SNY", "Sanofi SA", "Stock"),
    ("GSK", "GSK plc", "Stock"),
    ("BP", "BP plc", "Stock"),
    ("SHEL", "Shell plc", "Stock"),
    ("TTE", "TotalEnergies SE", "Stock"),
    ("RIO", "Rio Tinto Group", "Stock"),
    ("BHP", "BHP Group Limited", "Stock"),
    ("VALE", "Vale S.A.", "Stock"),
    ("GOLD", "Barrick Gold Corp.", "Stock"),
    ("NEM", "Newmont Corporation", "Stock"),
    ("WPM", "Wheaton Precious Metals", "Stock"),
]


def search_tickers(query: str, limit: int = 10) -> List[Tuple[str, str, str]]:
    """
    Search ticker database for matches.
    Returns list of (symbol, name, type) tuples.
    """
    if not query or len(query) < 1:
        return []

    query = query.upper().strip()
    results = []

    # First, exact symbol matches
    for symbol, name, asset_type in TICKER_DATABASE:
        if symbol == query:
            results.append((symbol, name, asset_type))

    # Then, symbol starts with query
    for symbol, name, asset_type in TICKER_DATABASE:
        if symbol.startswith(query) and (symbol, name, asset_type) not in results:
            results.append((symbol, name, asset_type))

    # Then, name contains query
    for symbol, name, asset_type in TICKER_DATABASE:
        if query.lower() in name.lower() and (symbol, name, asset_type) not in results:
            results.append((symbol, name, asset_type))

    return results[:limit]

POPULAR_ETFS = [
    ("SPY", "S&P 500 ETF"),
    ("QQQ", "Nasdaq 100 ETF"),
    ("IWM", "Russell 2000 ETF"),
    ("VTI", "Total Stock Market ETF"),
    ("VOO", "Vanguard S&P 500"),
    ("VUG", "Vanguard Growth ETF"),
    ("VTV", "Vanguard Value ETF"),
    ("SCHD", "Schwab US Dividend ETF"),
    ("VYM", "Vanguard High Dividend"),
    ("XLK", "Technology Select Sector"),
    ("XLF", "Financial Select Sector"),
    ("XLV", "Health Care Select Sector"),
]

POPULAR_BOND_ETFS = [
    ("BND", "Total Bond Market ETF"),
    ("AGG", "Core US Aggregate Bond"),
    ("TLT", "20+ Year Treasury Bond"),
    ("IEF", "7-10 Year Treasury Bond"),
    ("SHY", "1-3 Year Treasury Bond"),
    ("LQD", "Investment Grade Corporate"),
    ("HYG", "High Yield Corporate Bond"),
    ("TIP", "TIPS Bond ETF"),
    ("MUB", "National Muni Bond ETF"),
    ("VCIT", "Intermediate-Term Corp Bond"),
]

POPULAR_REITS = [
    ("VNQ", "Vanguard Real Estate ETF"),
    ("SCHH", "Schwab US REIT ETF"),
    ("O", "Realty Income Corp"),
    ("AMT", "American Tower Corp"),
    ("PLD", "Prologis Inc"),
    ("SPG", "Simon Property Group"),
]


# ─────────────────────────────────────────────
# Dividend Calendar Functions
# ─────────────────────────────────────────────

def get_dividend_history(symbol: str, lookback_years: int = 5) -> pd.DataFrame:
    """
    Fetch dividend history for a symbol using all available data.
    Returns DataFrame with ex-date, amount, and additional metadata.
    """
    try:
        ticker = yf.Ticker(symbol)
        dividends = ticker.dividends

        if dividends.empty:
            return pd.DataFrame()

        # Use more historical data for better predictions (up to 5 years)
        cutoff = datetime.now() - timedelta(days=lookback_years * 365)
        dividends = dividends[dividends.index >= cutoff]

        if dividends.empty:
            return pd.DataFrame()

        df = pd.DataFrame({
            "ex_date": dividends.index,
            "amount": dividends.values,
        })
        df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.tz_localize(None)

        # Add month and day-of-month for pattern detection
        df["month"] = df["ex_date"].dt.month
        df["day"] = df["ex_date"].dt.day
        df["year"] = df["ex_date"].dt.year

        return df
    except Exception:
        return pd.DataFrame()


def estimate_dividend_frequency(div_history: pd.DataFrame) -> Tuple[str, int, List[int]]:
    """
    Estimate dividend frequency from history.
    Returns (frequency_label, months_between_payments, typical_months).
    """
    if div_history.empty or len(div_history) < 2:
        return "Unknown", 3, []

    # Calculate average days between dividends
    dates = sorted(div_history["ex_date"].tolist())
    if len(dates) < 2:
        return "Unknown", 3, []

    gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    avg_gap = np.mean(gaps)

    # Get typical months when dividends are paid
    typical_months = div_history["month"].value_counts().head(4).index.tolist()

    if avg_gap < 45:  # ~Monthly
        return "Monthly", 1, list(range(1, 13))
    elif avg_gap < 120:  # ~Quarterly
        return "Quarterly", 3, typical_months
    elif avg_gap < 200:  # ~Semi-Annual
        return "Semi-Annual", 6, typical_months
    else:  # Annual
        return "Annual", 12, typical_months


def get_typical_div_day(div_history: pd.DataFrame, target_month: int) -> int:
    """Get the typical day of month for dividend in a specific month."""
    month_divs = div_history[div_history["month"] == target_month]
    if not month_divs.empty:
        return int(month_divs["day"].median())
    # Fallback to overall median day
    return int(div_history["day"].median()) if not div_history.empty else 15


def predict_future_dividends(
    symbol: str,
    name: str,
    shares: float,
    div_history: pd.DataFrame,
    months_ahead: int = 12
) -> List[DividendEvent]:
    """
    Predict future dividend payments based on comprehensive historical pattern analysis.
    Uses all available data for more accurate predictions.
    """
    if div_history.empty:
        return []

    frequency, months_between, typical_months = estimate_dividend_frequency(div_history)

    # Sort by date
    div_history = div_history.sort_values("ex_date")

    # Get last dividend info
    last_div = div_history.iloc[-1]
    last_date = last_div["ex_date"]

    # Calculate average dividend amount with trend consideration
    recent_divs = div_history.tail(8)  # Use last 8 payments for averaging
    if len(recent_divs) >= 4:
        # Weight recent dividends more heavily
        weights = np.array([1, 1, 2, 2, 3, 3, 4, 4][-len(recent_divs):])
        weights = weights / weights.sum()
        avg_amount = np.average(recent_divs["amount"].values, weights=weights)
    else:
        avg_amount = recent_divs["amount"].mean()

    # Calculate dividend growth rate if enough history
    if len(div_history) >= 8:
        old_avg = div_history.head(4)["amount"].mean()
        new_avg = div_history.tail(4)["amount"].mean()
        if old_avg > 0:
            growth_rate = (new_avg / old_avg) ** 0.25 - 1  # Quarterly growth
        else:
            growth_rate = 0
    else:
        growth_rate = 0

    # Project future dividends
    future_events = []
    current_date = datetime.now()
    end_date = current_date + relativedelta(months=months_ahead)

    if frequency == "Monthly":
        # For monthly, iterate through each month
        next_date = last_date + relativedelta(months=1)
        payment_count = 0
        while next_date <= end_date:
            if next_date > current_date:
                # Adjust amount for growth
                adjusted_amount = avg_amount * ((1 + growth_rate) ** payment_count)
                pay_date = next_date + timedelta(days=14)

                future_events.append(DividendEvent(
                    symbol=symbol,
                    name=name,
                    ex_date=next_date,
                    pay_date=pay_date,
                    amount=adjusted_amount,
                    shares=shares,
                    expected_income=adjusted_amount * shares,
                    frequency=frequency,
                ))
                payment_count += 1
            next_date = next_date + relativedelta(months=1)
    else:
        # For quarterly/semi-annual/annual, use typical months
        if typical_months:
            # Find next occurrence based on typical payment months
            for i in range(months_ahead + 12):
                check_date = current_date + relativedelta(months=i)
                check_month = check_date.month

                if check_month in typical_months:
                    # Get typical day for this month
                    typical_day = get_typical_div_day(div_history, check_month)
                    try:
                        proj_date = datetime(check_date.year, check_month, min(typical_day, 28))
                    except ValueError:
                        proj_date = datetime(check_date.year, check_month, 15)

                    if proj_date > current_date and proj_date <= end_date:
                        # Check if we already have this date
                        existing = [e for e in future_events
                                   if abs((e.ex_date - proj_date).days) < 20]
                        if not existing:
                            payment_count = len(future_events)
                            adjusted_amount = avg_amount * ((1 + growth_rate) ** payment_count)
                            pay_date = proj_date + timedelta(days=21)

                            future_events.append(DividendEvent(
                                symbol=symbol,
                                name=name,
                                ex_date=proj_date,
                                pay_date=pay_date,
                                amount=adjusted_amount,
                                shares=shares,
                                expected_income=adjusted_amount * shares,
                                frequency=frequency,
                            ))
        else:
            # Fallback: project from last date
            next_date = last_date
            payment_count = 0
            while next_date <= end_date:
                next_date = next_date + relativedelta(months=months_between)
                if next_date > current_date and next_date <= end_date:
                    adjusted_amount = avg_amount * ((1 + growth_rate) ** payment_count)
                    pay_date = next_date + timedelta(days=21)

                    future_events.append(DividendEvent(
                        symbol=symbol,
                        name=name,
                        ex_date=next_date,
                        pay_date=pay_date,
                        amount=adjusted_amount,
                        shares=shares,
                        expected_income=adjusted_amount * shares,
                        frequency=frequency,
                    ))
                    payment_count += 1

    return future_events


def build_dividend_calendar(
    holdings: List[Holding],
    target_year: int,
    target_month: int
) -> Dict:
    """
    Build dividend calendar for a specific month.

    Returns dict with:
        - events: List of DividendEvent for the month
        - calendar_grid: 2D list for calendar display
        - monthly_total: Total expected income for the month
        - by_day: Dict mapping day -> list of events
    """
    # Get all dividend events for all holdings
    all_events = []

    for h in holdings:
        if h.dividend_yield <= 0:
            continue

        div_history = get_dividend_history(h.symbol)
        if div_history.empty:
            continue

        # Get predicted dividends for next 12 months
        future_divs = predict_future_dividends(
            h.symbol, h.name, h.shares, div_history, months_ahead=12
        )
        all_events.extend(future_divs)

    # Filter to target month
    month_events = [
        e for e in all_events
        if e.ex_date.year == target_year and e.ex_date.month == target_month
    ]

    # Sort by date
    month_events.sort(key=lambda x: x.ex_date)

    # Build by-day mapping
    by_day = {}
    for event in month_events:
        day = event.ex_date.day
        if day not in by_day:
            by_day[day] = []
        by_day[day].append(event)

    # Calculate monthly total
    monthly_total = sum(e.expected_income for e in month_events)

    # Build calendar grid
    cal = calendar.Calendar(firstweekday=6)  # Start on Sunday
    month_days = cal.monthdayscalendar(target_year, target_month)

    calendar_grid = []
    for week in month_days:
        week_data = []
        for day in week:
            if day == 0:
                week_data.append({"day": 0, "events": []})
            else:
                week_data.append({
                    "day": day,
                    "events": by_day.get(day, [])
                })
        calendar_grid.append(week_data)

    return {
        "events": month_events,
        "calendar_grid": calendar_grid,
        "monthly_total": monthly_total,
        "by_day": by_day,
        "year": target_year,
        "month": target_month,
        "month_name": calendar.month_name[target_month],
    }


def get_annual_dividend_projection(holdings: List[Holding]) -> Dict:
    """
    Get 12-month dividend projection by month.
    """
    current = datetime.now()
    monthly_totals = {}

    for i in range(12):
        target_date = current + relativedelta(months=i)
        year = target_date.year
        month = target_date.month

        cal_data = build_dividend_calendar(holdings, year, month)
        month_key = f"{year}-{month:02d}"
        monthly_totals[month_key] = {
            "month_name": cal_data["month_name"],
            "year": year,
            "total": cal_data["monthly_total"],
            "event_count": len(cal_data["events"]),
        }

    return monthly_totals


# ─────────────────────────────────────────────
# Earnings Calendar Functions
# ─────────────────────────────────────────────

def get_earnings_dates(symbol: str, name: str) -> List[EarningsEvent]:
    """
    Fetch upcoming and recent earnings dates for a symbol using all available data.
    Returns list of EarningsEvent objects.
    """
    try:
        ticker = yf.Ticker(symbol)
        earnings_events = []
        historical_dates = []  # Track historical pattern for estimation

        # Get earnings dates from calendar (confirmed upcoming)
        try:
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                if 'Earnings Date' in cal.index:
                    earnings_dates = cal.loc['Earnings Date']
                    if isinstance(earnings_dates, pd.Series):
                        for date in earnings_dates:
                            if pd.notna(date):
                                parsed_date = pd.to_datetime(date).to_pydatetime().replace(tzinfo=None)
                                earnings_events.append(EarningsEvent(
                                    symbol=symbol,
                                    name=name,
                                    earnings_date=parsed_date,
                                    time_of_day="Unknown",
                                    eps_estimate=None,
                                    revenue_estimate=None,
                                    is_confirmed=True,
                                ))
                    elif pd.notna(earnings_dates):
                        parsed_date = pd.to_datetime(earnings_dates).to_pydatetime().replace(tzinfo=None)
                        earnings_events.append(EarningsEvent(
                            symbol=symbol,
                            name=name,
                            earnings_date=parsed_date,
                            time_of_day="Unknown",
                            eps_estimate=None,
                            revenue_estimate=None,
                            is_confirmed=True,
                        ))
        except Exception:
            pass

        # Get from earnings_dates property - includes historical data
        try:
            earnings_df = ticker.earnings_dates
            if earnings_df is not None and not earnings_df.empty:
                for idx in earnings_df.index:
                    try:
                        date = pd.to_datetime(idx).to_pydatetime()
                        if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                            date = date.replace(tzinfo=None)

                        # Collect historical dates for pattern analysis
                        if date < datetime.now():
                            historical_dates.append(date)

                        # Include future dates and recent past (for verification)
                        if date >= datetime.now() - timedelta(days=30):
                            eps_est = None
                            rev_est = None

                            if 'EPS Estimate' in earnings_df.columns:
                                eps_val = earnings_df.loc[idx].get('EPS Estimate')
                                eps_est = float(eps_val) if pd.notna(eps_val) else None

                            if 'Revenue Estimate' in earnings_df.columns:
                                rev_val = earnings_df.loc[idx].get('Revenue Estimate')
                                rev_est = float(rev_val) if pd.notna(rev_val) else None

                            existing = [e for e in earnings_events
                                       if abs((e.earnings_date - date).days) < 2]
                            if not existing:
                                earnings_events.append(EarningsEvent(
                                    symbol=symbol,
                                    name=name,
                                    earnings_date=date,
                                    time_of_day="Unknown",
                                    eps_estimate=eps_est,
                                    revenue_estimate=rev_est,
                                    is_confirmed=True,
                                ))
                    except Exception:
                        continue
        except Exception:
            pass

        # Try quarterly earnings for additional historical data
        try:
            quarterly = ticker.quarterly_earnings
            if quarterly is not None and not quarterly.empty:
                for idx in quarterly.index:
                    try:
                        date = pd.to_datetime(idx).to_pydatetime()
                        if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                            date = date.replace(tzinfo=None)
                        historical_dates.append(date)
                    except Exception:
                        continue
        except Exception:
            pass

        # If we have historical data but no future events, estimate based on pattern
        if not earnings_events and historical_dates:
            earnings_events = estimate_future_earnings_from_history(symbol, name, historical_dates)
        elif not earnings_events:
            earnings_events = estimate_future_earnings(symbol, name)

        return earnings_events

    except Exception:
        return estimate_future_earnings(symbol, name)


def estimate_future_earnings_from_history(
    symbol: str,
    name: str,
    historical_dates: List[datetime]
) -> List[EarningsEvent]:
    """
    Estimate future earnings dates based on actual historical reporting patterns.
    Uses historical data to predict more accurate future dates.
    """
    if not historical_dates:
        return estimate_future_earnings(symbol, name)

    earnings_events = []
    current = datetime.now()

    # Sort historical dates
    historical_dates = sorted(historical_dates)

    # Analyze historical pattern - which months and approximate days
    month_day_map = {}
    for date in historical_dates:
        month = date.month
        if month not in month_day_map:
            month_day_map[month] = []
        month_day_map[month].append(date.day)

    # Get typical reporting months (should have at least 2 entries)
    typical_months = [m for m, days in month_day_map.items() if len(days) >= 1]

    if not typical_months:
        return estimate_future_earnings(symbol, name)

    # Calculate typical day for each reporting month
    typical_month_days = {}
    for month in typical_months:
        typical_month_days[month] = int(np.median(month_day_map[month]))

    # Project next 4 quarters based on pattern
    for i in range(12):
        check_date = current + relativedelta(months=i)
        check_month = check_date.month

        if check_month in typical_months:
            typical_day = typical_month_days[check_month]
            try:
                proj_date = datetime(check_date.year, check_month, min(typical_day, 28))
            except ValueError:
                proj_date = datetime(check_date.year, check_month, 15)

            if proj_date > current:
                # Check if we already have a nearby date
                existing = [e for e in earnings_events
                           if abs((e.earnings_date - proj_date).days) < 25]
                if not existing:
                    earnings_events.append(EarningsEvent(
                        symbol=symbol,
                        name=name,
                        earnings_date=proj_date,
                        time_of_day="Unknown",
                        eps_estimate=None,
                        revenue_estimate=None,
                        is_confirmed=False,
                    ))

            # Limit to 4 future quarters
            if len(earnings_events) >= 4:
                break

    return earnings_events


def estimate_future_earnings(symbol: str, name: str) -> List[EarningsEvent]:
    """
    Estimate future earnings dates based on typical quarterly pattern.
    Uses standard corporate reporting calendar as fallback.
    """
    earnings_events = []
    current = datetime.now()

    # Standard earnings reporting windows:
    # Q4 (Dec) -> Late Jan / Early Feb
    # Q1 (Mar) -> Late Apr / Early May
    # Q2 (Jun) -> Late Jul / Early Aug
    # Q3 (Sep) -> Late Oct / Early Nov

    # Map current month to next reporting period
    reporting_schedule = [
        (1, 25),   # Q4 results in late January
        (4, 25),   # Q1 results in late April
        (7, 25),   # Q2 results in late July
        (10, 25),  # Q3 results in late October
    ]

    # Find next 4 reporting dates
    for month, day in reporting_schedule:
        # Calculate next occurrence of this month
        year = current.year
        estimated_date = datetime(year, month, day)

        # If this date has passed, use next year
        if estimated_date <= current:
            estimated_date = datetime(year + 1, month, day)

        earnings_events.append(EarningsEvent(
            symbol=symbol,
            name=name,
            earnings_date=estimated_date,
            time_of_day="Unknown",
            eps_estimate=None,
            revenue_estimate=None,
            is_confirmed=False,
        ))

    # Sort by date and return next 4
    earnings_events.sort(key=lambda x: x.earnings_date)
    return earnings_events[:4]


def build_earnings_calendar(
    holdings: List[Holding],
    target_year: int,
    target_month: int
) -> Dict:
    """
    Build earnings calendar for a specific month.
    """
    all_events = []

    for h in holdings:
        # Skip ETFs - they don't have earnings
        if h.asset_type in ["ETF", "Bond ETF"]:
            continue

        earnings = get_earnings_dates(h.symbol, h.name)
        all_events.extend(earnings)

    # Filter to target month
    month_events = [
        e for e in all_events
        if e.earnings_date.year == target_year and e.earnings_date.month == target_month
    ]

    # Sort by date
    month_events.sort(key=lambda x: x.earnings_date)

    # Build by-day mapping
    by_day = {}
    for event in month_events:
        day = event.earnings_date.day
        if day not in by_day:
            by_day[day] = []
        by_day[day].append(event)

    return {
        "events": month_events,
        "by_day": by_day,
        "year": target_year,
        "month": target_month,
        "month_name": calendar.month_name[target_month],
    }


@dataclass
class FedEvent:
    """A Federal Reserve or major financial news event."""
    event_date: datetime
    event_type: str  # "FOMC", "Jobs Report", "CPI", "GDP", etc.
    description: str
    importance: str  # "HIGH", "MEDIUM", "LOW"


# Federal Reserve FOMC meeting dates for 2024-2026
# These are announced in advance and are reliable
FED_FOMC_DATES = [
    # 2024
    ("2024-01-31", "FOMC Decision", "HIGH"),
    ("2024-03-20", "FOMC Decision", "HIGH"),
    ("2024-05-01", "FOMC Decision", "HIGH"),
    ("2024-06-12", "FOMC Decision", "HIGH"),
    ("2024-07-31", "FOMC Decision", "HIGH"),
    ("2024-09-18", "FOMC Decision", "HIGH"),
    ("2024-11-07", "FOMC Decision", "HIGH"),
    ("2024-12-18", "FOMC Decision", "HIGH"),
    # 2025
    ("2025-01-29", "FOMC Decision", "HIGH"),
    ("2025-03-19", "FOMC Decision", "HIGH"),
    ("2025-05-07", "FOMC Decision", "HIGH"),
    ("2025-06-18", "FOMC Decision", "HIGH"),
    ("2025-07-30", "FOMC Decision", "HIGH"),
    ("2025-09-17", "FOMC Decision", "HIGH"),
    ("2025-11-05", "FOMC Decision", "HIGH"),
    ("2025-12-17", "FOMC Decision", "HIGH"),
    # 2026
    ("2026-01-28", "FOMC Decision", "HIGH"),
    ("2026-03-18", "FOMC Decision", "HIGH"),
    ("2026-04-29", "FOMC Decision", "HIGH"),
    ("2026-06-17", "FOMC Decision", "HIGH"),
    ("2026-07-29", "FOMC Decision", "HIGH"),
    ("2026-09-16", "FOMC Decision", "HIGH"),
    ("2026-11-04", "FOMC Decision", "HIGH"),
    ("2026-12-16", "FOMC Decision", "HIGH"),
]

# Economic calendar - recurring monthly events
# Jobs report: First Friday of each month
# CPI: Around 10th-14th of month
# GDP: End of month (quarterly)


def get_fed_and_econ_events(target_year: int, target_month: int) -> List[FedEvent]:
    """
    Get Fed decisions and major economic events for a specific month.
    """
    events = []

    # Add FOMC dates
    for date_str, event_type, importance in FED_FOMC_DATES:
        event_date = datetime.strptime(date_str, "%Y-%m-%d")
        if event_date.year == target_year and event_date.month == target_month:
            events.append(FedEvent(
                event_date=event_date,
                event_type="FOMC",
                description=f"Federal Reserve {event_type}",
                importance=importance,
            ))

    # Add recurring economic events
    # Jobs Report - First Friday of month
    first_day = datetime(target_year, target_month, 1)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    events.append(FedEvent(
        event_date=first_friday,
        event_type="Jobs Report",
        description="Non-Farm Payrolls & Unemployment",
        importance="HIGH",
    ))

    # CPI - Usually around 10th-13th
    cpi_day = 12 if target_month % 2 == 1 else 13
    try:
        cpi_date = datetime(target_year, target_month, cpi_day)
        # Adjust for weekends
        if cpi_date.weekday() == 5:  # Saturday
            cpi_date -= timedelta(days=1)
        elif cpi_date.weekday() == 6:  # Sunday
            cpi_date += timedelta(days=1)
        events.append(FedEvent(
            event_date=cpi_date,
            event_type="CPI",
            description="Consumer Price Index Report",
            importance="HIGH",
        ))
    except ValueError:
        pass

    # Options expiration - Third Friday of month
    third_friday = first_day + timedelta(days=days_until_friday + 14)
    events.append(FedEvent(
        event_date=third_friday,
        event_type="OpEx",
        description="Monthly Options Expiration",
        importance="MEDIUM",
    ))

    # Quarterly events (GDP, etc.)
    if target_month in [1, 4, 7, 10]:
        # GDP usually released ~25th-28th
        gdp_day = 26
        try:
            gdp_date = datetime(target_year, target_month, gdp_day)
            if gdp_date.weekday() == 5:
                gdp_date -= timedelta(days=1)
            elif gdp_date.weekday() == 6:
                gdp_date += timedelta(days=1)
            events.append(FedEvent(
                event_date=gdp_date,
                event_type="GDP",
                description="Quarterly GDP Report",
                importance="HIGH",
            ))
        except ValueError:
            pass

    return events


def build_combined_calendar(
    holdings: List[Holding],
    target_year: int,
    target_month: int
) -> Dict:
    """
    Build combined dividend + earnings + Fed/econ calendar for a specific month.
    """
    # Get dividend events
    div_events = []
    for h in holdings:
        if h.dividend_yield <= 0:
            continue
        div_history = get_dividend_history(h.symbol)
        if div_history.empty:
            continue
        future_divs = predict_future_dividends(
            h.symbol, h.name, h.shares, div_history, months_ahead=12
        )
        div_events.extend(future_divs)

    # Filter dividends to target month
    month_div_events = [
        e for e in div_events
        if e.ex_date.year == target_year and e.ex_date.month == target_month
    ]

    # Get earnings events
    earnings_events = []
    for h in holdings:
        if h.asset_type in ["ETF", "Bond ETF"]:
            continue
        earnings = get_earnings_dates(h.symbol, h.name)
        earnings_events.extend(earnings)

    # Filter earnings to target month
    month_earnings_events = [
        e for e in earnings_events
        if e.earnings_date.year == target_year and e.earnings_date.month == target_month
    ]

    # Get Fed and economic events
    fed_events = get_fed_and_econ_events(target_year, target_month)

    # Build by-day mapping (combined)
    by_day = {}

    for event in month_div_events:
        day = event.ex_date.day
        if day not in by_day:
            by_day[day] = {"dividends": [], "earnings": [], "fed_events": []}
        by_day[day]["dividends"].append(event)

    for event in month_earnings_events:
        day = event.earnings_date.day
        if day not in by_day:
            by_day[day] = {"dividends": [], "earnings": [], "fed_events": []}
        by_day[day]["earnings"].append(event)

    for event in fed_events:
        day = event.event_date.day
        if day not in by_day:
            by_day[day] = {"dividends": [], "earnings": [], "fed_events": []}
        by_day[day]["fed_events"].append(event)

    # Calculate monthly dividend total
    monthly_div_total = sum(e.expected_income for e in month_div_events)

    # Build calendar grid
    cal = calendar.Calendar(firstweekday=6)  # Start on Sunday
    month_days = cal.monthdayscalendar(target_year, target_month)

    calendar_grid = []
    for week in month_days:
        week_data = []
        for day in week:
            if day == 0:
                week_data.append({"day": 0, "dividends": [], "earnings": [], "fed_events": []})
            else:
                day_data = by_day.get(day, {"dividends": [], "earnings": [], "fed_events": []})
                week_data.append({
                    "day": day,
                    "dividends": day_data.get("dividends", []),
                    "earnings": day_data.get("earnings", []),
                    "fed_events": day_data.get("fed_events", []),
                })
        calendar_grid.append(week_data)

    return {
        "dividend_events": month_div_events,
        "earnings_events": month_earnings_events,
        "fed_events": fed_events,
        "calendar_grid": calendar_grid,
        "monthly_div_total": monthly_div_total,
        "by_day": by_day,
        "year": target_year,
        "month": target_month,
        "month_name": calendar.month_name[target_month],
    }
