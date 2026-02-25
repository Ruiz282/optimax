"""
Options Data Engine — Core computation module for OptiMax.

Fetches live options chains via yfinance, computes Black-Scholes Greeks,
and enriches chain data for strategy analysis.

All yfinance calls should go through cached wrappers in optimax.py
to avoid rate limiting on Streamlit Cloud.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# Black-Scholes Pricing & Greeks (self-contained)
# ─────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="call"):
    """European option price via Black-Scholes. Works with numpy arrays."""
    S, K, T, r, sigma = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    T_safe = np.maximum(T, 1e-10)
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # At expiration, use intrinsic value
    expired = T <= 0
    if np.any(expired):
        if option_type == "call":
            intrinsic = np.maximum(S - K, 0)
        else:
            intrinsic = np.maximum(K - S, 0)
        price = np.where(expired, intrinsic, price)

    return price


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Compute all five Greeks. Fully vectorized with numpy arrays."""
    S, K, T, r, sigma = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    T_safe = np.maximum(T, 1e-10)
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    pdf_d1 = norm.pdf(d1)
    disc = np.exp(-r * T_safe)

    # Gamma and Vega are the same for calls and puts
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100  # per 1% vol change

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 - r * K * disc * norm.cdf(d2)) / 365
        rho = K * T_safe * disc * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 + r * K * disc * norm.cdf(-d2)) / 365
        rho = -K * T_safe * disc * norm.cdf(-d2) / 100

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


# ─────────────────────────────────────────────
# Data Fetching & Enrichment
# ─────────────────────────────────────────────

def get_ticker_info(symbol):
    """Get basic ticker info: spot price, name, etc.
    Uses fast_info to avoid expensive ticker.info calls.
    Returns a serializable dict (no yf.Ticker object) for caching."""
    ticker = yf.Ticker(symbol)
    try:
        spot = ticker.fast_info["lastPrice"]
    except Exception:
        spot = None
    # Only call ticker.info if fast_info failed for spot
    if spot is None:
        try:
            info = ticker.info
            spot = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            name = info.get("shortName", symbol)
        except Exception:
            spot = 0
            name = symbol
    else:
        name = symbol
    return {"symbol": symbol, "spot": spot, "name": name}


def get_expirations(symbol):
    """Get all available expiration dates for a ticker."""
    ticker = yf.Ticker(symbol)
    return list(ticker.options)


def get_enriched_chain(symbol, expiration, risk_free_rate=0.045, spot=None, ticker_obj=None):
    """
    Fetch options chain from yfinance and enrich with computed Greeks.

    Args:
        symbol: Ticker symbol
        expiration: Expiration date string
        risk_free_rate: Risk-free rate for BS model
        spot: Pre-fetched spot price (avoids redundant API call)
        ticker_obj: Pre-fetched yf.Ticker object (avoids redundant API call)

    Returns a DataFrame with columns:
        contractSymbol, strike, bid, ask, lastPrice, volume, openInterest,
        impliedVolatility, optionType, spot, T, delta, gamma, theta, vega,
        rho, bsPrice, midPrice, moneyness, moneyness_label
    """
    if ticker_obj is None:
        ticker_obj = yf.Ticker(symbol)

    if spot is None:
        try:
            spot = ticker_obj.fast_info["lastPrice"]
        except Exception:
            spot = 0

    chain = ticker_obj.option_chain(expiration)

    exp_date = datetime.strptime(expiration, "%Y-%m-%d")
    today = datetime.now()
    days_to_exp = max((exp_date - today).days, 1)
    T = days_to_exp / 365.0

    results = []
    for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
        df = df.copy()
        df["optionType"] = opt_type
        df["spot"] = spot
        df["T"] = T
        df["dte"] = days_to_exp

        # Use yfinance IV; replace NaN/zero with 0.30 fallback
        sigma = df["impliedVolatility"].fillna(0.30).values
        sigma = np.where(sigma < 0.01, 0.30, sigma)
        K = df["strike"].values

        # Compute Greeks
        greeks = bs_greeks(spot, K, T, risk_free_rate, sigma, option_type=opt_type)
        df["delta"] = greeks["delta"]
        df["gamma"] = greeks["gamma"]
        df["theta"] = greeks["theta"]
        df["vega"] = greeks["vega"]
        df["rho"] = greeks["rho"]

        # BS theoretical price
        df["bsPrice"] = bs_price(spot, K, T, risk_free_rate, sigma, option_type=opt_type)

        # Mid price (average of bid/ask)
        df["midPrice"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2

        # Moneyness
        df["moneyness"] = df["strike"] / spot
        df["moneyness_label"] = "ATM"
        if opt_type == "call":
            df.loc[df["strike"] < spot * 0.97, "moneyness_label"] = "ITM"
            df.loc[df["strike"] > spot * 1.03, "moneyness_label"] = "OTM"
        else:
            df.loc[df["strike"] > spot * 1.03, "moneyness_label"] = "ITM"
            df.loc[df["strike"] < spot * 0.97, "moneyness_label"] = "OTM"

        results.append(df)

    combined = pd.concat(results, ignore_index=True)
    info = {"symbol": symbol, "spot": spot, "name": symbol}
    return combined, info


def compute_iv_percentile(symbol, lookback_days=90):
    """
    Compute where current IV sits vs. historical realized volatility.

    Returns dict with:
        current_iv (avg ATM IV), historical_rv, iv_percentile, iv_rank_label
    """
    ticker = yf.Ticker(symbol)

    # Get historical prices for realized vol
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 30)
    hist = ticker.history(start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"))

    if hist.empty or len(hist) < 20:
        return None

    # Realized volatility (annualized std of log returns)
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    rv_30 = log_returns.tail(30).std() * np.sqrt(252)
    rv_60 = log_returns.tail(60).std() * np.sqrt(252) if len(log_returns) >= 60 else rv_30
    rv_90 = log_returns.std() * np.sqrt(252)

    # Rolling 20-day realized vol for percentile calculation
    rolling_rv = log_returns.rolling(20).std() * np.sqrt(252)
    rolling_rv = rolling_rv.dropna()

    # Get current ATM IV from nearest expiration
    expirations = list(ticker.options)
    if not expirations:
        return None

    # Pick expiration closest to 30 DTE
    target_dte = 30
    best_exp = min(expirations, key=lambda e: abs(
        (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days - target_dte
    ))

    try:
        chain = ticker.option_chain(best_exp)
        spot = ticker.fast_info["lastPrice"]

        # ATM calls (within 3% of spot)
        atm_calls = chain.calls[
            abs(chain.calls["strike"] / spot - 1.0) < 0.03
        ]
        if atm_calls.empty:
            atm_calls = chain.calls.iloc[
                (chain.calls["strike"] - spot).abs().argsort()[:3]
            ]

        current_iv = atm_calls["impliedVolatility"].mean()
    except Exception:
        current_iv = rv_30  # fallback

    # IV percentile: where does current IV sit vs. rolling RV distribution
    if len(rolling_rv) > 0:
        iv_percentile = (rolling_rv < current_iv).mean() * 100
    else:
        iv_percentile = 50.0

    # Label
    if iv_percentile >= 70:
        label = "HIGH"
    elif iv_percentile <= 30:
        label = "LOW"
    else:
        label = "MODERATE"

    return {
        "current_iv": current_iv,
        "rv_30": rv_30,
        "rv_60": rv_60,
        "rv_90": rv_90,
        "iv_percentile": iv_percentile,
        "iv_rank_label": label,
        "lookback_days": lookback_days,
        "expiration_used": best_exp,
    }
