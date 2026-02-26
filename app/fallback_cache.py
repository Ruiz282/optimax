"""
Fallback Cache — Disk-based cache for offline resilience.

When yfinance calls fail (rate limit, network error), the app can
show the last successful data instead of an error message.
"""

import json
import os
import pickle
from datetime import datetime, timedelta


CACHE_DIR = os.path.join(os.path.dirname(__file__), "user_data", "cache")


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def save_fallback(key: str, data):
    """Save data to disk as a pickled fallback.
    Works with any Python object (DataFrames, dataclasses, dicts, etc.)."""
    _ensure_dir()
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    meta_path = os.path.join(CACHE_DIR, f"{key}.meta")
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        with open(meta_path, "w") as f:
            json.dump({"saved_at": datetime.now().isoformat()}, f)
    except Exception:
        pass  # silently fail — fallback cache is best-effort


def load_fallback(key: str, max_age_hours: int = 24):
    """Load fallback data if it exists and isn't too old.

    Returns (data, age_minutes) or (None, None) if no valid cache exists.
    """
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    meta_path = os.path.join(CACHE_DIR, f"{key}.meta")

    if not os.path.exists(path) or not os.path.exists(meta_path):
        return None, None

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        saved_at = datetime.fromisoformat(meta["saved_at"])
        age = datetime.now() - saved_at

        if age > timedelta(hours=max_age_hours):
            return None, None

        with open(path, "rb") as f:
            data = pickle.load(f)

        age_minutes = int(age.total_seconds() / 60)
        return data, age_minutes
    except Exception:
        return None, None
