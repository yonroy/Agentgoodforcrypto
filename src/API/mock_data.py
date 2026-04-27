"""Mock data for testing without hitting real APIs.

Usage:
    Set USE_MOCK_DATA=true in .env to use mock data instead of real API calls.
"""

import random
import time
import math

# Base prices for mock symbols
_BASE_PRICES = {
    "BTCUSDT": 94500.0,
    "ETHUSDT": 1800.0,
    "BNBUSDT": 600.0,
    "SOLUSDT": 150.0,
    "XRPUSDT": 0.55,
}

# Track price drift across calls so mock data is somewhat realistic
_price_state: dict[str, float] = {}


def _get_base_price(symbol: str) -> float:
    return _BASE_PRICES.get(symbol.upper(), 100.0)


def _drift_price(symbol: str, volatility: float = 0.002) -> float:
    """Simulate small random price drift."""
    if symbol not in _price_state:
        _price_state[symbol] = _get_base_price(symbol)
    change = random.gauss(0, volatility)
    _price_state[symbol] *= (1 + change)
    return _price_state[symbol]


def mock_get_klines(symbol: str, interval: str, limit: int = 100) -> list[dict]:
    """Generate realistic-looking kline/candlestick mock data."""
    base = _get_base_price(symbol)
    now_ms = int(time.time() * 1000)

    interval_ms = {
        "1m": 60_000, "5m": 300_000,
        "15m": 900_000, "30m": 1_800_000,
    }.get(interval, 300_000)

    klines = []
    price = base * (1 + random.uniform(-0.03, 0.03))

    for i in range(limit):
        open_time = now_ms - (limit - i) * interval_ms
        # Random walk with slight upward/downward trend
        trend = math.sin(i / 20) * 0.001
        change = random.gauss(trend, 0.003)
        open_price = price
        close_price = open_price * (1 + change)
        high = max(open_price, close_price) * (1 + random.uniform(0, 0.002))
        low = min(open_price, close_price) * (1 - random.uniform(0, 0.002))
        volume = random.uniform(100, 5000)

        klines.append({
            "open_time": open_time,
            "open": round(open_price, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "close": round(close_price, 4),
            "volume": round(volume, 2),
            "close_time": open_time + interval_ms - 1,
        })
        price = close_price

    # Update global state to last close
    _price_state[symbol] = price
    return klines


def mock_get_current_price(symbol: str) -> float:
    """Return a mock current price with slight drift."""
    price = _drift_price(symbol)
    return round(price, 4)


def mock_get_ticker_24h(symbol: str) -> dict:
    """Generate mock 24h ticker data."""
    base = _drift_price(symbol)
    change_pct = random.uniform(-5, 5)
    return {
        "price_change_percent": round(change_pct, 2),
        "high": round(base * (1 + random.uniform(0.01, 0.05)), 4),
        "low": round(base * (1 - random.uniform(0.01, 0.05)), 4),
        "volume": round(random.uniform(10000, 100000), 2),
        "last_price": round(base, 4),
    }


def mock_get_fear_and_greed(limit: int = 10) -> list[dict]:
    """Generate mock Fear & Greed Index data."""
    classifications = {
        (0, 25): "Extreme Fear",
        (25, 40): "Fear",
        (40, 60): "Neutral",
        (60, 75): "Greed",
        (75, 101): "Extreme Greed",
    }

    base_value = random.randint(20, 80)
    results = []
    now = int(time.time())

    for i in range(limit):
        value = max(0, min(100, base_value + random.randint(-10, 10)))
        classification = "Neutral"
        for (lo, hi), name in classifications.items():
            if lo <= value < hi:
                classification = name
                break
        results.append({
            "value": value,
            "classification": classification,
            "timestamp": str(now - i * 86400),
        })

    return results


def mock_get_current_fng() -> dict:
    """Get mock current Fear & Greed value."""
    results = mock_get_fear_and_greed(limit=1)
    return results[0]
