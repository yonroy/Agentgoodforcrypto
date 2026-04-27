import os
import requests
from typing import Optional

USE_MOCK = os.getenv("USE_MOCK_DATA", "false").lower() == "true"

BINANCE_BASE_URL = "https://api.binance.com/api/v3"


def get_klines(symbol: str, interval: str, limit: int = 100) -> list[dict]:
    """Fetch kline/candlestick data from Binance API.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        interval: Kline interval (e.g., "1m", "5m", "15m", "30m")
        limit: Number of klines to return (max 1000)
    """
    if USE_MOCK:
        from src.API.mock_data import mock_get_klines
        return mock_get_klines(symbol, interval, limit)

    url = f"{BINANCE_BASE_URL}/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    klines = []
    for k in response.json():
        klines.append({
            "open_time": k[0],
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time": k[6],
        })
    return klines


def get_current_price(symbol: str) -> float:
    """Fetch current price for a symbol."""
    if USE_MOCK:
        from src.API.mock_data import mock_get_current_price
        return mock_get_current_price(symbol)

    url = f"{BINANCE_BASE_URL}/ticker/price"
    params = {"symbol": symbol.upper()}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return float(response.json()["price"])


def get_ticker_24h(symbol: str) -> dict:
    """Fetch 24h ticker statistics."""
    if USE_MOCK:
        from src.API.mock_data import mock_get_ticker_24h
        return mock_get_ticker_24h(symbol)

    url = f"{BINANCE_BASE_URL}/ticker/24hr"
    params = {"symbol": symbol.upper()}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return {
        "price_change_percent": float(data["priceChangePercent"]),
        "high": float(data["highPrice"]),
        "low": float(data["lowPrice"]),
        "volume": float(data["volume"]),
        "last_price": float(data["lastPrice"]),
    }
