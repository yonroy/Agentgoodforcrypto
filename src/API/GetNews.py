import os
import requests

USE_MOCK = os.getenv("USE_MOCK_DATA", "false").lower() == "true"

FNG_API_URL = "https://api.alternative.me/fng/"


def get_fear_and_greed(limit: int = 10) -> list[dict]:
    """Fetch Fear & Greed Index data.

    Args:
        limit: Number of data points to return
    Returns:
        List of dicts with value, classification, and timestamp
    """
    if USE_MOCK:
        from src.API.mock_data import mock_get_fear_and_greed
        return mock_get_fear_and_greed(limit)

    params = {"limit": limit, "format": "json"}
    response = requests.get(FNG_API_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = []
    for entry in data.get("data", []):
        results.append({
            "value": int(entry["value"]),
            "classification": entry["value_classification"],
            "timestamp": entry["timestamp"],
        })
    return results


def get_current_fng() -> dict:
    """Get current Fear & Greed Index value."""
    if USE_MOCK:
        from src.API.mock_data import mock_get_current_fng
        return mock_get_current_fng()

    results = get_fear_and_greed(limit=1)
    if results:
        return results[0]
    return {"value": 0, "classification": "Unknown", "timestamp": "0"}
