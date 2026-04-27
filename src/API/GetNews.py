import requests


FNG_API_URL = "https://api.alternative.me/fng/"


def get_fear_and_greed(limit: int = 10) -> list[dict]:
    """Fetch Fear & Greed Index data.

    Args:
        limit: Number of data points to return
    Returns:
        List of dicts with value, classification, and timestamp
    """
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
    results = get_fear_and_greed(limit=1)
    if results:
        return results[0]
    return {"value": 0, "classification": "Unknown", "timestamp": "0"}
