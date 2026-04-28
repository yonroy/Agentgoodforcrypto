import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openai import OpenAI
from src.API.GetNews import get_fear_and_greed, get_current_fng

client = OpenAI()
NEWS_MODEL = os.getenv("NEWS_MODEL", "gpt-3.5-turbo")
NEWS_TEMPERATURE = float(os.getenv("NEWS_MODEL_TEMPERATURE", "0.7"))
NEWS_MAX_TOKENS = int(os.getenv("NEWS_MODEL_MAX_TOKENS", "1500"))

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "..", "SystemPrompt", "read_news_prompt.txt")
with open(PROMPT_FILE, "r", encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read()


def analyze_sentiment(fng_data: list[dict]) -> dict:
    """Analyze Fear & Greed Index trend over time."""
    if not fng_data:
        return {"sentiment": "Unknown", "trend": "N/A", "details": []}

    current = fng_data[0]
    values = [d["value"] for d in fng_data]

    # Current sentiment
    value = current["value"]
    if value <= 25:
        sentiment = "Extreme Fear"
    elif value <= 40:
        sentiment = "Fear"
    elif value <= 60:
        sentiment = "Neutral"
    elif value <= 75:
        sentiment = "Greed"
    else:
        sentiment = "Extreme Greed"

    # Trend direction — analyse the sequential movement, not just
    # block averages, so V-shape recoveries and sell-offs are detected.
    if len(values) >= 3:
        # values is ordered newest-first: values[0] = today
        # Compute short-term momentum (last 3 days, newest vs oldest)
        short_delta = values[0] - values[2]  # positive = improving

        # Check for V-shape: did the series dip then recover (or spike then fall)?
        min_val = min(values[:5]) if len(values) >= 5 else min(values)
        max_val = max(values[:5]) if len(values) >= 5 else max(values)

        # Current value is near the recent high and we climbed from a dip
        recovering = (values[0] >= max_val - 3) and (values[0] - min_val >= 10)
        # Current value is near the recent low and we fell from a peak
        falling = (values[0] <= min_val + 3) and (max_val - values[0] >= 10)

        if recovering and short_delta > 0:
            trend = "RECOVERING"
        elif falling and short_delta < 0:
            trend = "FALLING"
        elif short_delta > 5:
            trend = "IMPROVING"
        elif short_delta < -5:
            trend = "DECLINING"
        else:
            trend = "STABLE"
    else:
        trend = "INSUFFICIENT_DATA"

    return {
        "current_value": value,
        "sentiment": sentiment,
        "classification": current["classification"],
        "trend": trend,
        "avg_7d": round(sum(values[:min(7, len(values))]) / min(7, len(values)), 1),
        "history": fng_data[:7],
    }


def get_market_sentiment_signal(sentiment_data: dict) -> str:
    """Convert sentiment analysis into a trading signal."""
    value = sentiment_data.get("current_value", 50)
    trend = sentiment_data.get("trend", "STABLE")

    # Extreme fear = potential buying opportunity (contrarian)
    # Extreme greed = potential selling signal (contrarian)
    if value <= 20:
        signal = "STRONG_BUY_SIGNAL"
    elif value <= 35:
        signal = "BUY_SIGNAL"
    elif value >= 80:
        signal = "STRONG_SELL_SIGNAL"
    elif value >= 65:
        signal = "SELL_SIGNAL"
    else:
        signal = "NEUTRAL"

    return signal


def run_news_analysis() -> dict:
    """Run full news/sentiment analysis.

    Returns sentiment data with Fear & Greed index analysis.
    """
    fng_data = get_fear_and_greed(limit=10)
    sentiment = analyze_sentiment(fng_data)
    signal = get_market_sentiment_signal(sentiment)

    return {
        "fear_and_greed": sentiment,
        "signal": signal,
    }


def ai_summarize_sentiment(analysis: dict) -> tuple[str, dict]:
    """Use OpenAI to summarize and interpret market sentiment.

    Returns (ai_summary, usage_info).
    """
    report = format_news_report(analysis)

    response = client.chat.completions.create(
        model=NEWS_MODEL,
        temperature=NEWS_TEMPERATURE,
        max_tokens=NEWS_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": report},
        ],
    )

    content = response.choices[0].message.content
    usage = {
        "model": NEWS_MODEL,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    return content, usage


def format_news_report(analysis: dict) -> str:
    """Format news analysis into a readable report."""
    fng = analysis.get("fear_and_greed", {})
    lines = [
        "=== Market Sentiment Analysis ===\n",
        f"Fear & Greed Index: {fng.get('current_value', 'N/A')} ({fng.get('sentiment', 'N/A')})",
        f"Classification: {fng.get('classification', 'N/A')}",
        f"Trend: {fng.get('trend', 'N/A')}",
        f"7-day Average: {fng.get('avg_7d', 'N/A')}",
        f"\nSignal: {analysis.get('signal', 'N/A')}",
    ]

    history = fng.get("history", [])
    if history:
        lines.append("\nRecent History (7d):")
        for entry in history[:7]:
            lines.append(f"  {entry['classification']}: {entry['value']}")

    return "\n".join(lines)
