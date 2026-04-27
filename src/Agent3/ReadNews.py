import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openai import OpenAI
from src.API.GetNews import get_fear_and_greed, get_current_fng

client = OpenAI()
NEWS_MODEL = os.getenv("NEWS_MODEL", "gpt-3.5-turbo")
NEWS_TEMPERATURE = float(os.getenv("NEWS_MODEL_TEMPERATURE", "0.7"))
NEWS_MAX_TOKENS = int(os.getenv("NEWS_MODEL_MAX_TOKENS", "1500"))


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

    # Trend direction
    if len(values) >= 3:
        recent_avg = sum(values[:3]) / 3
        older_avg = sum(values[3:min(6, len(values))]) / max(1, min(3, len(values) - 3))
        if recent_avg > older_avg + 5:
            trend = "IMPROVING"
        elif recent_avg < older_avg - 5:
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
    prompt = (
        "Bạn là chuyên gia phân tích tâm lý thị trường crypto. "
        "Dựa trên dữ liệu Fear & Greed Index dưới đây, "
        "hãy đọc, phân tích và tóm tắt thông tin quan trọng. "
        "Kết quả: Summary ngắn gọn về tâm lý thị trường.\n\n"
        f"{report}"
    )

    response = client.chat.completions.create(
        model=NEWS_MODEL,
        temperature=NEWS_TEMPERATURE,
        max_tokens=NEWS_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
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
        lines.append("\nRecent History:")
        for entry in history[:5]:
            lines.append(f"  {entry['classification']}: {entry['value']}")

    return "\n".join(lines)
