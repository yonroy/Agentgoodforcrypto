import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openai import OpenAI
from src.API.GetDataCrypto import get_klines, get_current_price, get_ticker_24h

client = OpenAI()
ANALYTICS_MODEL = os.getenv("ANALYTICS_MODEL", "gpt-3.5-turbo")
ANALYTICS_TEMPERATURE = float(os.getenv("ANALYTICS_MODEL_TEMPERATURE", "0.7"))
ANALYTICS_MAX_TOKENS = int(os.getenv("ANALYTICS_MODEL_MAX_TOKENS", "1500"))

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "..", "SystemPrompt", "analysis_data_prompt.txt")
with open(PROMPT_FILE, "r", encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read()

# Timeframe configs from diagram:
# Khung    | SMA      | RSI | MACD
# 1m-5m    | 5,10,20  | 14  | 12,26,9
# 15m      | 7,25,50  | 14  | 12,26,9
# 30m      | 7,14,30  | 14  | 12,26,9

TIMEFRAME_CONFIG = {
    "1m": {"sma_periods": [5, 10, 20], "rsi_period": 14, "macd": (12, 26, 9)},
    "5m": {"sma_periods": [5, 10, 20], "rsi_period": 14, "macd": (12, 26, 9)},
    "15m": {"sma_periods": [7, 25, 50], "rsi_period": 14, "macd": (12, 26, 9)},
    "30m": {"sma_periods": [7, 14, 30], "rsi_period": 14, "macd": (12, 26, 9)},
}


def calculate_sma(closes: list[float], period: int) -> float | None:
    """Calculate Simple Moving Average."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def calculate_rsi(closes: list[float], period: int = 14) -> float | None:
    """Calculate Relative Strength Index."""
    if len(closes) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(-period, 0):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_ema(closes: list[float], period: int) -> float | None:
    """Calculate Exponential Moving Average."""
    if len(closes) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict | None:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    if len(closes) < slow + signal:
        return None

    ema_fast = calculate_ema(closes, fast)
    ema_slow = calculate_ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None

    macd_line = ema_fast - ema_slow

    # Calculate signal line using MACD values over recent periods
    macd_values = []
    for i in range(signal + slow, len(closes) + 1):
        subset = closes[:i]
        ef = calculate_ema(subset, fast)
        es = calculate_ema(subset, slow)
        if ef is not None and es is not None:
            macd_values.append(ef - es)

    if len(macd_values) < signal:
        return {"macd": macd_line, "signal": 0, "histogram": macd_line}

    signal_line = sum(macd_values[-signal:]) / signal
    histogram = macd_line - signal_line

    return {"macd": round(macd_line, 6), "signal": round(signal_line, 6), "histogram": round(histogram, 6)}


def analyze_timeframe(symbol: str, interval: str) -> dict:
    """Analyze a single timeframe with SMA, RSI, MACD indicators."""
    config = TIMEFRAME_CONFIG.get(interval, TIMEFRAME_CONFIG["5m"])
    klines = get_klines(symbol, interval, limit=100)
    closes = [k["close"] for k in klines]

    # SMA
    sma_values = {}
    for period in config["sma_periods"]:
        val = calculate_sma(closes, period)
        sma_values[f"SMA_{period}"] = round(val, 4) if val else None

    # RSI
    rsi = calculate_rsi(closes, config["rsi_period"])

    # MACD
    fast, slow, sig = config["macd"]
    macd = calculate_macd(closes, fast, slow, sig)

    current_price = closes[-1] if closes else 0

    # Determine trend
    trend = determine_trend(current_price, sma_values, rsi, macd)

    return {
        "timeframe": interval,
        "current_price": round(current_price, 4),
        "sma": sma_values,
        "rsi": round(rsi, 2) if rsi else None,
        "macd": macd,
        "trend": trend,
    }


def determine_trend(price: float, sma: dict, rsi: float | None, macd: dict | None) -> str:
    """Determine trend based on indicators."""
    signals = []

    # SMA signals
    sma_vals = [v for v in sma.values() if v is not None]
    if sma_vals:
        above_count = sum(1 for v in sma_vals if price > v)
        if above_count == len(sma_vals):
            signals.append("BULLISH")
        elif above_count == 0:
            signals.append("BEARISH")
        else:
            signals.append("NEUTRAL")

    # RSI signals
    if rsi is not None:
        if rsi > 70:
            signals.append("OVERBOUGHT")
        elif rsi < 30:
            signals.append("OVERSOLD")
        elif rsi > 55:
            signals.append("BULLISH")
        elif rsi < 45:
            signals.append("BEARISH")
        else:
            signals.append("NEUTRAL")

    # MACD signals
    if macd and macd.get("histogram") is not None:
        if macd["histogram"] > 0:
            signals.append("BULLISH")
        else:
            signals.append("BEARISH")

    bullish = signals.count("BULLISH") + signals.count("OVERSOLD")
    bearish = signals.count("BEARISH") + signals.count("OVERBOUGHT")

    if bullish > bearish:
        return "BULLISH"
    elif bearish > bullish:
        return "BEARISH"
    return "NEUTRAL"


def run_analysis(symbol: str) -> dict:
    """Run full analysis across all timeframes for a symbol.

    Returns analysis results with trends and technical indicators.
    """
    results = {}
    for interval in TIMEFRAME_CONFIG:
        try:
            results[interval] = analyze_timeframe(symbol, interval)
        except Exception as e:
            results[interval] = {"timeframe": interval, "error": str(e)}

    # Get 24h stats
    try:
        ticker = get_ticker_24h(symbol)
    except Exception:
        ticker = {}

    # Overall trend summary
    trends = [r.get("trend") for r in results.values() if "trend" in r]
    bullish_count = trends.count("BULLISH")
    bearish_count = trends.count("BEARISH")

    if bullish_count > bearish_count:
        overall = "BULLISH"
    elif bearish_count > bullish_count:
        overall = "BEARISH"
    else:
        overall = "NEUTRAL"

    return {
        "symbol": symbol,
        "ticker_24h": ticker,
        "timeframes": results,
        "overall_trend": overall,
    }


def ai_interpret_analysis(analysis: dict) -> tuple[str, dict]:
    """Use OpenAI to interpret technical analysis data.

    Returns (ai_commentary, usage_info).
    """
    report = format_analysis_report(analysis)

    response = client.chat.completions.create(
        model=ANALYTICS_MODEL,
        temperature=ANALYTICS_TEMPERATURE,
        max_tokens=ANALYTICS_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": report},
        ],
    )

    content = response.choices[0].message.content
    usage = {
        "model": ANALYTICS_MODEL,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    return content, usage


def format_analysis_report(analysis: dict) -> str:
    """Format analysis results into a readable report."""
    symbol = analysis["symbol"]
    lines = [f"=== {symbol} Technical Analysis ===\n"]

    ticker = analysis.get("ticker_24h", {})
    if ticker:
        lines.append(f"24h Change: {ticker.get('price_change_percent', 'N/A')}%")
        lines.append(f"24h High/Low: {ticker.get('high', 'N/A')} / {ticker.get('low', 'N/A')}")
        lines.append(f"Volume: {ticker.get('volume', 'N/A')}\n")

    for interval, data in analysis.get("timeframes", {}).items():
        if "error" in data:
            lines.append(f"[{interval}] Error: {data['error']}")
            continue
        lines.append(f"--- {interval} ---")
        lines.append(f"  Price: {data['current_price']}")
        lines.append(f"  SMA: {data['sma']}")
        lines.append(f"  RSI: {data['rsi']}")
        if data['macd']:
            lines.append(f"  MACD: {data['macd']['macd']} | Signal: {data['macd']['signal']} | Hist: {data['macd']['histogram']}")
        lines.append(f"  Trend: {data['trend']}\n")

    lines.append(f">>> Overall Trend: {analysis['overall_trend']} <<<")
    return "\n".join(lines)
