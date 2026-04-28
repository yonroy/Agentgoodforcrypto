import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openai import OpenAI
from src.Agent3.AnalysisData import run_analysis, format_analysis_report, ai_interpret_analysis
from src.Agent3.ReadNews import run_news_analysis, format_news_report, ai_summarize_sentiment
from src.tracker import CostTracker, AccuracyTracker

client = OpenAI()
ORCHESTRA_MODEL = os.getenv("ORCHESTRA_MODEL", "gpt-3.5-turbo")
ORCHESTRA_TEMPERATURE = float(os.getenv("ORCHESTRA_MODEL_TEMPERATURE", "0.7"))
ORCHESTRA_MAX_TOKENS = int(os.getenv("ORCHESTRA_MODEL_MAX_TOKENS", "1500"))

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "..", "SystemPrompt", "orchestrator_prompt.txt")
with open(PROMPT_FILE, "r", encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read()

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "memory.json")
MAX_MEMORY = 10

# Shared trackers
cost_tracker = CostTracker()
accuracy_tracker = AccuracyTracker()


def load_memory() -> list[dict]:
    if os.path.exists(MEMORY_FILE) and os.path.getsize(MEMORY_FILE) > 0:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_memory(memory: list[dict]):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    memory = memory[-MAX_MEMORY:]
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2, default=str)


def synthesize_results(tech_analysis: dict, news_analysis: dict, memory: list[dict]) -> dict:
    """Combine technical + sentiment signals into a final trend."""
    overall_tech = tech_analysis.get("overall_trend", "NEUTRAL")
    sentiment_signal = news_analysis.get("signal", "NEUTRAL")
    fng = news_analysis.get("fear_and_greed", {})
    fng_value = fng.get("current_value", 50)

    score = 0
    if overall_tech == "BULLISH":
        score += 2
    elif overall_tech == "BEARISH":
        score -= 2

    if sentiment_signal in ("STRONG_BUY_SIGNAL", "BUY_SIGNAL"):
        score += 1
    elif sentiment_signal in ("STRONG_SELL_SIGNAL", "SELL_SIGNAL"):
        score -= 1

    reversal_signal = "STABLE"
    if len(memory) >= 2:
        prev_trends = [m.get("final_trend") for m in memory[-3:]]
        if all(t == "BULLISH" for t in prev_trends) and score < 0:
            reversal_signal = "REVERSAL_DOWN"
        elif all(t == "BEARISH" for t in prev_trends) and score > 0:
            reversal_signal = "REVERSAL_UP"

    if score >= 2:
        final_trend = "BULLISH"
    elif score <= -2:
        final_trend = "BEARISH"
    elif score > 0:
        final_trend = "SLIGHTLY_BULLISH"
    elif score < 0:
        final_trend = "SLIGHTLY_BEARISH"
    else:
        final_trend = "NEUTRAL"

    # Hard-coded oversold reversal rule:
    # When RSI is extremely low (< 20) on medium timeframes and all signals
    # point BEARISH, the LLM tends to blindly confirm BEARISH — but classic TA
    # says this is a high-probability bounce zone.  Downgrade to NEUTRAL.
    oversold_flag = None
    rsi_values = []
    for tf in ("15m", "30m"):
        tf_data = tech_analysis.get("timeframes", {}).get(tf, {})
        rsi = tf_data.get("rsi")
        if rsi is not None:
            rsi_values.append(rsi)

    if rsi_values and min(rsi_values) < 20 and overall_tech == "BEARISH":
        oversold_flag = "OVERSOLD_REVERSAL_RISK"
        if final_trend == "BEARISH":
            final_trend = "NEUTRAL"
        elif final_trend == "SLIGHTLY_BEARISH":
            final_trend = "NEUTRAL"

    return {
        "final_trend": final_trend,
        "confidence_score": score,
        "technical_trend": overall_tech,
        "sentiment_signal": sentiment_signal,
        "fear_greed_value": fng_value,
        "reversal_signal": reversal_signal,
        "oversold_flag": oversold_flag,
    }


def build_signal_summary(tech_analysis: dict) -> str:
    """Pre-compute per-timeframe signal counts so the LLM doesn't have to.

    Returns a structured text block listing bullish/bearish indicators per
    timeframe and flagging any mixed-signal timeframes the LLM must address.
    """
    lines = ["--- Bảng tóm tắt chỉ báo (đã tính sẵn, KHÔNG ĐƯỢC thay đổi) ---"]

    total_bullish = 0
    total_bearish = 0
    mixed_timeframes = []

    for tf, data in tech_analysis.get("timeframes", {}).items():
        if "error" in data:
            continue

        price = data["current_price"]
        sma = data.get("sma", {})
        rsi = data.get("rsi")
        macd = data.get("macd")

        bullish = 0
        bearish = 0
        details = []

        # SMA signals
        for sma_name, sma_val in sma.items():
            if sma_val is not None:
                if price > sma_val:
                    bullish += 1
                    details.append(f"{sma_name}: BULLISH (price > {sma_val:.2f})")
                else:
                    bearish += 1
                    details.append(f"{sma_name}: BEARISH (price < {sma_val:.2f})")

        # RSI signal
        if rsi is not None:
            if rsi < 32:
                bullish += 1  # contrarian: oversold = bullish signal
                details.append(f"RSI: OVERSOLD ({rsi:.2f}) -> bullish contrarian")
            elif rsi > 68:
                bearish += 1  # contrarian: overbought = bearish signal
                details.append(f"RSI: OVERBOUGHT ({rsi:.2f}) -> bearish contrarian")
            else:
                details.append(f"RSI: NEUTRAL ({rsi:.2f})")

        # MACD signal
        if macd and macd.get("histogram") is not None:
            hist = macd["histogram"]
            if hist > 0:
                bullish += 1
                details.append(f"MACD Hist: BULLISH (+{hist})")
            else:
                bearish += 1
                details.append(f"MACD Hist: BEARISH ({hist})")

        total_bullish += bullish
        total_bearish += bearish

        # Flag mixed signals
        if bullish > 0 and bearish > 0:
            mixed_timeframes.append(tf)

        trend_label = data.get("trend", "N/A")
        lines.append(f"[{tf}] Bullish: {bullish}, Bearish: {bearish}, Trend: {trend_label}")
        for d in details:
            lines.append(f"    {d}")

    lines.append(f"\nTOTAL: Bullish={total_bullish}, Bearish={total_bearish}")

    if mixed_timeframes:
        lines.append(f"MIXED SIGNALS tại: {', '.join(mixed_timeframes)} — BẮT BUỘC phải đề cập trong phân tích")
    else:
        lines.append("Không có mixed signals — tất cả timeframe đồng thuận.")

    return "\n".join(lines)


def ai_synthesize(tech_report: str, tech_ai: str, news_report: str, news_ai: str,
                  synthesis: dict, tech_analysis: dict, memory: list[dict]) -> tuple[str, dict]:
    """Use OpenAI orchestrator to produce the final synthesis."""
    memory_summary = ""
    if memory:
        recent = memory[-5:]
        memory_lines = [f"  R{m.get('round', '?')}: {m['final_trend']} (confidence={m['confidence']})" for m in recent]
        memory_summary = "Lịch sử gần đây:\n" + "\n".join(memory_lines)

    signal_summary = build_signal_summary(tech_analysis)

    # Format tech_ai: if it's a dict (structured JSON from analytics agent),
    # serialize it so the orchestrator sees structured data, not prose.
    if isinstance(tech_ai, dict):
        tech_ai_text = json.dumps(tech_ai, ensure_ascii=False, indent=2)
    else:
        tech_ai_text = str(tech_ai)

    user_content = (
        f"{signal_summary}\n\n"
        f"--- Phân tích kỹ thuật (structured) ---\n{tech_ai_text}\n\n"
        f"--- Phân tích tâm lý ---\n{news_ai}\n\n"
        f"--- Signal tổng hợp ---\n"
        f"Trend: {synthesis['final_trend']}, Score: {synthesis['confidence_score']}, "
        f"Reversal Signal: {synthesis['reversal_signal']}"
        f"{', Oversold Flag: ' + synthesis['oversold_flag'] if synthesis.get('oversold_flag') else ''}\n\n"
        f"{memory_summary}"
    )

    response = client.chat.completions.create(
        model=ORCHESTRA_MODEL,
        temperature=ORCHESTRA_TEMPERATURE,
        max_tokens=ORCHESTRA_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    content = response.choices[0].message.content
    usage = {
        "model": ORCHESTRA_MODEL,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    return content, usage


def run_orchestrator(symbol: str, round_number: int = 0) -> dict:
    """Main orchestrator flow:

    1. Load memory (10 most recent)
    2. Evaluate previous prediction accuracy
    3. Run AnalysisData + ReadNews agents (with AI interpretation)
    4. Synthesize results (with AI orchestrator)
    5. Track cost & accuracy
    6. Save to memory
    """
    timestamp = datetime.now().isoformat()
    memory = load_memory()

    # Step 2: Run AnalysisData agent FIRST to get the canonical price
    print(f"[{timestamp}] Running AnalysisData for {symbol}...")
    tech_analysis = run_analysis(symbol)

    # Use the 1m candle close as the single source of truth for price.
    # This is the same price shown in the report, so accuracy % is verifiable.
    current_price = tech_analysis["timeframes"]["1m"]["current_price"]

    # Evaluate previous round's prediction using the canonical price
    prev_eval = accuracy_tracker.evaluate_previous(current_price)
    if prev_eval:
        print(f"  Previous prediction: {prev_eval['predicted_trend']} -> "
              f"Actual: {prev_eval['actual_direction']} "
              f"({'OK' if prev_eval['is_correct'] else 'MISS'})")

    tech_report = format_analysis_report(tech_analysis)
    tech_ai_commentary, tech_usage = ai_interpret_analysis(tech_analysis)
    cost_tracker.record_call("AnalysisData", tech_usage["model"],
                             tech_usage["prompt_tokens"], tech_usage["completion_tokens"])

    # Step 3: Run ReadNews agent
    print(f"[{timestamp}] Running ReadNews...")
    news_analysis = run_news_analysis()
    news_report = format_news_report(news_analysis)
    news_ai_summary, news_usage = ai_summarize_sentiment(news_analysis)
    cost_tracker.record_call("ReadNews", news_usage["model"],
                             news_usage["prompt_tokens"], news_usage["completion_tokens"])

    # Step 4: Synthesize
    synthesis = synthesize_results(tech_analysis, news_analysis, memory)
    orchestrator_ai, orch_usage = ai_synthesize(
        tech_report, tech_ai_commentary, news_report, news_ai_summary,
        synthesis, tech_analysis, memory
    )
    cost_tracker.record_call("Orchestrator", orch_usage["model"],
                             orch_usage["prompt_tokens"], orch_usage["completion_tokens"])

    # Step 5: Track cost & accuracy
    # Dynamic neutral band: 30% of 24h ATR as percentage, floor 0.25%
    ticker_24h = tech_analysis.get("ticker_24h", {})
    high = ticker_24h.get("high", 0)
    low = ticker_24h.get("low", 0)
    last = ticker_24h.get("last_price", 0)
    if last > 0 and high > 0 and low > 0:
        atr_24h_pct = ((high - low) / last) * 100
        neutral_band_pct = max(0.25, 0.3 * atr_24h_pct)
    else:
        neutral_band_pct = 0.25

    round_cost = cost_tracker.finish_round(round_number)
    accuracy_tracker.record_prediction(round_number, symbol,
                                       synthesis["final_trend"], current_price,
                                       neutral_band_pct=round(neutral_band_pct, 4))

    result = {
        "timestamp": timestamp,
        "round": round_number,
        "symbol": symbol,
        "current_price": current_price,
        "technical": tech_analysis,
        "news": news_analysis,
        "synthesis": synthesis,
        "tech_report": tech_report,
        "tech_ai": tech_ai_commentary,
        "news_report": news_report,
        "news_ai": news_ai_summary,
        "orchestrator_ai": orchestrator_ai,
        "cost": round_cost,
        "accuracy": accuracy_tracker.get_accuracy_stats(),
    }

    # Step 6: Save to memory
    memory_entry = {
        "timestamp": timestamp,
        "round": round_number,
        "symbol": symbol,
        "price": current_price,
        "final_trend": synthesis["final_trend"],
        "confidence": synthesis["confidence_score"],
        "technical_trend": synthesis["technical_trend"],
        "sentiment_signal": synthesis["sentiment_signal"],
        "fear_greed": synthesis["fear_greed_value"],
        "oversold_flag": synthesis.get("oversold_flag"),
    }
    memory.append(memory_entry)
    save_memory(memory)

    return result


def _format_tech_ai(tech_ai) -> str:
    """Render structured tech AI result into readable text."""
    if isinstance(tech_ai, str):
        return tech_ai

    lines = []
    # Per-timeframe analysis
    for tf, info in tech_ai.get("timeframes", {}).items():
        trend = info.get("trend", "N/A")
        summary = info.get("summary", "")
        lines.append(f"  [{tf}] {trend}: {summary}")
        levels = info.get("key_levels", {})
        support = levels.get("support")
        resistance = levels.get("resistance")
        if support or resistance:
            lines.append(f"    Support: {support or 'N/A'} | Resistance: {resistance or 'N/A'}")

    # Scenarios
    scenarios = tech_ai.get("scenarios", [])
    if scenarios:
        lines.append("")
        for sc in scenarios:
            lines.append(f"  {sc.get('id', '?')}: {sc.get('description', '')} ({sc.get('probability', '')})")

    # Overall
    overall = tech_ai.get("overall_summary", "")
    if overall:
        lines.append(f"\n  Overall: {overall}")

    return "\n".join(lines)


def format_final_report(result: dict) -> str:
    """Format the complete orchestrator output into a Telegram-ready message."""
    s = result["synthesis"]
    symbol = result["symbol"]
    ts = result["timestamp"]
    r = result["round"]

    tech_ai_text = _format_tech_ai(result["tech_ai"])

    lines = [
        f"{'='*35}",
        f"  CRYPTO ANALYSIS: {symbol}",
        f"  Round #{r} | {ts}",
        f"  Price: {result['current_price']}",
        f"{'='*35}",
        "",
        result["tech_report"],
        "",
        f"--- AI Technical Commentary ---",
        tech_ai_text,
        "",
        result["news_report"],
        "",
        f"--- AI Sentiment Summary ---",
        result["news_ai"],
        "",
        f"{'='*35}",
        f"  ORCHESTRATOR VERDICT",
        f"{'='*35}",
        result["orchestrator_ai"],
        "",
        f"{'='*35}",
        f"  SIGNALS",
        f"{'='*35}",
        f"Trend: {s['final_trend']}",
        f"Confidence Score: {s['confidence_score']}",
        f"Technical: {s['technical_trend']}",
        f"Sentiment: {s['sentiment_signal']}",
        f"Fear & Greed: {s['fear_greed_value']}",
        f"Reversal Signal: {s['reversal_signal']}",
        f"Oversold Flag: {s['oversold_flag'] or 'None'}",
        "",
        cost_tracker.format_cost_report(result["cost"]),
        "",
        accuracy_tracker.format_accuracy_report(),
        f"{'='*35}",
    ]
    return "\n".join(lines)
