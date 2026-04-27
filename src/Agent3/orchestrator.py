import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openai import OpenAI
from src.API.GetDataCrypto import get_current_price
from src.Agent3.AnalysisData import run_analysis, format_analysis_report, ai_interpret_analysis
from src.Agent3.ReadNews import run_news_analysis, format_news_report, ai_summarize_sentiment
from src.tracker import CostTracker, AccuracyTracker

client = OpenAI()
ORCHESTRA_MODEL = os.getenv("ORCHESTRA_MODEL", "gpt-3.5-turbo")
ORCHESTRA_TEMPERATURE = float(os.getenv("ORCHESTRA_MODEL_TEMPERATURE", "0.7"))
ORCHESTRA_MAX_TOKENS = int(os.getenv("ORCHESTRA_MODEL_MAX_TOKENS", "1500"))

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "..", "Systemprompt", "orchestrator_prompt.txt")
with open(PROMPT_FILE, "r", encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read()

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "memory.json")
MAX_MEMORY = 10

# Shared trackers
cost_tracker = CostTracker()
accuracy_tracker = AccuracyTracker()


def load_memory() -> list[dict]:
    if os.path.exists(MEMORY_FILE):
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

    trend_change = "STABLE"
    if len(memory) >= 2:
        prev_trends = [m.get("final_trend") for m in memory[-3:]]
        if all(t == "BULLISH" for t in prev_trends) and score < 0:
            trend_change = "REVERSAL_DOWN"
        elif all(t == "BEARISH" for t in prev_trends) and score > 0:
            trend_change = "REVERSAL_UP"

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

    return {
        "final_trend": final_trend,
        "confidence_score": score,
        "technical_trend": overall_tech,
        "sentiment_signal": sentiment_signal,
        "fear_greed_value": fng_value,
        "trend_change": trend_change,
    }


def ai_synthesize(tech_report: str, tech_ai: str, news_report: str, news_ai: str,
                  synthesis: dict, memory: list[dict]) -> tuple[str, dict]:
    """Use OpenAI orchestrator to produce the final synthesis."""
    memory_summary = ""
    if memory:
        recent = memory[-5:]
        memory_lines = [f"  R{m.get('round', '?')}: {m['final_trend']} (confidence={m['confidence']})" for m in recent]
        memory_summary = "Lịch sử gần đây:\n" + "\n".join(memory_lines)

    user_content = (
        f"--- Phân tích kỹ thuật ---\n{tech_ai}\n\n"
        f"--- Phân tích tâm lý ---\n{news_ai}\n\n"
        f"--- Signal tổng hợp ---\n"
        f"Trend: {synthesis['final_trend']}, Score: {synthesis['confidence_score']}, "
        f"Trend Change: {synthesis['trend_change']}\n\n"
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

    # Get current price for accuracy evaluation
    current_price = get_current_price(symbol)

    # Evaluate previous round's prediction
    prev_eval = accuracy_tracker.evaluate_previous(current_price)
    if prev_eval:
        print(f"  Previous prediction: {prev_eval['predicted_trend']} → "
              f"Actual: {prev_eval['actual_direction']} "
              f"({'OK' if prev_eval['is_correct'] else 'MISS'})")

    # Step 2: Run AnalysisData agent
    print(f"[{timestamp}] Running AnalysisData for {symbol}...")
    tech_analysis = run_analysis(symbol)
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
        tech_report, tech_ai_commentary, news_report, news_ai_summary, synthesis, memory
    )
    cost_tracker.record_call("Orchestrator", orch_usage["model"],
                             orch_usage["prompt_tokens"], orch_usage["completion_tokens"])

    # Step 5: Track cost & accuracy
    round_cost = cost_tracker.finish_round(round_number)
    accuracy_tracker.record_prediction(round_number, symbol,
                                       synthesis["final_trend"], current_price)

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
    }
    memory.append(memory_entry)
    save_memory(memory)

    return result


def format_final_report(result: dict) -> str:
    """Format the complete orchestrator output into a Telegram-ready message."""
    s = result["synthesis"]
    symbol = result["symbol"]
    ts = result["timestamp"]
    r = result["round"]

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
        result["tech_ai"],
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
        f"Trend Change: {s['trend_change']}",
        "",
        cost_tracker.format_cost_report(result["cost"]),
        "",
        accuracy_tracker.format_accuracy_report(),
        f"{'='*35}",
    ]
    return "\n".join(lines)
