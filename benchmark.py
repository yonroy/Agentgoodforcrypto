"""
Benchmark: Evaluate and compare different model configurations.

Runs the same mock data through multiple model setups,
measures cost, latency, response quality, and prediction consistency.

Usage:
    python benchmark.py
"""

import os
import sys
import time
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Force mock data for benchmarking
os.environ["USE_MOCK_DATA"] = "true"

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from src.API.GetDataCrypto import get_klines, get_current_price, get_ticker_24h
from src.API.GetNews import get_fear_and_greed
from src.Agent3.AnalysisData import run_analysis, format_analysis_report
from src.Agent3.ReadNews import run_news_analysis, format_news_report
from src.tracker import MODEL_PRICING

client = OpenAI()

# ============================================================
#  Model configurations to benchmark
# ============================================================
BENCHMARK_CONFIGS = [
    {
        "name": "Budget (all gpt-4o-mini)",
        "orchestrator": {"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 1500},
        "analytics": {"model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 1500},
        "news": {"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 1500},
    },
    {
        "name": "Balanced (gpt-4o + gpt-4o-mini)",
        "orchestrator": {"model": "gpt-4o", "temperature": 0.5, "max_tokens": 1500},
        "analytics": {"model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 1500},
        "news": {"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 1500},
    },
    {
        "name": "Quality (all gpt-4o)",
        "orchestrator": {"model": "gpt-4o", "temperature": 0.5, "max_tokens": 2000},
        "analytics": {"model": "gpt-4o", "temperature": 0.2, "max_tokens": 1500},
        "news": {"model": "gpt-4o", "temperature": 0.5, "max_tokens": 1500},
    },
]

SYMBOL = os.getenv("CRYPTO_SYMBOL", "BTCUSDT")
BENCHMARK_ROUNDS = int(os.getenv("BENCHMARK_ROUNDS", "3"))

# Load system prompts
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "src", "SystemPrompt")

with open(os.path.join(PROMPT_DIR, "analysis_data_prompt.txt"), "r", encoding="utf-8") as f:
    ANALYSIS_SYSTEM_PROMPT = f.read()
with open(os.path.join(PROMPT_DIR, "orchestrator_prompt.txt"), "r", encoding="utf-8") as f:
    ORCHESTRATOR_SYSTEM_PROMPT = f.read()
with open(os.path.join(PROMPT_DIR, "read_news_prompt.txt"), "r", encoding="utf-8") as f:
    NEWS_SYSTEM_PROMPT = f.read()


# ============================================================
#  Benchmark runner
# ============================================================

def call_model(system_prompt: str, user_content: str,
               model: str, temperature: float, max_tokens: int) -> dict:
    """Call OpenAI and return response + metrics."""
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    latency = time.time() - start

    content = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    pricing = MODEL_PRICING.get(model, {"input": 0.01, "output": 0.03})
    cost = (prompt_tokens / 1000) * pricing["input"] + \
           (completion_tokens / 1000) * pricing["output"]

    return {
        "content": content,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": round(cost, 6),
        "latency_s": round(latency, 2),
        "response_length": len(content),
    }


def evaluate_response_quality(content: str) -> dict:
    """Score response quality based on structural criteria."""
    score = 0
    checks = {}

    # Has trend keyword?
    trend_keywords = ["BULLISH", "BEARISH", "NEUTRAL", "tăng", "giảm", "sideway",
                      "xu hướng", "trend"]
    has_trend = any(kw.lower() in content.lower() for kw in trend_keywords)
    checks["has_trend"] = has_trend
    if has_trend:
        score += 25

    # Has data evidence (numbers)?
    import re
    numbers = re.findall(r'\d+\.?\d*', content)
    has_evidence = len(numbers) >= 3
    checks["has_evidence"] = has_evidence
    if has_evidence:
        score += 25

    # Has structured analysis (multiple sections)?
    has_structure = content.count("\n") >= 5
    checks["has_structure"] = has_structure
    if has_structure:
        score += 25

    # Reasonable length (not too short, not too verbose)
    length = len(content)
    good_length = 200 <= length <= 3000
    checks["good_length"] = good_length
    if good_length:
        score += 25

    return {"score": score, "checks": checks}


def run_single_benchmark(config: dict, round_num: int) -> dict:
    """Run one benchmark round with a specific config."""
    print(f"    Round {round_num}...", end=" ", flush=True)

    # Generate same mock data for all configs
    tech_analysis = run_analysis(SYMBOL)
    tech_report = format_analysis_report(tech_analysis)
    news_analysis = run_news_analysis()
    news_report = format_news_report(news_analysis)

    # Agent 1: AnalysisData
    analytics_cfg = config["analytics"]
    analytics_result = call_model(
        ANALYSIS_SYSTEM_PROMPT, tech_report,
        analytics_cfg["model"], analytics_cfg["temperature"], analytics_cfg["max_tokens"]
    )
    analytics_quality = evaluate_response_quality(analytics_result["content"])

    # Agent 2: ReadNews
    news_cfg = config["news"]
    news_result = call_model(
        NEWS_SYSTEM_PROMPT, news_report,
        news_cfg["model"], news_cfg["temperature"], news_cfg["max_tokens"]
    )
    news_quality = evaluate_response_quality(news_result["content"])

    # Agent 3: Orchestrator
    orch_cfg = config["orchestrator"]
    orch_input = (
        f"--- Phân tích kỹ thuật ---\n{analytics_result['content']}\n\n"
        f"--- Phân tích tâm lý ---\n{news_result['content']}\n\n"
        f"--- Signal ---\nOverall: {tech_analysis['overall_trend']}, "
        f"Sentiment: {news_analysis.get('signal', 'N/A')}"
    )
    orch_result = call_model(
        ORCHESTRATOR_SYSTEM_PROMPT, orch_input,
        orch_cfg["model"], orch_cfg["temperature"], orch_cfg["max_tokens"]
    )
    orch_quality = evaluate_response_quality(orch_result["content"])

    total_tokens = (analytics_result["total_tokens"] +
                    news_result["total_tokens"] +
                    orch_result["total_tokens"])
    total_cost = (analytics_result["cost_usd"] +
                  news_result["cost_usd"] +
                  orch_result["cost_usd"])
    total_latency = (analytics_result["latency_s"] +
                     news_result["latency_s"] +
                     orch_result["latency_s"])
    avg_quality = (analytics_quality["score"] +
                   news_quality["score"] +
                   orch_quality["score"]) / 3

    print(f"Tokens: {total_tokens} | Cost: ${total_cost:.6f} | "
          f"Latency: {total_latency:.1f}s | Quality: {avg_quality:.0f}/100")

    return {
        "round": round_num,
        "agents": {
            "analytics": {**analytics_result, "quality": analytics_quality},
            "news": {**news_result, "quality": news_quality},
            "orchestrator": {**orch_result, "quality": orch_quality},
        },
        "totals": {
            "tokens": total_tokens,
            "cost_usd": round(total_cost, 6),
            "latency_s": round(total_latency, 2),
            "avg_quality": round(avg_quality, 1),
        },
    }


def run_benchmark():
    """Run full benchmark across all configs."""
    print("=" * 60)
    print("  BENCHMARK: Model Configuration Comparison")
    print(f"  Symbol: {SYMBOL} | Rounds: {BENCHMARK_ROUNDS}")
    print("=" * 60)

    all_results = {}

    for config in BENCHMARK_CONFIGS:
        name = config["name"]
        print(f"\n--- {name} ---")
        print(f"  Orchestrator: {config['orchestrator']['model']} (temp={config['orchestrator']['temperature']})")
        print(f"  Analytics:    {config['analytics']['model']} (temp={config['analytics']['temperature']})")
        print(f"  News:         {config['news']['model']} (temp={config['news']['temperature']})")

        rounds = []
        for r in range(1, BENCHMARK_ROUNDS + 1):
            result = run_single_benchmark(config, r)
            rounds.append(result)

        # Aggregate
        avg_tokens = sum(r["totals"]["tokens"] for r in rounds) / len(rounds)
        avg_cost = sum(r["totals"]["cost_usd"] for r in rounds) / len(rounds)
        avg_latency = sum(r["totals"]["latency_s"] for r in rounds) / len(rounds)
        avg_quality = sum(r["totals"]["avg_quality"] for r in rounds) / len(rounds)
        total_cost = sum(r["totals"]["cost_usd"] for r in rounds)

        all_results[name] = {
            "config": config,
            "rounds": rounds,
            "summary": {
                "avg_tokens_per_round": round(avg_tokens),
                "avg_cost_per_round": round(avg_cost, 6),
                "total_cost": round(total_cost, 6),
                "avg_latency_s": round(avg_latency, 2),
                "avg_quality": round(avg_quality, 1),
            },
        }

    # Print comparison table
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)

    header = f"{'Config':<35} {'Tokens':>7} {'Cost/Round':>12} {'Latency':>8} {'Quality':>8}"
    print(header)
    print("-" * 70)

    for name, data in all_results.items():
        s = data["summary"]
        print(f"{name:<35} {s['avg_tokens_per_round']:>7} "
              f"${s['avg_cost_per_round']:<11.6f} "
              f"{s['avg_latency_s']:>6.1f}s "
              f"{s['avg_quality']:>6.1f}/100")

    # Recommendation
    print("\n" + "=" * 60)
    print("  RECOMMENDATION")
    print("=" * 60)

    best_quality = max(all_results.items(), key=lambda x: x[1]["summary"]["avg_quality"])
    best_cost = min(all_results.items(), key=lambda x: x[1]["summary"]["avg_cost_per_round"])
    best_balanced = max(all_results.items(),
                        key=lambda x: x[1]["summary"]["avg_quality"] / max(x[1]["summary"]["avg_cost_per_round"], 0.000001))

    print(f"  Best Quality:        {best_quality[0]} (Quality: {best_quality[1]['summary']['avg_quality']}/100)")
    print(f"  Lowest Cost:         {best_cost[0]} (${best_cost[1]['summary']['avg_cost_per_round']:.6f}/round)")
    print(f"  Best Value (Q/Cost): {best_balanced[0]}")

    # Estimate monthly cost
    print(f"\n  Estimated monthly cost (1h interval, 24/7):")
    rounds_per_month = 30 * 24  # 720 rounds
    for name, data in all_results.items():
        monthly = data["summary"]["avg_cost_per_round"] * rounds_per_month
        print(f"    {name:<35} ${monthly:.2f}/month")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    run_benchmark()
