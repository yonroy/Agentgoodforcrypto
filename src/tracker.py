import json
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
COST_FILE = os.path.join(DATA_DIR, "cost_log.json")
ACCURACY_FILE = os.path.join(DATA_DIR, "accuracy_log.json")

# Model pricing (USD per 1K tokens)
MODEL_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4.0": {"input": 0.03, "output": 0.06},
    "gpt-5.4": {"input": 0.05, "output": 0.15},
}


# ============================================================
#  Cost Tracker
# ============================================================

class CostTracker:
    def __init__(self):
        self.rounds: list[dict] = self._load(COST_FILE)
        self.current_round_costs: list[dict] = []

    def _load(self, path: str) -> list:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(COST_FILE, "w") as f:
            json.dump(self.rounds, f, indent=2, default=str)

    def record_call(self, agent_name: str, model: str,
                    prompt_tokens: int, completion_tokens: int):
        """Record a single OpenAI API call cost."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-3.5-turbo"])
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        entry = {
            "agent": agent_name,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(total_cost, 6),
        }
        self.current_round_costs.append(entry)
        return entry

    def finish_round(self, round_number: int):
        """Finalize the current round and save."""
        total_tokens = sum(c["total_tokens"] for c in self.current_round_costs)
        total_cost = sum(c["cost_usd"] for c in self.current_round_costs)

        round_summary = {
            "round": round_number,
            "timestamp": datetime.now().isoformat(),
            "calls": self.current_round_costs,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
        }
        self.rounds.append(round_summary)
        self.current_round_costs = []
        self._save()
        return round_summary

    def get_cumulative_stats(self) -> dict:
        """Get cumulative cost statistics across all rounds."""
        if not self.rounds:
            return {"total_rounds": 0, "total_tokens": 0, "total_cost_usd": 0}

        total_tokens = sum(r["total_tokens"] for r in self.rounds)
        total_cost = sum(r["total_cost_usd"] for r in self.rounds)
        avg_cost = total_cost / len(self.rounds)

        return {
            "total_rounds": len(self.rounds),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_per_round": round(avg_cost, 6),
        }

    def format_cost_report(self, round_summary: dict) -> str:
        """Format cost report for a single round."""
        cumulative = self.get_cumulative_stats()
        lines = [
            "=== Cost Report ===",
            f"Round #{round_summary['round']}:",
        ]
        for call in round_summary["calls"]:
            lines.append(f"  {call['agent']}: {call['total_tokens']} tokens (${call['cost_usd']:.6f})")
        lines.append(f"  Round Total: {round_summary['total_tokens']} tokens (${round_summary['total_cost_usd']:.6f})")
        lines.append(f"\nCumulative ({cumulative['total_rounds']} rounds):")
        lines.append(f"  Total Tokens: {cumulative['total_tokens']}")
        lines.append(f"  Total Cost: ${cumulative['total_cost_usd']:.6f}")
        lines.append(f"  Avg Cost/Round: ${cumulative['avg_cost_per_round']:.6f}")
        return "\n".join(lines)


# ============================================================
#  Accuracy Tracker
# ============================================================

class AccuracyTracker:
    def __init__(self):
        self.predictions: list[dict] = self._load(ACCURACY_FILE)

    def _load(self, path: str) -> list:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(ACCURACY_FILE, "w") as f:
            json.dump(self.predictions, f, indent=2, default=str)

    def record_prediction(self, round_number: int, symbol: str,
                          predicted_trend: str, price_at_prediction: float):
        """Record a new prediction with the current price."""
        entry = {
            "round": round_number,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "predicted_trend": predicted_trend,
            "price_at_prediction": price_at_prediction,
            "actual_direction": None,
            "is_correct": None,
        }
        self.predictions.append(entry)
        self._save()

    def evaluate_previous(self, current_price: float):
        """Evaluate the previous prediction against current price.

        Compares predicted trend with actual price movement:
        - BULLISH/SLIGHTLY_BULLISH → price should go UP
        - BEARISH/SLIGHTLY_BEARISH → price should go DOWN
        - NEUTRAL → always counted as correct (no directional bet)
        """
        if len(self.predictions) < 2:
            return None

        prev = self.predictions[-2]
        if prev["is_correct"] is not None:
            return prev  # Already evaluated

        prev_price = prev["price_at_prediction"]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100

        if abs(price_change_pct) < 0.05:
            actual_direction = "FLAT"
        elif price_change > 0:
            actual_direction = "UP"
        else:
            actual_direction = "DOWN"

        predicted = prev["predicted_trend"]
        if predicted in ("BULLISH", "SLIGHTLY_BULLISH"):
            is_correct = actual_direction in ("UP", "FLAT")
        elif predicted in ("BEARISH", "SLIGHTLY_BEARISH"):
            is_correct = actual_direction in ("DOWN", "FLAT")
        else:  # NEUTRAL
            is_correct = True

        prev["actual_direction"] = actual_direction
        prev["actual_price"] = current_price
        prev["price_change_pct"] = round(price_change_pct, 4)
        prev["is_correct"] = is_correct
        self._save()
        return prev

    def get_accuracy_stats(self) -> dict:
        """Calculate accuracy statistics across all evaluated rounds."""
        evaluated = [p for p in self.predictions if p["is_correct"] is not None]
        if not evaluated:
            return {
                "total_predictions": len(self.predictions),
                "evaluated": 0,
                "correct": 0,
                "accuracy_pct": 0.0,
            }

        correct = sum(1 for p in evaluated if p["is_correct"])
        total = len(evaluated)

        return {
            "total_predictions": len(self.predictions),
            "evaluated": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy_pct": round((correct / total) * 100, 2),
        }

    def format_accuracy_report(self) -> str:
        """Format accuracy report."""
        stats = self.get_accuracy_stats()
        lines = [
            "=== Accuracy Report ===",
            f"Total Predictions: {stats['total_predictions']}",
            f"Evaluated: {stats['evaluated']}",
            f"Correct: {stats['correct']}",
            f"Incorrect: {stats.get('incorrect', 0)}",
            f"Accuracy: {stats['accuracy_pct']}%",
        ]

        # Show last 5 evaluated predictions
        evaluated = [p for p in self.predictions if p["is_correct"] is not None]
        if evaluated:
            lines.append("\nRecent Predictions:")
            for p in evaluated[-5:]:
                icon = "OK" if p["is_correct"] else "MISS"
                lines.append(
                    f"  R{p['round']}: {p['predicted_trend']} → {p['actual_direction']} "
                    f"({p.get('price_change_pct', 0):+.4f}%) [{icon}]"
                )

        return "\n".join(lines)
