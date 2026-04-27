import os
import sys
import time
import requests
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from src.Agent3.orchestrator import run_orchestrator, format_final_report

# ============================================================
#  Configuration (loaded from .env)
# ============================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SYMBOL = os.getenv("CRYPTO_SYMBOL", "BTCUSDT")
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL", "3600"))


def send_telegram_message(message: str):
    """Send a message via Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARNING] Telegram not configured.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"[ERROR] Telegram send failed: {e}")


def run_once(symbol: str, round_number: int) -> str:
    """Run the 3-agent analysis once and return the report."""
    print(f"\n[{datetime.now()}] === Round #{round_number} === Starting analysis for {symbol}...")

    result = run_orchestrator(symbol, round_number)
    report = format_final_report(result)

    print(report)
    send_telegram_message(report)

    trend = result["synthesis"]["final_trend"]
    cost = result["cost"]["total_cost_usd"]
    acc = result["accuracy"]["accuracy_pct"]
    print(f"\n[{datetime.now()}] Round #{round_number} Done. "
          f"Trend: {trend} | Cost: ${cost:.6f} | Accuracy: {acc}%")
    return report


def main():
    """Main loop: Start(symbol, interval) -> 3 Agent -> Send Telegram."""
    print(f"=== AgentGoodForCrypto ===")
    print(f"Symbol: {SYMBOL}")
    print(f"Loop Interval: {LOOP_INTERVAL}s")
    print(f"Telegram: {'OK' if TELEGRAM_BOT_TOKEN else 'Not configured'}")
    print(f"OpenAI: {'OK' if os.getenv('OPENAI_API_KEY') else 'Not configured'}")
    print()

    round_number = 1
    while True:
        try:
            run_once(SYMBOL, round_number)
            round_number += 1
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            error_msg = f"[ERROR] Round #{round_number} failed: {e}"
            print(error_msg)
            send_telegram_message(error_msg)

        print(f"\nNext run in {LOOP_INTERVAL}s...")
        try:
            time.sleep(LOOP_INTERVAL)
        except KeyboardInterrupt:
            print("\nStopping...")
            break


if __name__ == "__main__":
    main()
