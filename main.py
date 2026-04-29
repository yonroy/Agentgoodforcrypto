import os
import sys
import time
import random
import requests
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(__file__))

from src.Agent3.orchestrator import run_orchestrator, format_final_report
from src.Agent3.Len import run_len, format_len_report
from src.Agent3.Betting import process_signal, format_betting_report

# ============================================================
#  Configuration (loaded from .env)
# ============================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM_MESSAGES", "true").lower() == "true"
SYMBOL = os.getenv("CRYPTO_SYMBOL", "BTCUSDT")
LOOP_INTERVAL_MIN = int(os.getenv("LOOP_INTERVAL_MIN", "10"))
LOOP_INTERVAL_MAX = int(os.getenv("LOOP_INTERVAL_MAX", "30"))
LOOP_MAX_ITERATIONS = int(os.getenv("LOOP_MAX_ITERATIONS", "0"))  # 0 = unlimited
LEN_AGENT_ENABLED = os.getenv("LEN_AGENT_ENABLED", "true").lower() == "true"


def send_telegram_message(message: str):
    """Send a message via Telegram Bot API."""
    if not SEND_TELEGRAM:
        return
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


def run_full_analysis(symbol: str, round_number: int) -> str:
    """Run the 3-agent analysis once and return the report."""
    print(f"\n[{datetime.now()}] === Round #{round_number} === Starting 3-Agent analysis for {symbol}...")

    result = run_orchestrator(symbol, round_number)
    report = format_final_report(result)

    # Run Betting Agent based on orchestrator verdict
    synthesis = result["synthesis"]
    betting_result = process_signal(
        synthesis["final_trend"],
        result["current_price"],
        synthesis["confidence_score"],
    )
    betting_report = format_betting_report(betting_result)
    print(betting_report)

    print(report)
    message = (
        f"Orchestrator AI: {symbol} - {result['orchestrator_ai']}\n"
        f"Final Trend: {synthesis['final_trend']}\n"
        f"Cost: ${result['cost']['total_cost_usd']:.6f}\n"
        f"Accuracy: {result['accuracy']['accuracy_pct']}%\n\n"
        f"{betting_report}"
    )
    send_telegram_message(message)

    trend = synthesis["final_trend"]
    cost = result["cost"]["total_cost_usd"]
    acc = result["accuracy"]["accuracy_pct"]
    print(f"\n[{datetime.now()}] Round #{round_number} Done. "
          f"Trend: {trend} | Cost: ${cost:.6f} | Accuracy: {acc}%")
    return report


def main():
    """Main loop: Len Agent fast scan -> trigger 3-Agent on signal -> Send Telegram.
    If LEN_AGENT_ENABLED=false, skip Len and run full analysis every DIRECT_INTERVAL_SEC seconds.
    """
    use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
    print(f"=== AgentGoodForCrypto ===")
    print(f"Symbol: {SYMBOL}")
    if LEN_AGENT_ENABLED:
        print(f"Mode: Len gatekeeper | Scan interval: {LOOP_INTERVAL_MIN}-{LOOP_INTERVAL_MAX}s (random)")
    else:
        print(f"Mode: Direct prediction | Interval: {LOOP_INTERVAL_MIN}-{LOOP_INTERVAL_MAX}s (random) | Len: OFF")
    print(f"Telegram: {'ON' if SEND_TELEGRAM else 'OFF'}")
    print(f"OpenAI: {'OK' if os.getenv('OPENAI_API_KEY') else 'Not configured'}")
    print(f"Mock Data: {'ON' if use_mock else 'OFF'}")
    print(f"Max Iterations: {LOOP_MAX_ITERATIONS if LOOP_MAX_ITERATIONS > 0 else 'Unlimited'}")
    print()

    scan_number = 0
    round_number = 0

    while True:
        scan_number += 1
        try:
            if LEN_AGENT_ENABLED:
                print(f"[{datetime.now()}] Len scan #{scan_number}...")
                signal = run_len(SYMBOL)
                print(format_len_report(signal))

                if signal is not None:
                    # Len detected oversold/overbought — trigger full 3-Agent pipeline
                    round_number += 1
                    run_full_analysis(SYMBOL, round_number)
            else:
                # Len disabled — run full analysis directly every cycle
                round_number += 1
                run_full_analysis(SYMBOL, round_number)
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            error_msg = f"[ERROR] {'Scan' if LEN_AGENT_ENABLED else 'Round'} #{scan_number} failed: {e}"
            print(error_msg)
            send_telegram_message(error_msg)

        if LOOP_MAX_ITERATIONS > 0 and scan_number >= LOOP_MAX_ITERATIONS:
            print(f"\nReached max iterations ({LOOP_MAX_ITERATIONS}). Stopping.")
            break

        delay = random.randint(LOOP_INTERVAL_MIN, LOOP_INTERVAL_MAX)
        if LEN_AGENT_ENABLED:
            print(f"[{datetime.now()}] Next Len scan in {delay}s... "
                  f"(scan {scan_number + 1}/{LOOP_MAX_ITERATIONS if LOOP_MAX_ITERATIONS > 0 else '~'} | "
                  f"triggered {round_number} analyses)")
        else:
            print(f"[{datetime.now()}] Next prediction in {delay}s "
                  f"(round {round_number + 1}/{LOOP_MAX_ITERATIONS if LOOP_MAX_ITERATIONS > 0 else '~'})")
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            print("\nStopping...")
            break


if __name__ == "__main__":
    main()
