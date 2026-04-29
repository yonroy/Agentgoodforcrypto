import os
import sys
import time
import threading
from collections import deque
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
sys.path.insert(0, os.path.dirname(__file__))

LEN_AGENT_ENABLED = os.getenv("LEN_AGENT_ENABLED", "true").lower() == "true"
MARK_TO_MARKET_DURATION_MIN = float(os.getenv("MARK_TO_MARKET_DURATION_MIN", "5"))

from src.API.GetDataCrypto import get_current_price
from src.Agent3.Betting import _save_state, mark_to_market, process_signal, close_position_and_reopen
import src.Agent3.Betting as BettingModule
from src.Agent3.Len import run_len
from src.Agent3.orchestrator import run_orchestrator


class BotConfig(BaseModel):
    symbol: str = Field(default="BTCUSDT", min_length=3, max_length=20)
    capital: float = Field(default=1000, gt=0)
    leverage: int = Field(default=3, ge=1, le=50)
    interval_sec: int = Field(default=10, ge=2, le=300)


app = FastAPI(title="AgentGoodForCrypto API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime_lock = threading.Lock()
stop_event = threading.Event()
worker_thread: threading.Thread | None = None

runtime = {
    "running": False,
    "config": BotConfig().model_dump(),
    "scan_count": 0,
    "round_count": 0,
    "last_price": None,
    "last_trend": "NEUTRAL",
    "last_confidence": 0,
    "last_error": None,
    "portfolio": {
        "balance": 0,
        "position": None,
        "entry_price": None,
        "position_size": None,
        "leverage": 0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0,
        "realized_pnl": 0,
        "unrealized_pnl": 0,
        "total_pnl": 0,
    },
    "last_actions": [],
    "updated_at": None,
    "last_reeval_time": None,  # datetime of last agent re-evaluation while position is open
}
logs = deque(maxlen=200)


def add_log(message: str):
    logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def reset_betting_state(capital: float, leverage: int):
    BettingModule.INITIAL_MONEY = float(capital)
    BettingModule.LEVERAGE = int(leverage)
    _save_state(
        {
            "balance": float(capital),
            "position": None,
            "entry_price": None,
            "position_size": None,
            "entry_time": None,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "history": [],
        }
    )


def bot_worker():
    add_log("Backend worker started.")
    add_log("Phase 1: use agents to open initial position; Phase 2: mark-to-market only.")
    while not stop_event.is_set():
        with runtime_lock:
            cfg = dict(runtime["config"])
            runtime["scan_count"] += 1
            scan_id = runtime["scan_count"]
            current_position = runtime["portfolio"].get("position")

        try:
            symbol = cfg["symbol"].upper()
            price = float(get_current_price(symbol))

            # Phase 1: no position yet -> use agents to find trend and open first position
            if current_position is None:
                if not LEN_AGENT_ENABLED:
                    # Len disabled: run full orchestrator directly every cycle
                    with runtime_lock:
                        runtime["round_count"] += 1
                        round_id = runtime["round_count"]
                    add_log(f"Direct analysis (Len OFF) round #{round_id} for {symbol}")
                    orchestration = run_orchestrator(symbol, round_id)
                    trend = orchestration["synthesis"]["final_trend"]
                    confidence = int(orchestration["synthesis"]["confidence_score"])
                    betting_result = process_signal(trend, price, confidence)
                else:
                    add_log(f"Signal scan #{scan_id} for {symbol} (finding first entry)")
                    len_signal = run_len(symbol)
                    if len_signal is not None:
                        with runtime_lock:
                            runtime["round_count"] += 1
                            round_id = runtime["round_count"]
                        add_log(f"Len triggered full analysis (round #{round_id})")
                        orchestration = run_orchestrator(symbol, round_id)
                        trend = orchestration["synthesis"]["final_trend"]
                        confidence = int(orchestration["synthesis"]["confidence_score"])
                        betting_result = process_signal(trend, price, confidence)
                    else:
                        betting_result = {
                            "actions": [{"action": "NONE", "reason": "Waiting for Len trigger to open first position"}],
                            "portfolio": runtime["portfolio"],
                        }
                        trend = "WAITING_FIRST_ENTRY"
                        confidence = 0
            else:
                # Phase 2: position exists
                last_reeval = runtime.get("last_reeval_time")
                if last_reeval is not None:
                    minutes_since_reeval = (datetime.now() - last_reeval).total_seconds() / 60
                else:
                    minutes_since_reeval = MARK_TO_MARKET_DURATION_MIN  # first time: trigger immediately

                if minutes_since_reeval < MARK_TO_MARKET_DURATION_MIN:
                    # Still within cooldown window — mark-to-market only
                    remaining = MARK_TO_MARKET_DURATION_MIN - minutes_since_reeval
                    add_log(
                        f"PnL tick #{scan_id} for {symbol} | next re-eval in {remaining:.1f}m"
                    )
                    betting_result = mark_to_market(price)
                    trend = "MARK_TO_MARKET_ONLY"
                    confidence = 0
                else:
                    # Cooldown expired — close current position, re-run agents, reopen if signal
                    with runtime_lock:
                        runtime["round_count"] += 1
                        round_id = runtime["round_count"]
                        runtime["last_reeval_time"] = datetime.now()
                    add_log(
                        f"Re-eval round #{round_id} for {symbol} | "
                        f"{minutes_since_reeval:.1f}m since last re-eval, closing & re-evaluating"
                    )
                    orchestration = run_orchestrator(symbol, round_id)
                    trend = orchestration["synthesis"]["final_trend"]
                    confidence = int(orchestration["synthesis"]["confidence_score"])
                    betting_result = close_position_and_reopen(trend, price, confidence)

            new_portfolio = betting_result.get("portfolio", runtime["portfolio"])
            with runtime_lock:
                runtime["last_price"] = price
                runtime["last_trend"] = trend
                runtime["last_confidence"] = confidence
                runtime["last_error"] = None
                runtime["portfolio"] = new_portfolio
                runtime["last_actions"] = betting_result.get("actions", [])
                runtime["updated_at"] = datetime.now().isoformat()
                # Reset reeval timer when position is closed
                if new_portfolio.get("position") is None:
                    runtime["last_reeval_time"] = None

            actions = betting_result.get("actions", [])
            if actions:
                for a in actions:
                    add_log(f"{a.get('action', 'NONE')} | reason: {a.get('reason', '-')}")
            else:
                add_log("No betting action.")

        except Exception as exc:
            with runtime_lock:
                runtime["last_error"] = str(exc)
                runtime["updated_at"] = datetime.now().isoformat()
            add_log(f"ERROR: {exc}")

        interval = int(cfg["interval_sec"])
        for _ in range(interval):
            if stop_event.is_set():
                break
            time.sleep(1)

    add_log("Backend worker stopped.")


def ensure_stopped():
    global worker_thread
    stop_event.set()
    if worker_thread and worker_thread.is_alive():
        worker_thread.join(timeout=5)
    worker_thread = None
    with runtime_lock:
        runtime["running"] = False


@app.get("/api/health")
def health():
    return {"ok": True, "service": "AgentGoodForCrypto API"}


@app.get("/api/status")
def status():
    with runtime_lock:
        return {
            "running": runtime["running"],
            "config": runtime["config"],
            "scan_count": runtime["scan_count"],
            "round_count": runtime["round_count"],
            "last_price": runtime["last_price"],
            "last_trend": runtime["last_trend"],
            "last_confidence": runtime["last_confidence"],
            "last_error": runtime["last_error"],
            "portfolio": runtime["portfolio"],
            "last_actions": runtime["last_actions"],
            "updated_at": runtime["updated_at"],
            "logs": list(logs),
        }


@app.post("/api/start")
def start(config: BotConfig):
    global worker_thread
    ensure_stopped()

    reset_betting_state(config.capital, config.leverage)
    with runtime_lock:
        runtime["running"] = True
        runtime["config"] = config.model_dump()
        runtime["scan_count"] = 0
        runtime["round_count"] = 0
        runtime["last_price"] = None
        runtime["last_trend"] = "NEUTRAL"
        runtime["last_confidence"] = 0
        runtime["last_error"] = None
        runtime["last_actions"] = []
        runtime["updated_at"] = datetime.now().isoformat()
        runtime["last_reeval_time"] = None
        runtime["portfolio"] = {
            "balance": round(config.capital, 4),
            "position": None,
            "entry_price": None,
            "position_size": None,
            "leverage": config.leverage,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
        }

    logs.clear()
    add_log(
        f"Bot started | symbol={config.symbol.upper()} capital={config.capital} "
        f"leverage=x{config.leverage} interval={config.interval_sec}s"
    )

    stop_event.clear()
    worker_thread = threading.Thread(target=bot_worker, daemon=True)
    worker_thread.start()
    return {"ok": True}


@app.post("/api/stop")
def stop():
    ensure_stopped()
    add_log("Bot stopped by user.")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
