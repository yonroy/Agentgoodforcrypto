"""Betting / Position Manager — simulates futures trading based on orchestrator signals.

Uses MONEY (initial capital) and FUTURE_LEVERAGE from .env to track a virtual
futures portfolio.  Persists state to data/betting.json.

Position logic:
  - BULLISH / SLIGHTLY_BULLISH  → open LONG  (or hold if already LONG)
  - BEARISH / SLIGHTLY_BEARISH  → open SHORT (or hold if already SHORT)
  - NEUTRAL                     → close any open position
  - Signal flip (LONG→BEARISH)  → close current, open opposite
"""

import json
import os
from datetime import datetime

BETTING_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "betting.json")

INITIAL_MONEY = float(os.getenv("MONEY", "100"))
LEVERAGE = int(os.getenv("FUTURE_LEVERAGE", "5"))
NEUTRAL_CLOSE_MIN_HOLD_MINUTES = int(os.getenv("NEUTRAL_CLOSE_MIN_HOLD_MINUTES", "5"))


def _load_state() -> dict:
    if os.path.exists(BETTING_FILE) and os.path.getsize(BETTING_FILE) > 0:
        with open(BETTING_FILE, "r") as f:
            return json.load(f)
    return {
        "balance": INITIAL_MONEY,
        "position": None,        # "LONG" | "SHORT" | None
        "entry_price": None,
        "position_size": None,   # USD notional (balance * leverage)
        "entry_time": None,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "history": [],           # last N trades
    }


def _save_state(state: dict):
    os.makedirs(os.path.dirname(BETTING_FILE), exist_ok=True)
    # Keep only last 20 trades in history
    state["history"] = state["history"][-20:]
    with open(BETTING_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _close_position(state: dict, current_price: float, reason: str) -> dict:
    """Close the current position and record PnL."""
    if state["position"] is None:
        return {"action": "NONE", "reason": "No position to close"}

    entry_price = state["entry_price"]
    position_size = state["position_size"]
    direction = state["position"]

    # PnL calculation for futures
    if direction == "LONG":
        price_change_pct = (current_price - entry_price) / entry_price
    else:  # SHORT
        price_change_pct = (entry_price - current_price) / entry_price

    pnl = position_size * price_change_pct
    pnl_pct = price_change_pct * LEVERAGE * 100

    # Update balance
    state["balance"] += pnl
    state["total_pnl"] += pnl
    state["total_trades"] += 1
    if pnl >= 0:
        state["winning_trades"] += 1
    else:
        state["losing_trades"] += 1

    trade_record = {
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": current_price,
        "pnl": round(pnl, 4),
        "pnl_pct": round(pnl_pct, 2),
        "balance_after": round(state["balance"], 4),
        "reason": reason,
        "closed_at": datetime.now().isoformat(),
    }
    state["history"].append(trade_record)

    result = {
        "action": f"CLOSE_{direction}",
        "entry_price": entry_price,
        "exit_price": current_price,
        "pnl": round(pnl, 4),
        "pnl_pct": round(pnl_pct, 2),
        "reason": reason,
    }

    # Clear position
    state["position"] = None
    state["entry_price"] = None
    state["position_size"] = None
    state["entry_time"] = None

    return result


def _open_position(state: dict, direction: str, current_price: float) -> dict:
    """Open a new LONG or SHORT position."""
    if state["balance"] <= 0:
        return {"action": "NONE", "reason": "Balance depleted — cannot open position"}

    state["position"] = direction
    state["entry_price"] = current_price
    state["position_size"] = state["balance"] * LEVERAGE
    state["entry_time"] = datetime.now().isoformat()

    return {
        "action": f"OPEN_{direction}",
        "entry_price": current_price,
        "position_size": round(state["position_size"], 4),
        "leverage": LEVERAGE,
        "balance_used": round(state["balance"], 4),
    }


def _trend_to_direction(final_trend: str) -> str | None:
    """Map orchestrator trend to position direction."""
    trend_upper = final_trend.upper()
    if "BULLISH" in trend_upper:
        return "LONG"
    elif "BEARISH" in trend_upper:
        return "SHORT"
    return None  # NEUTRAL


def _minutes_since_entry(entry_time: str | None) -> float | None:
    if not entry_time:
        return None
    try:
        return (datetime.now() - datetime.fromisoformat(entry_time)).total_seconds() / 60
    except ValueError:
        return None


def process_signal(final_trend: str, current_price: float, confidence_score: int) -> dict:
    """Main entry point: decide what to do based on the orchestrator's verdict.

    Returns a dict describing actions taken and current portfolio state.
    """
    state = _load_state()
    desired_direction = _trend_to_direction(final_trend)
    current_position = state["position"]

    actions = []

    if desired_direction is None:
        # NEUTRAL → close any open position
        if current_position is not None:
            held_minutes = _minutes_since_entry(state.get("entry_time"))
            if held_minutes is None or held_minutes >= NEUTRAL_CLOSE_MIN_HOLD_MINUTES:
                close_result = _close_position(state, current_price, "Signal turned NEUTRAL")
                actions.append(close_result)
            else:
                entry_price = state["entry_price"]
                position_size = state["position_size"]
                if current_position == "LONG":
                    price_change_pct = (current_price - entry_price) / entry_price
                else:
                    price_change_pct = (entry_price - current_price) / entry_price
                unrealized_pnl = position_size * price_change_pct
                actions.append({
                    "action": f"HOLD_{current_position}",
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "unrealized_pnl": round(unrealized_pnl, 4),
                    "unrealized_pnl_pct": round(price_change_pct * LEVERAGE * 100, 2),
                    "reason": (
                        f"Signal NEUTRAL but minimum hold not reached "
                        f"({held_minutes:.1f}m < {NEUTRAL_CLOSE_MIN_HOLD_MINUTES}m)"
                    ),
                })
    elif current_position == desired_direction:
        # Already in the right direction → hold
        # Calculate unrealized PnL
        entry_price = state["entry_price"]
        position_size = state["position_size"]
        if desired_direction == "LONG":
            price_change_pct = (current_price - entry_price) / entry_price
        else:
            price_change_pct = (entry_price - current_price) / entry_price
        unrealized_pnl = position_size * price_change_pct

        actions.append({
            "action": f"HOLD_{desired_direction}",
            "entry_price": entry_price,
            "current_price": current_price,
            "unrealized_pnl": round(unrealized_pnl, 4),
            "unrealized_pnl_pct": round(price_change_pct * LEVERAGE * 100, 2),
        })
    else:
        # Different direction or no position
        if current_position is not None:
            # Close opposite position first
            close_result = _close_position(state, current_price, f"Signal flipped to {desired_direction}")
            actions.append(close_result)
        # Open new position
        open_result = _open_position(state, desired_direction, current_price)
        actions.append(open_result)

    _save_state(state)

    # Include unrealized PnL from current open position in total_pnl display
    unrealized = next(
        (a.get("unrealized_pnl", 0) for a in actions if a.get("action", "").startswith("HOLD_")),
        0,
    )
    return {
        "actions": actions,
        "portfolio": {
            "balance": round(state["balance"], 4),
            "position": state["position"],
            "entry_price": state["entry_price"],
            "entry_time": state["entry_time"],
            "held_minutes": round(_minutes_since_entry(state.get("entry_time")) or 0, 2) if state["position"] else None,
            "position_size": round(state["position_size"], 4) if state["position_size"] else None,
            "leverage": LEVERAGE,
            "total_trades": state["total_trades"],
            "winning_trades": state["winning_trades"],
            "losing_trades": state["losing_trades"],
            "win_rate": round(state["winning_trades"] / state["total_trades"] * 100, 1) if state["total_trades"] > 0 else 0,
            "realized_pnl": round(state["total_pnl"], 4),
            "unrealized_pnl": round(unrealized, 4),
            "total_pnl": round(state["total_pnl"] + unrealized, 4),
        },
    }


def mark_to_market(current_price: float) -> dict:
    """Update unrealized PnL view without changing position state."""
    state = _load_state()
    current_position = state.get("position")
    actions = []

    if current_position is not None and state.get("entry_price") and state.get("position_size"):
        entry_price = state["entry_price"]
        position_size = state["position_size"]
        if current_position == "LONG":
            price_change_pct = (current_price - entry_price) / entry_price
        else:
            price_change_pct = (entry_price - current_price) / entry_price
        unrealized_pnl = position_size * price_change_pct
        actions.append({
            "action": f"HOLD_{current_position}",
            "entry_price": entry_price,
            "current_price": current_price,
            "unrealized_pnl": round(unrealized_pnl, 4),
            "unrealized_pnl_pct": round(price_change_pct * LEVERAGE * 100, 2),
            "reason": "Mark-to-market only (agents disabled)",
        })
    else:
        actions.append({
            "action": "NONE",
            "reason": "No open position",
        })

    # Include unrealized PnL from current open position in total_pnl display
    unrealized = next(
        (a.get("unrealized_pnl", 0) for a in actions if a.get("action", "").startswith("HOLD_")),
        0,
    )
    return {
        "actions": actions,
        "portfolio": {
            "balance": round(state["balance"], 4),
            "position": state["position"],
            "entry_price": state["entry_price"],
            "entry_time": state["entry_time"],
            "held_minutes": round(_minutes_since_entry(state.get("entry_time")) or 0, 2) if state["position"] else None,
            "position_size": round(state["position_size"], 4) if state["position_size"] else None,
            "leverage": LEVERAGE,
            "total_trades": state["total_trades"],
            "winning_trades": state["winning_trades"],
            "losing_trades": state["losing_trades"],
            "win_rate": round(state["winning_trades"] / state["total_trades"] * 100, 1) if state["total_trades"] > 0 else 0,
            "realized_pnl": round(state["total_pnl"], 4),
            "unrealized_pnl": round(unrealized, 4),
            "total_pnl": round(state["total_pnl"] + unrealized, 4),
        },
    }


def close_position_and_reopen(final_trend: str, current_price: float, confidence_score: int) -> dict:
    """Force-close the current position (records PnL win/loss), then immediately
    run process_signal to decide whether to open a new one.

    Used at end of MARK_TO_MARKET_DURATION_MIN so every period generates a trade record.
    """
    state = _load_state()
    actions = []

    # Step 1: always close current position if any
    if state["position"] is not None:
        close_result = _close_position(state, current_price, "Periodic re-eval close")
        actions.append(close_result)

    # Step 2: decide whether to open a new position based on fresh agent signal
    desired_direction = _trend_to_direction(final_trend)
    if desired_direction is not None and state["balance"] > 0:
        open_result = _open_position(state, desired_direction, current_price)
        actions.append(open_result)
    else:
        actions.append({"action": "NONE", "reason": f"Signal NEUTRAL after re-eval close"})

    _save_state(state)

    unrealized = next(
        (a.get("unrealized_pnl", 0) for a in actions if a.get("action", "").startswith("HOLD_")),
        0,
    )
    return {
        "actions": actions,
        "portfolio": {
            "balance": round(state["balance"], 4),
            "position": state["position"],
            "entry_price": state["entry_price"],
            "entry_time": state["entry_time"],
            "held_minutes": round(_minutes_since_entry(state.get("entry_time")) or 0, 2) if state["position"] else None,
            "position_size": round(state["position_size"], 4) if state["position_size"] else None,
            "leverage": LEVERAGE,
            "total_trades": state["total_trades"],
            "winning_trades": state["winning_trades"],
            "losing_trades": state["losing_trades"],
            "win_rate": round(state["winning_trades"] / state["total_trades"] * 100, 1) if state["total_trades"] > 0 else 0,
            "realized_pnl": round(state["total_pnl"], 4),
            "unrealized_pnl": round(unrealized, 4),
            "total_pnl": round(state["total_pnl"] + unrealized, 4),
        },
    }


def format_betting_report(betting_result: dict) -> str:
    """Format betting result for display."""
    p = betting_result["portfolio"]
    actions = betting_result["actions"]

    lines = [
        f"{'='*35}",
        f"  BETTING (Futures x{p['leverage']})",
        f"{'='*35}",
    ]

    for a in actions:
        action = a["action"]
        if action.startswith("OPEN_"):
            lines.append(f"  >> {action} @ {a['entry_price']}")
            lines.append(f"     Size: ${a['position_size']} (Balance: ${a['balance_used']})")
        elif action.startswith("CLOSE_"):
            pnl_sign = "+" if a["pnl"] >= 0 else ""
            lines.append(f"  >> {action}: {a['entry_price']} -> {a['exit_price']}")
            lines.append(f"     PnL: {pnl_sign}${a['pnl']} ({pnl_sign}{a['pnl_pct']}%)")
            lines.append(f"     Reason: {a['reason']}")
        elif action.startswith("HOLD_"):
            pnl_sign = "+" if a["unrealized_pnl"] >= 0 else ""
            lines.append(f"  >> {action} (entry: {a['entry_price']})")
            lines.append(f"     Unrealized: {pnl_sign}${a['unrealized_pnl']} ({pnl_sign}{a['unrealized_pnl_pct']}%)")
        elif action == "NONE":
            lines.append(f"  >> No action: {a['reason']}")

    lines.append(f"")
    lines.append(f"  Balance: ${p['balance']} (Initial: ${INITIAL_MONEY})")
    total_return = ((p['balance'] - INITIAL_MONEY) / INITIAL_MONEY) * 100
    ret_sign = "+" if total_return >= 0 else ""
    lines.append(f"  Total Return: {ret_sign}{total_return:.2f}%")
    lines.append(f"  Position: {p['position'] or 'NONE'}")
    if p["total_trades"] > 0:
        lines.append(f"  Trades: {p['total_trades']} (W:{p['winning_trades']} L:{p['losing_trades']} WR:{p['win_rate']}%)")
    lines.append(f"  Total PnL: ${p['total_pnl']}")

    return "\n".join(lines)
