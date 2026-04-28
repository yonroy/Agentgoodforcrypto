"""Len Agent - lightweight gatekeeper that screens for oversold/overbought
conditions on short timeframes before triggering the full 3-Agent pipeline.

Runs on a fast loop (default 3 min).  No LLM calls - pure indicator math.

Input:  Binance klines for 1m, 5m, 15m timeframes.
Output:
  - Case 1 (oversold/overbought detected): dict with signal details
  - Case 2 (nothing notable):              None
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.API.GetDataCrypto import get_klines, get_current_price
from src.Agent3.AnalysisData import (
    calculate_sma,
    calculate_rsi,
    calculate_macd,
)

# Len Agent rate-limiting settings
LEN_ACTIONS = int(os.getenv("LEN_ACTIONS", "3"))
COOLDOWN_PERIOD = int(os.getenv("COOLDOWN_PERIOD", "15"))  # minutes

# Timeframe configs from diagram (only short timeframes for Len)
# Khung   | SMA      | RSI | MACD
# 1m-5m   | 5,10,20  | 14  | 12,26,9
# 15m     | 7,25,50  | 14  | 12,26,9
TIMEFRAME_CONFIG = {
    "1m":  {"sma_periods": [5, 10, 20],  "rsi_period": 14, "macd": (12, 26, 9)},
    "5m":  {"sma_periods": [5, 10, 20],  "rsi_period": 14, "macd": (12, 26, 9)},
    "15m": {"sma_periods": [7, 25, 50],  "rsi_period": 14, "macd": (12, 26, 9)},
}

# Thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_EXTREME_OVERSOLD = 20
RSI_EXTREME_OVERBOUGHT = 80


def _scan_timeframe(symbol: str, interval: str) -> dict:
    """Calculate indicators for a single timeframe."""
    config = TIMEFRAME_CONFIG[interval]
    klines = get_klines(symbol, interval, limit=100)
    closes = [k["close"] for k in klines]

    current_price = closes[-1] if closes else 0

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

    # Trend from SMA
    sma_vals = [v for v in sma_values.values() if v is not None]
    if sma_vals:
        above_count = sum(1 for v in sma_vals if current_price > v)
        if above_count == len(sma_vals):
            sma_trend = "BULLISH"
        elif above_count == 0:
            sma_trend = "BEARISH"
        else:
            sma_trend = "NEUTRAL"
    else:
        sma_trend = "NEUTRAL"

    return {
        "timeframe": interval,
        "current_price": round(current_price, 4),
        "sma": sma_values,
        "sma_trend": sma_trend,
        "rsi": round(rsi, 2) if rsi else None,
        "macd": macd,
    }


def _detect_extreme(scans: dict[str, dict]) -> dict | None:
    """Check scans for oversold or overbought extremes.

    Returns a signal dict if any timeframe shows extreme RSI, or None.
    Priority: extreme (RSI < 20 / > 80) first, then normal (< 30 / > 70).
    At least 2 timeframes must agree on the direction to trigger.
    """
    oversold_tfs = []
    overbought_tfs = []
    extreme_oversold_tfs = []
    extreme_overbought_tfs = []

    for tf, data in scans.items():
        rsi = data.get("rsi")
        if rsi is None:
            continue

        if rsi <= RSI_EXTREME_OVERSOLD:
            extreme_oversold_tfs.append(tf)
            oversold_tfs.append(tf)
        elif rsi <= RSI_OVERSOLD:
            oversold_tfs.append(tf)

        if rsi >= RSI_EXTREME_OVERBOUGHT:
            extreme_overbought_tfs.append(tf)
            overbought_tfs.append(tf)
        elif rsi >= RSI_OVERBOUGHT:
            overbought_tfs.append(tf)

    # MACD confirmation: histogram sign matches the RSI signal
    def _macd_confirms(tfs: list[str], bullish: bool) -> int:
        count = 0
        for tf in tfs:
            macd = scans[tf].get("macd")
            if macd and macd.get("histogram") is not None:
                if bullish and macd["histogram"] > 0:
                    count += 1
                elif not bullish and macd["histogram"] < 0:
                    count += 1
        return count

    # Extreme cases: single timeframe is enough
    if extreme_oversold_tfs:
        return _build_signal(scans, extreme_oversold_tfs, oversold=True, extreme=True)
    if extreme_overbought_tfs:
        return _build_signal(scans, extreme_overbought_tfs, oversold=False, extreme=True)

    # Normal cases: need >= 2 timeframes agreeing
    if len(oversold_tfs) >= 2:
        return _build_signal(scans, oversold_tfs, oversold=True, extreme=False)
    if len(overbought_tfs) >= 2:
        return _build_signal(scans, overbought_tfs, oversold=False, extreme=False)

    return None


def _build_signal(scans: dict[str, dict], trigger_tfs: list[str],
                  *, oversold: bool, extreme: bool) -> dict:
    """Build the output signal dict."""
    # Aggregate technical trend across all scanned timeframes
    trends = [d.get("sma_trend", "NEUTRAL") for d in scans.values()]
    bullish_count = trends.count("BULLISH")
    bearish_count = trends.count("BEARISH")
    if bullish_count > bearish_count:
        technical_trend = "BULLISH"
    elif bearish_count > bullish_count:
        technical_trend = "BEARISH"
    else:
        technical_trend = "NEUTRAL"

    # RSI values for trigger timeframes
    rsi_details = {tf: scans[tf]["rsi"] for tf in trigger_tfs}

    if oversold:
        flag_type = "EXTREME_OVERSOLD" if extreme else "OVERSOLD"
        reversal_signal = "REVERSAL_UP"
    else:
        flag_type = "EXTREME_OVERBOUGHT" if extreme else "OVERBOUGHT"
        reversal_signal = "REVERSAL_DOWN"

    return {
        "oversold_flag": oversold,
        "overbought_flag": not oversold,
        "flag_type": flag_type,
        "trigger_timeframes": trigger_tfs,
        "rsi_details": rsi_details,
        "technical_trend": technical_trend,
        "reversal_signal": reversal_signal,
    }


class CooldownTracker:
    """Tracks how many times Len has triggered and enforces cooldown."""

    def __init__(self, max_actions: int = LEN_ACTIONS,
                 cooldown_minutes: int = COOLDOWN_PERIOD):
        self.max_actions = max_actions
        self.cooldown_minutes = cooldown_minutes
        self._actions_used = 0
        self._cooldown_until: datetime | None = None

    def is_on_cooldown(self) -> bool:
        if self._cooldown_until is None:
            return False
        if datetime.now() >= self._cooldown_until:
            # Cooldown expired - reset
            self._actions_used = 0
            self._cooldown_until = None
            return False
        return True

    def remaining_cooldown(self) -> timedelta:
        if self._cooldown_until is None:
            return timedelta(0)
        remaining = self._cooldown_until - datetime.now()
        return max(remaining, timedelta(0))

    def actions_remaining(self) -> int:
        if self.is_on_cooldown():
            return 0
        return self.max_actions - self._actions_used

    def record_action(self):
        self._actions_used += 1
        if self._actions_used >= self.max_actions:
            self._cooldown_until = datetime.now() + timedelta(minutes=self.cooldown_minutes)
            print(f"[Len] Action limit reached ({self.max_actions}). "
                  f"Cooldown until {self._cooldown_until.strftime('%H:%M:%S')}")


# Module-level tracker (persists across iterations within the same process)
_cooldown_tracker = CooldownTracker()


def run_len(symbol: str) -> dict | None:
    """Run the Len Agent scan for a symbol.

    Respects LEN_ACTIONS and COOLDOWN_PERIOD limits.

    Returns:
        dict with signal details if oversold/overbought detected, else None.
    """
    # Check cooldown first
    if _cooldown_tracker.is_on_cooldown():
        remaining = _cooldown_tracker.remaining_cooldown()
        mins = int(remaining.total_seconds() // 60)
        secs = int(remaining.total_seconds() % 60)
        print(f"[Len] On cooldown - {mins}m{secs}s remaining. Skipping scan.")
        return None

    scans = {}
    for interval in TIMEFRAME_CONFIG:
        try:
            scans[interval] = _scan_timeframe(symbol, interval)
        except Exception as e:
            print(f"[Len] Error scanning {interval}: {e}")

    if not scans:
        return None

    signal = _detect_extreme(scans)

    if signal:
        # Attach symbol and price info
        first_scan = next(iter(scans.values()))
        signal["symbol_name"] = symbol
        signal["current_price"] = first_scan["current_price"]
        signal["scans"] = scans
        signal["actions_remaining"] = _cooldown_tracker.actions_remaining() - 1

        # Record this trigger
        _cooldown_tracker.record_action()

    return signal


def format_len_report(signal: dict | None) -> str:
    """Format Len Agent result for logging."""
    if signal is None:
        if _cooldown_tracker.is_on_cooldown():
            remaining = _cooldown_tracker.remaining_cooldown()
            mins = int(remaining.total_seconds() // 60)
            return f"[Len] On cooldown - {mins}m remaining."
        return "[Len] No oversold/overbought detected - skipping 3-Agent."

    lines = [
        f"[Len] === {signal['flag_type']} detected for {signal['symbol_name']} ===",
        f"  Price: {signal['current_price']}",
        f"  Trigger timeframes: {', '.join(signal['trigger_timeframes'])}",
        f"  RSI: {signal['rsi_details']}",
        f"  Technical trend: {signal['technical_trend']}",
        f"  Reversal signal: {signal['reversal_signal']}",
        f"  Actions remaining: {signal.get('actions_remaining', '?')}",
        f"  >> Triggering full 3-Agent analysis...",
    ]
    return "\n".join(lines)
