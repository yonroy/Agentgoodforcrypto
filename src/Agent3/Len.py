"""Len Agent - lightweight gatekeeper that screens for oversold/overbought
conditions on short timeframes before triggering the full 3-Agent pipeline.

Runs on a fast loop (default 3 min).  No LLM calls - pure indicator math.

Input:  Binance klines for 1m, 5m, 15m timeframes (fetched in parallel).
Output:
  - Case 1 (signal detected): dict with signal details
  - Case 2 (nothing notable): None

Priority (highest → lowest):
  P1: RSI Divergence         — leading, fires 5-15 bars before reversal
  P2: MACD/Stoch cross + RSI near threshold — early entry confirmation
  P3: ≥2 TF agreement        — higher-confidence multi-timeframe signal
  P4: Extreme RSI            — lagging but urgent; 1m alone needs volume spike

DIVERGENT_TF (1m vs 15m conflict): logged as a warning, pipeline NOT triggered.
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.API.GetDataCrypto import get_klines
from src.Agent3.AnalysisData import calculate_sma, calculate_rsi, calculate_macd

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("Len")

# ============================================================
# Constants — tune here, not in the logic below
# ============================================================

# Minutes to suppress same-direction re-triggers after a signal fires
COOLDOWN_PERIOD = int(os.getenv("COOLDOWN_PERIOD", "15"))

# RSI thresholds
RSI_OVERSOLD          = 30
RSI_OVERBOUGHT        = 70
RSI_EXTREME_OVERSOLD  = 20
RSI_EXTREME_OVERBOUGHT = 80

# P2: RSI can be this many points above/below the threshold and still qualify
# when confirmed by a MACD or Stochastic crossover (e.g. RSI=34 with BULLISH_CROSS)
MACD_RSI_BUFFER = 5

# Fractal wings: bars each side required to confirm a swing high/low.
# 1 = original (noisy), 3 = Williams fractal (standard), 5 = very stable
PIVOT_WINGS = 3

# Bars to look back when searching for RSI divergence pivots
DIVERGENCE_LOOKBACK = 20  # 20 bars is enough to capture 5-15 bar reversal setups

# Bars used for RSI velocity (rate of change) computation
RSI_VELOCITY_BARS = 3  # 3 bars is fast enough without excessive lag

# RSI must deepen by this much (same direction) to override an active cooldown
RSI_DEEPENING_THRESHOLD = 5  # empirically: less noise than 3, not too slow as 10

# Volume spike threshold: current bar volume vs 20-bar average
VOLUME_SPIKE_RATIO = 2.0  # 2× average = notable absorption/distribution

# Bollinger Band bandwidth below this ratio = squeeze (breakout coming soon)
BB_SQUEEZE_THRESHOLD = 0.02  # 2% bandwidth is tight for crypto 1-15m frames

# Stochastic standard parameters
STOCH_PERIOD = 14
STOCH_SMOOTH = 3   # D-line smoothing periods

# Minimum klines to proceed (need at least slow MACD EMA + signal window)
MIN_KLINES = 60

# Klines to fetch per timeframe (generously above MIN_KLINES)
KLINES_LIMIT = 150

# Timeframe configs
TIMEFRAME_CONFIG: dict[str, dict] = {
    "1m":  {"sma_periods": [5, 10, 20],  "rsi_period": 14, "macd": (12, 26, 9)},
    "5m":  {"sma_periods": [5, 10, 20],  "rsi_period": 14, "macd": (12, 26, 9)},
    "15m": {"sma_periods": [7, 25, 50],  "rsi_period": 14, "macd": (12, 26, 9)},
}

# Persistence paths
_DATA_DIR          = os.path.join(os.path.dirname(__file__), "..", "..", "data")
COOLDOWN_STATE_FILE = os.path.join(_DATA_DIR, "len_cooldown_state.json")
SIGNAL_LOG_FILE     = os.path.join(_DATA_DIR, "len_signals.json")


# ============================================================
# Indicator helpers
# ============================================================

def calculate_rsi_series(closes: list[float], period: int) -> list[float | None]:
    """Full RSI series via Wilder's smoothing — single O(n) pass.

    Returns a list aligned with *closes* (first *period* entries are None).
    Used by divergence detection to replace the original 20× repeated
    calculate_rsi() calls (each O(n)), reducing work to a single O(n) pass.
    """
    result: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return result

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    if len(gains) < period:
        return result

    def _to_rsi(ag: float, al: float) -> float:
        return 100.0 if al == 0 else 100 - (100 / (1 + ag / al))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    result[period] = _to_rsi(avg_gain, avg_loss)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        result[i + 1] = _to_rsi(avg_gain, avg_loss)

    return result


def calculate_stochastic(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    k_period: int = STOCH_PERIOD,
    d_smooth: int = STOCH_SMOOTH,
) -> dict | None:
    """Stochastic %K / %D.

    More sensitive than RSI on 1m-5m — crossovers typically appear 2-5 bars
    earlier, making it a useful P2 complement to MACD crossovers.

    Returns {"k", "d", "crossover": "BULLISH_CROSS"|"BEARISH_CROSS"|None}.
    """
    if len(closes) < k_period + d_smooth:
        return None

    k_series: list[float] = []
    for i in range(k_period - 1, len(closes)):
        h_max = max(highs[i - k_period + 1: i + 1])
        l_min = min(lows[i - k_period + 1: i + 1])
        denom = h_max - l_min
        k_series.append(50.0 if denom == 0 else 100 * (closes[i] - l_min) / denom)

    if len(k_series) < d_smooth:
        return None

    k = k_series[-1]
    d = sum(k_series[-d_smooth:]) / d_smooth

    crossover = None
    if len(k_series) >= d_smooth + 1:
        prev_d = sum(k_series[-(d_smooth + 1): -1]) / d_smooth
        prev_k = k_series[-2]
        if prev_k <= prev_d and k > d:
            crossover = "BULLISH_CROSS"
        elif prev_k >= prev_d and k < d:
            crossover = "BEARISH_CROSS"

    return {"k": round(k, 2), "d": round(d, 2), "crossover": crossover}


def calculate_bollinger_bands(
    closes: list[float],
    period: int = 20,
    mult: float = 2.0,
) -> dict | None:
    """Bollinger Bands + bandwidth ratio.

    bandwidth = (upper - lower) / middle.
    squeeze=True when bandwidth < BB_SQUEEZE_THRESHOLD, signalling that a
    breakout (in either direction) is imminent.
    """
    if len(closes) < period:
        return None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((p - mid) ** 2 for p in window) / period
    std = variance ** 0.5
    upper = mid + mult * std
    lower = mid - mult * std
    bandwidth = (upper - lower) / mid if mid != 0 else 0.0
    return {
        "upper":     round(upper, 4),
        "middle":    round(mid, 4),
        "lower":     round(lower, 4),
        "bandwidth": round(bandwidth, 6),
        "squeeze":   bandwidth < BB_SQUEEZE_THRESHOLD,
    }


# ============================================================
# Fractal pivot detection (replaces 1-bar comparison)
# ============================================================

def _is_pivot_low(series: list[float], i: int, wings: int = PIVOT_WINGS) -> bool:
    """True if series[i] is the lowest point within *wings* bars each side.

    Williams Fractal pattern — requires 2*wings+1 bars total.
    Significantly reduces false pivots vs the original single-neighbour check.
    """
    if i < wings or i >= len(series) - wings:
        return False
    return (all(series[i] <= series[i - k] for k in range(1, wings + 1)) and
            all(series[i] <= series[i + k] for k in range(1, wings + 1)))


def _is_pivot_high(series: list[float], i: int, wings: int = PIVOT_WINGS) -> bool:
    if i < wings or i >= len(series) - wings:
        return False
    return (all(series[i] >= series[i - k] for k in range(1, wings + 1)) and
            all(series[i] >= series[i + k] for k in range(1, wings + 1)))


# ============================================================
# Divergence detection
# ============================================================

def _detect_rsi_divergence(
    closes: list[float],
    rsi_period: int,
    lookback: int = DIVERGENCE_LOOKBACK,
) -> str | None:
    """Detect RSI divergence over the last *lookback* bars.

    Improvements over original:
    - Single-pass RSI series (calculate_rsi_series) instead of 20 repeated calls
    - Fractal pivot detection (PIVOT_WINGS=3) reduces false swing points

    Bullish divergence:  price makes lower low, RSI makes higher low
    Bearish divergence:  price makes higher high, RSI makes lower high

    Returns "BULLISH_DIVERGENCE", "BEARISH_DIVERGENCE", or None.
    """
    need = rsi_period + lookback + PIVOT_WINGS + 1
    if len(closes) < need:
        return None

    rsi_series  = calculate_rsi_series(closes, rsi_period)
    price_window = closes[-lookback:]
    rsi_window   = rsi_series[-lookback:]

    if any(v is None for v in rsi_window):
        return None

    rsi_vals = [float(v) for v in rsi_window]  # type: ignore[arg-type]

    price_lows  = [(i, price_window[i]) for i in range(len(price_window))
                   if _is_pivot_low(price_window, i)]
    price_highs = [(i, price_window[i]) for i in range(len(price_window))
                   if _is_pivot_high(price_window, i)]

    # Bullish: price lower low, RSI higher low
    if len(price_lows) >= 2:
        pi, pp = price_lows[-2]
        ci, cp = price_lows[-1]
        if cp < pp and rsi_vals[ci] > rsi_vals[pi]:
            return "BULLISH_DIVERGENCE"

    # Bearish: price higher high, RSI lower high
    if len(price_highs) >= 2:
        pi, pp = price_highs[-2]
        ci, cp = price_highs[-1]
        if cp > pp and rsi_vals[ci] < rsi_vals[pi]:
            return "BEARISH_DIVERGENCE"

    return None


# ============================================================
# Short indicator helpers
# ============================================================

def _detect_macd_crossover(
    closes: list[float], fast: int, slow: int, signal: int,
) -> str | None:
    """MACD histogram sign flip in the last 2 bars.

    NOTE: prefer _detect_macd_crossover_fast() when the current MACD is already
    computed — it avoids the extra calculate_macd() call on closes[:-1].
    """
    if len(closes) < slow + signal + 1:
        return None
    prev = calculate_macd(closes[:-1], fast, slow, signal)
    curr = calculate_macd(closes,      fast, slow, signal)
    return _macd_cross_from_dicts(prev, curr)


def _macd_cross_from_dicts(prev: dict | None, curr: dict | None) -> str | None:
    """Determine MACD crossover direction from two pre-computed MACD dicts.

    Separates the sign-flip logic from the data-fetching so _scan_timeframe
    can reuse the already-computed current MACD and only fetch one extra call
    (closes[:-1]) instead of two.
    """
    if not prev or not curr:
        return None
    h_p, h_c = prev.get("histogram"), curr.get("histogram")
    if h_p is None or h_c is None:
        return None
    if h_p <= 0 < h_c:
        return "BULLISH_CROSS"
    if h_p >= 0 > h_c:
        return "BEARISH_CROSS"
    return None


def _rsi_velocity(
    closes: list[float],
    rsi_period: int,
    rsi_series: list[float | None] | None = None,
) -> float | None:
    """RSI change over the last RSI_VELOCITY_BARS bars.

    Negative = RSI dropping (approaching oversold faster).
    Positive = RSI rising  (approaching overbought faster).

    Pass *rsi_series* (from calculate_rsi_series) to avoid a redundant O(n)
    compute — _scan_timeframe already builds the series for divergence detection.
    """
    bars = RSI_VELOCITY_BARS
    if len(closes) < rsi_period + bars + 1:
        return None

    if rsi_series is not None and len(rsi_series) >= bars + 1:
        now  = rsi_series[-1]
        prev = rsi_series[-(bars + 1)]
    else:
        now  = calculate_rsi(closes,         rsi_period)
        prev = calculate_rsi(closes[:-bars], rsi_period)

    if now is None or prev is None:
        return None
    return round(now - prev, 2)


def _detect_volume_absorption(klines: list[dict]) -> dict | None:
    """Detect volume spike paired with narrow price range = absorption.

    High volume + tight range means smart money is absorbing retail flow,
    which typically precedes a reversal.  Condition:
      vol_ratio ≥ VOLUME_SPIKE_RATIO  AND  normalized_range < 0.5 ATR

    Returns {"absorption": bool, "vol_ratio": float, "normalized_range": float}.
    """
    if len(klines) < 21:
        return None

    volumes = [k["volume"] for k in klines]
    highs   = [k["high"]   for k in klines]
    lows    = [k["low"]    for k in klines]
    closes  = [k["close"]  for k in klines]

    avg_vol = sum(volumes[-21:-1]) / 20   # 20-bar avg, exclude current bar
    if avg_vol == 0:
        return None

    # ATR-14 for normalising the price range
    true_ranges = [
        max(highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]))
        for i in range(1, len(klines))
    ]
    if len(true_ranges) < 14:
        return None
    atr = sum(true_ranges[-14:]) / 14
    if atr == 0:
        return None

    vol_ratio        = volumes[-1] / avg_vol
    normalized_range = (highs[-1] - lows[-1]) / atr
    absorption       = (vol_ratio >= VOLUME_SPIKE_RATIO and normalized_range < 0.5)

    return {
        "absorption":      absorption,
        "vol_ratio":       round(vol_ratio, 2),
        "normalized_range": round(normalized_range, 2),
    }


# ============================================================
# Timeframe scan
# ============================================================

def _scan_timeframe(symbol: str, interval: str) -> dict:
    """Fetch klines and compute all indicators for one timeframe.

    Raises ValueError if the API returns fewer klines than MIN_KLINES — a
    silent return of None indicators would produce false or missed signals.
    """
    config = TIMEFRAME_CONFIG[interval]
    klines = get_klines(symbol, interval, limit=KLINES_LIMIT)

    if len(klines) < MIN_KLINES:
        raise ValueError(
            f"Insufficient klines for {symbol}/{interval}: "
            f"got {len(klines)}, need {MIN_KLINES}"
        )

    closes = [k["close"]  for k in klines]
    highs  = [k["high"]   for k in klines]
    lows   = [k["low"]    for k in klines]
    current_price = closes[-1]

    # SMA
    sma_values: dict[str, float | None] = {}
    for period in config["sma_periods"]:
        val = calculate_sma(closes, period)
        sma_values[f"SMA_{period}"] = round(val, 4) if val is not None else None

    # RSI series — single O(n) pass; reused by divergence detection and velocity
    rsi_series = calculate_rsi_series(closes, config["rsi_period"])
    rsi = rsi_series[-1]  # current RSI scalar

    # MACD — compute current bar and previous bar together to detect crossover
    # without an extra O(n²) calculate_macd() call on closes[:-1]
    fast, slow, sig = config["macd"]
    macd      = calculate_macd(closes,      fast, slow, sig)
    macd_prev = calculate_macd(closes[:-1], fast, slow, sig)

    # Divergence — reuses rsi_series; no extra RSI computation
    divergence = _detect_rsi_divergence(closes, config["rsi_period"])

    # MACD crossover — reuses the two already-computed MACD dicts
    macd_cross = _macd_cross_from_dicts(macd_prev, macd)

    # RSI velocity — reuses rsi_series (no extra O(n) RSI calls)
    rsi_vel = _rsi_velocity(closes, config["rsi_period"], rsi_series=rsi_series)

    # Stochastic (more sensitive than RSI on short TFs)
    stochastic = calculate_stochastic(closes, highs, lows)

    # Bollinger Bands (squeeze = imminent breakout)
    bb = calculate_bollinger_bands(closes)

    # Volume absorption (absorption = reversal precursor)
    vol_absorption = _detect_volume_absorption(klines)

    # SMA trend
    sma_vals = [v for v in sma_values.values() if v is not None]
    if sma_vals:
        above = sum(1 for v in sma_vals if current_price > v)
        sma_trend = "BULLISH" if above == len(sma_vals) else (
            "BEARISH" if above == 0 else "NEUTRAL")
    else:
        sma_trend = "NEUTRAL"

    return {
        "timeframe":          interval,
        "current_price":      round(current_price, 4),
        "sma":                sma_values,
        "sma_trend":          sma_trend,
        "rsi":                round(rsi, 2) if rsi is not None else None,
        "rsi_velocity":       rsi_vel,
        "divergence":         divergence,
        "macd":               macd,
        "macd_crossover":     macd_cross,
        "stochastic":         stochastic,
        "bollinger_bands":    bb,
        "volume_absorption":  vol_absorption,
    }


_TF_FETCH_TIMEOUT = 15   # seconds; generous for slow Binance responses


def _scan_all_timeframes(symbol: str) -> dict[str, dict]:
    """Fetch all timeframes in parallel — saves ~2/3 of sequential API wait."""
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(TIMEFRAME_CONFIG)) as pool:
        futures = {pool.submit(_scan_timeframe, symbol, tf): tf
                   for tf in TIMEFRAME_CONFIG}
        for future in as_completed(futures):
            tf = futures[future]
            try:
                results[tf] = future.result(timeout=_TF_FETCH_TIMEOUT)
            except TimeoutError:
                _log.error("Timeout fetching %s/%s after %ds", symbol, tf, _TF_FETCH_TIMEOUT)
            except requests.RequestException as exc:
                _log.error("API error %s/%s: %s", symbol, tf, exc)
            except (ValueError, KeyError) as exc:
                _log.warning("Data error %s/%s: %s", symbol, tf, exc)
            except Exception as exc:
                _log.error("Unexpected error %s/%s: %s", symbol, tf, exc, exc_info=True)
    return results


# ============================================================
# Signal detection — priority-ordered
# ============================================================

def _detect_extreme(scans: dict[str, dict]) -> dict | None:  # noqa: C901
    """Classify market conditions into a signal (or None).

    Priority order (critical design decision — leading before lagging):
      P1: RSI Divergence             — leading, precedes reversal 5-15 bars early
      P2: MACD/Stoch cross + near-RSI — early entry with momentum confirmation
      P3: ≥2 TF agreement            — adds confidence via multi-timeframe consensus
      P4: Extreme RSI                — lagging but important; 1m-only needs vol spike

    Special: DIVERGENT_TF (1m oversold but 15m overbought or vice-versa) is logged
    as a module-level warning and returns None — pipeline is NOT triggered.
    """
    global _last_divergent_tf_warning

    oversold_tfs:           list[str] = []
    overbought_tfs:         list[str] = []
    extreme_oversold_tfs:   list[str] = []
    extreme_overbought_tfs: list[str] = []
    divergence_bull_tfs:    list[str] = []
    divergence_bear_tfs:    list[str] = []
    early_oversold_tfs:     list[str] = []
    early_overbought_tfs:   list[str] = []
    volume_spike_tfs:       list[str] = []

    for tf, data in scans.items():
        rsi = data.get("rsi")
        if rsi is None:
            continue

        div   = data.get("divergence")
        mx    = data.get("macd_crossover")
        vol   = data.get("volume_absorption") or {}
        stoch = data.get("stochastic") or {}

        # RSI classification
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

        # Divergence (P1) — only meaningful when RSI is actually near the threshold.
        # A bullish divergence at RSI=52 is a chart artefact, not a signal.
        # Buffer of 15 pts: RSI ≤ 45 for bullish, RSI ≥ 55 for bearish.
        if div == "BULLISH_DIVERGENCE" and rsi <= RSI_OVERSOLD + 15:
            divergence_bull_tfs.append(tf)
        elif div == "BEARISH_DIVERGENCE" and rsi >= RSI_OVERBOUGHT - 15:
            divergence_bear_tfs.append(tf)

        # MACD or Stochastic cross + RSI within buffer of threshold (P2)
        has_bull_momentum = (mx == "BULLISH_CROSS" or stoch.get("crossover") == "BULLISH_CROSS")
        has_bear_momentum = (mx == "BEARISH_CROSS" or stoch.get("crossover") == "BEARISH_CROSS")
        if rsi <= RSI_OVERSOLD + MACD_RSI_BUFFER and has_bull_momentum:
            early_oversold_tfs.append(tf)
        if rsi >= RSI_OVERBOUGHT - MACD_RSI_BUFFER and has_bear_momentum:
            early_overbought_tfs.append(tf)

        # Volume spike (boosts severity in P3/P4)
        if vol.get("absorption"):
            volume_spike_tfs.append(tf)

    # ── DIVERGENT_TF guard ─────────────────────────────────────────────────
    # 1m and 15m pointing opposite directions signals an unreliable market.
    # Log as a warning but do NOT trigger the pipeline — wait for resolution.
    conflict = (
        ("1m" in oversold_tfs   and "15m" in overbought_tfs) or
        ("1m" in overbought_tfs and "15m" in oversold_tfs)
    )
    if conflict:
        _last_divergent_tf_warning = {
            "timestamp": datetime.now().isoformat(),
            "1m_rsi":    scans.get("1m",  {}).get("rsi"),
            "15m_rsi":   scans.get("15m", {}).get("rsi"),
            "message":   "1m and 15m RSI conflict — ambiguous, not triggering pipeline",
        }
        _log.warning("DIVERGENT_TF: 1m RSI=%s vs 15m RSI=%s — skipping",
                     _last_divergent_tf_warning["1m_rsi"],
                     _last_divergent_tf_warning["15m_rsi"])
        return None

    _last_divergent_tf_warning = None

    # ── P1: RSI Divergence ─────────────────────────────────────────────────
    if divergence_bull_tfs:
        sev = "HIGH" if len(divergence_bull_tfs) >= 2 else "MEDIUM"
        return _build_signal(scans, divergence_bull_tfs, oversold=True, extreme=False,
                             trigger_reason="RSI_DIVERGENCE", severity=sev)
    if divergence_bear_tfs:
        sev = "HIGH" if len(divergence_bear_tfs) >= 2 else "MEDIUM"
        return _build_signal(scans, divergence_bear_tfs, oversold=False, extreme=False,
                             trigger_reason="RSI_DIVERGENCE", severity=sev)

    # ── P2: MACD/Stochastic cross + RSI near threshold ────────────────────
    if early_oversold_tfs:
        return _build_signal(scans, early_oversold_tfs, oversold=True, extreme=False,
                             trigger_reason="MACD_CROSSOVER_CONFIRM", severity="MEDIUM")
    if early_overbought_tfs:
        return _build_signal(scans, early_overbought_tfs, oversold=False, extreme=False,
                             trigger_reason="MACD_CROSSOVER_CONFIRM", severity="MEDIUM")

    # ── P3: ≥2 TF agreement ────────────────────────────────────────────────
    if len(oversold_tfs) >= 2:
        has_vol = bool(set(oversold_tfs) & set(volume_spike_tfs))
        sev = "HIGH" if has_vol else "MEDIUM"
        return _build_signal(scans, oversold_tfs, oversold=True, extreme=False,
                             trigger_reason="MULTI_TF_AGREE", severity=sev)
    if len(overbought_tfs) >= 2:
        has_vol = bool(set(overbought_tfs) & set(volume_spike_tfs))
        sev = "HIGH" if has_vol else "MEDIUM"
        return _build_signal(scans, overbought_tfs, oversold=False, extreme=False,
                             trigger_reason="MULTI_TF_AGREE", severity=sev)

    # ── P4: Extreme RSI ────────────────────────────────────────────────────
    # 1m RSI alone is very noisy (hits 20/80 frequently with no reversal).
    # Require a volume spike on 1m-only triggers to reduce false positives.
    if extreme_oversold_tfs:
        # Use set equality — list order is non-deterministic (ThreadPoolExecutor)
        only_1m = set(extreme_oversold_tfs) == {"1m"}
        if only_1m and "1m" not in volume_spike_tfs:
            _log.debug("Extreme oversold on 1m only, no volume spike — skipped (noise filter)")
        else:
            return _build_signal(scans, extreme_oversold_tfs, oversold=True, extreme=True,
                                 trigger_reason="EXTREME_RSI", severity="CRITICAL")
    if extreme_overbought_tfs:
        only_1m = set(extreme_overbought_tfs) == {"1m"}
        if only_1m and "1m" not in volume_spike_tfs:
            _log.debug("Extreme overbought on 1m only, no volume spike — skipped (noise filter)")
        else:
            return _build_signal(scans, extreme_overbought_tfs, oversold=False, extreme=True,
                                 trigger_reason="EXTREME_RSI", severity="CRITICAL")

    return None


def _build_signal(
    scans: dict[str, dict],
    trigger_tfs: list[str],
    *,
    oversold: bool,
    extreme: bool,
    trigger_reason: str = "RSI_THRESHOLD",
    severity: str = "MEDIUM",
) -> dict:
    """Construct the output signal dict."""
    trends    = [d.get("sma_trend", "NEUTRAL") for d in scans.values()]
    bull_cnt  = trends.count("BULLISH")
    bear_cnt  = trends.count("BEARISH")
    technical_trend = ("BULLISH" if bull_cnt > bear_cnt else
                       "BEARISH" if bear_cnt > bull_cnt else "NEUTRAL")

    if oversold:
        flag_type       = "EXTREME_OVERSOLD" if extreme else "OVERSOLD"
        reversal_signal = "REVERSAL_UP"
    else:
        flag_type       = "EXTREME_OVERBOUGHT" if extreme else "OVERBOUGHT"
        reversal_signal = "REVERSAL_DOWN"

    rsi_details = {tf: scans[tf]["rsi"]
                   for tf in trigger_tfs if scans[tf].get("rsi") is not None}

    # Collect early-detection context per trigger timeframe
    early_details: dict[str, dict] = {}
    for tf in trigger_tfs:
        d    = scans[tf]
        info: dict = {}
        if d.get("divergence"):
            info["divergence"] = d["divergence"]
        if d.get("macd_crossover"):
            info["macd_crossover"] = d["macd_crossover"]
        if d.get("rsi_velocity") is not None:
            info["rsi_velocity"] = d["rsi_velocity"]
        stoch = d.get("stochastic")
        if stoch:
            info["stochastic"] = (
                f"K={stoch['k']} D={stoch['d']}"
                + (f" [{stoch['crossover']}]" if stoch.get("crossover") else "")
            )
        vol = d.get("volume_absorption")
        if vol and vol.get("absorption"):
            info["absorption"] = f"vol_ratio={vol['vol_ratio']} range={vol['normalized_range']}"
        if info:
            early_details[tf] = info

    signal: dict = {
        "flag_type":          flag_type,
        "trigger_reason":     trigger_reason,
        "severity":           severity,
        "trigger_timeframes": trigger_tfs,
        "rsi_details":        rsi_details,
        "technical_trend":    technical_trend,
        "reversal_signal":    reversal_signal,
    }
    if early_details:
        signal["early_details"] = early_details
    return signal


# ============================================================
# Cooldown tracker — direction-aware + file persistence
# ============================================================

class CooldownTracker:
    """Independent oversold / overbought cooldowns, persisted to disk.

    Two separate cooldowns prevent a common miss: if oversold fires at 14:00
    and price reverses to overbought at 14:12, the original single cooldown
    would have blocked the overbought signal.  Now each direction is tracked
    independently.

    Persistence to COOLDOWN_STATE_FILE means cooldowns survive process restarts,
    avoiding duplicate triggers on the same signal after a crash or redeploy.
    """

    def __init__(self, cooldown_minutes: int = COOLDOWN_PERIOD):
        self.cooldown_minutes = cooldown_minutes
        self._state = self._load()

    def _load(self) -> dict:
        if os.path.exists(COOLDOWN_STATE_FILE):
            try:
                with open(COOLDOWN_STATE_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        return {
            "oversold":   {"until": None, "last_rsi": None},
            "overbought": {"until": None, "last_rsi": None},
        }

    def _save(self) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        try:
            with open(COOLDOWN_STATE_FILE, "w") as f:
                json.dump(self._state, f)
        except OSError as exc:
            _log.warning("Could not persist cooldown state: %s", exc)

    def _slot(self, oversold: bool) -> dict:
        return self._state["oversold" if oversold else "overbought"]

    def is_on_cooldown(self, current_rsi: float | None = None,
                       oversold: bool | None = None) -> bool:
        if oversold is None:
            return False
        slot = self._slot(oversold)
        if slot["until"] is None:
            return False
        until = datetime.fromisoformat(slot["until"])
        if datetime.now() >= until:
            slot["until"] = None
            slot["last_rsi"] = None
            self._save()
            return False

        # Allow re-trigger if RSI deepened significantly in the same direction
        last = slot["last_rsi"]
        if current_rsi is not None and last is not None:
            if oversold and current_rsi <= last - RSI_DEEPENING_THRESHOLD:
                _log.info("Cooldown override: oversold deepened %.1f → %.1f", last, current_rsi)
                return False
            if not oversold and current_rsi >= last + RSI_DEEPENING_THRESHOLD:
                _log.info("Cooldown override: overbought deepened %.1f → %.1f", last, current_rsi)
                return False

        return True

    def remaining_cooldown(self, oversold: bool) -> timedelta:
        slot = self._slot(oversold)
        if slot["until"] is None:
            return timedelta(0)
        return max(datetime.fromisoformat(slot["until"]) - datetime.now(), timedelta(0))

    def record_action(self, rsi: float | None = None, oversold: bool | None = None) -> None:
        if oversold is None:
            return
        direction = "oversold" if oversold else "overbought"
        until = datetime.now() + timedelta(minutes=self.cooldown_minutes)
        self._state[direction] = {"until": until.isoformat(), "last_rsi": rsi}
        self._save()
        _log.info("Cooldown[%s] %dm until %s (RSI=%.1f)",
                  direction, self.cooldown_minutes,
                  until.strftime("%H:%M:%S"), rsi or 0)


# ============================================================
# Signal logger — observability
# ============================================================

class SignalLogger:
    """Persist every fired signal to SIGNAL_LOG_FILE for offline analysis.

    Schema per entry:
      timestamp, symbol, flag_type, severity, trigger_reason,
      trigger_timeframes, rsi_details, outcome (None = unevaluated)

    After 1-2 weeks of data you can compute:
      - signals/hour distribution
      - false-positive rate (outcome == "WRONG" / total evaluated)
      - RSI distribution at trigger (tune thresholds per symbol)
    """

    MAX_RECORDS = 500   # cap file size; 500 entries ≈ 1-2 weeks at 3-scan/hour

    def log_signal(self, signal: dict, symbol: str) -> None:
        entry = {
            "timestamp":          datetime.now().isoformat(),
            "symbol":             symbol,
            "flag_type":          signal.get("flag_type"),
            "severity":           signal.get("severity"),
            "trigger_reason":     signal.get("trigger_reason"),
            "trigger_timeframes": signal.get("trigger_timeframes"),
            "rsi_details":        signal.get("rsi_details"),
            "outcome":            None,  # set externally: "CORRECT" | "WRONG"
        }
        records = self._load()
        records.append(entry)
        self._save(records[-self.MAX_RECORDS:])

    def get_stats(self) -> dict:
        records = self._load()
        if not records:
            return {"total_signals": 0, "signals_last_hour": 0,
                    "evaluated": 0, "false_positives": 0}
        one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        recent    = [r for r in records if r["timestamp"] >= one_hour_ago]
        evaluated = [r for r in records if r.get("outcome") is not None]
        fp        = sum(1 for r in evaluated if r.get("outcome") == "WRONG")
        return {
            "total_signals":     len(records),
            "signals_last_hour": len(recent),
            "evaluated":         len(evaluated),
            "false_positives":   fp,
        }

    def _load(self) -> list:
        if os.path.exists(SIGNAL_LOG_FILE):
            try:
                with open(SIGNAL_LOG_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save(self, records: list) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(SIGNAL_LOG_FILE, "w") as f:
            json.dump(records, f, indent=2, default=str)


# ============================================================
# Module-level singletons
# ============================================================
_cooldown_tracker        = CooldownTracker()
_signal_logger           = SignalLogger()
_last_divergent_tf_warning: dict | None = None  # written by _detect_extreme


# ============================================================
# Public API
# ============================================================

def run_len(symbol: str) -> dict | None:
    """Run the Len Agent scan for one symbol.

    Key behaviours:
    - Parallel klines fetch (ThreadPoolExecutor) — ~3× faster than sequential
    - Direction-aware cooldown — oversold/overbought tracked independently
    - Specific exception handling — API vs data errors distinguished in logs
    - All signals persisted to len_signals.json for observability

    Returns:
        dict with signal details if a condition is confirmed, else None.
    """
    scans = _scan_all_timeframes(symbol)
    if not scans:
        _log.error("All timeframe scans failed for %s", symbol)
        return None

    signal = _detect_extreme(scans)
    if signal is None:
        return None

    # Determine direction for cooldown check
    is_oversold = (signal["flag_type"] in ("OVERSOLD", "EXTREME_OVERSOLD") or
                   signal.get("reversal_signal") == "REVERSAL_UP")

    trigger_rsis = signal.get("rsi_details", {})
    avg_rsi = (sum(trigger_rsis.values()) / len(trigger_rsis)) if trigger_rsis else None

    if _cooldown_tracker.is_on_cooldown(current_rsi=avg_rsi, oversold=is_oversold):
        rem  = _cooldown_tracker.remaining_cooldown(is_oversold)
        mins = int(rem.total_seconds() // 60)
        secs = int(rem.total_seconds() % 60)
        _log.info("Cooldown[%s] active — %dm%ds remaining",
                  "oversold" if is_oversold else "overbought", mins, secs)
        return None

    # Attach metadata consumed by downstream (orchestrator, Telegram).
    # Prefer 1m as the canonical price source — most recent close, deterministic.
    _price_src = scans.get("1m") or next(iter(scans.values()))
    signal["symbol_name"]   = symbol
    signal["current_price"] = _price_src["current_price"]
    signal["scans"]         = scans

    _cooldown_tracker.record_action(rsi=avg_rsi, oversold=is_oversold)
    _signal_logger.log_signal(signal, symbol)

    _log.info("SIGNAL FIRED: %s | %s | severity=%s | reason=%s",
              symbol, signal["flag_type"], signal["severity"], signal["trigger_reason"])
    return signal


def format_len_report(signal: dict | None) -> str:
    """Format Len Agent result for console / Telegram output."""
    if signal is None:
        lines: list[str] = []

        # DIVERGENT_TF warning (set by _detect_extreme when 1m ↔ 15m conflict)
        if _last_divergent_tf_warning:
            w = _last_divergent_tf_warning
            lines.append(
                f"[Len] DIVERGENT_TF: 1m RSI={w['1m_rsi']} vs 15m RSI={w['15m_rsi']} "
                f"— TFs conflict, pipeline NOT triggered."
            )

        # Active cooldown(s)
        cd_parts: list[str] = []
        for label, oversold in [("oversold", True), ("overbought", False)]:
            rem = _cooldown_tracker.remaining_cooldown(oversold)
            if rem.total_seconds() > 0:
                mins = int(rem.total_seconds() // 60)
                secs = int(rem.total_seconds() % 60)
                cd_parts.append(f"{label}:{mins}m{secs}s")
        if cd_parts:
            lines.append(f"[Len] Cooldown active — {', '.join(cd_parts)}")
        elif not _last_divergent_tf_warning:
            lines.append("[Len] No oversold/overbought detected — skipping 3-Agent.")

        return "\n".join(lines) if lines else "[Len] No signal."

    sev_prefix = {"CRITICAL": "!!! ", "HIGH": "!! ", "MEDIUM": "! ", "WARNING": "~ "}.get(
        signal.get("severity", ""), "")

    lines = [
        f"[Len] {sev_prefix}=== {signal['flag_type']} [{signal.get('severity')}] ===",
        f"  Symbol: {signal['symbol_name']}  |  Price: {signal['current_price']}",
        f"  Reason: {signal.get('trigger_reason')}",
        f"  TFs:    {', '.join(signal['trigger_timeframes'])}",
        f"  RSI:    {signal['rsi_details']}",
        f"  Trend:  {signal['technical_trend']}  →  {signal['reversal_signal']}",
    ]

    for tf, info in signal.get("early_details", {}).items():
        parts = [f"{k}={v}" for k, v in info.items()]
        lines.append(f"  [{tf}] {', '.join(parts)}")

    # BB squeeze across all scanned timeframes
    for tf, data in signal.get("scans", {}).items():
        bb = data.get("bollinger_bands")
        if bb and bb.get("squeeze"):
            lines.append(f"  [{tf}] BB SQUEEZE (bandwidth={bb['bandwidth']:.4f}) — breakout imminent")

    stats = _signal_logger.get_stats()
    lines.append(
        f"  Stats: {stats['signals_last_hour']} signals/hr | "
        f"{stats['total_signals']} total | {stats['false_positives']} FP evaluated"
    )
    lines.append("  >> Triggering full 3-Agent analysis...")
    return "\n".join(lines)
