# AgentGoodForCrypto — CLAUDE.md

## Project overview

Hệ thống phân tích crypto tự động theo thời gian thực, gồm một **gatekeeper agent** (Len) và một **3-agent pipeline** (AnalysisData → ReadNews → Orchestrator). Kết quả được gửi qua Telegram và kết hợp với **Betting agent** mô phỏng giao dịch futures ảo.

Toàn bộ chỉ báo kỹ thuật (RSI, MACD, Stochastic, Bollinger Bands, v.v.) được tính thuần Python — không có thư viện TA. LLM calls chỉ dùng OpenAI, không có Anthropic SDK trong project này.

---

## Architecture

```
main.py (main loop)
│
├── Len Agent  [src/Agent3/Len.py]          — pure math, no LLM
│     Scan 1m/5m/15m mỗi LOOP_INTERVAL giây
│     Nếu phát hiện signal → trigger pipeline
│
└── 3-Agent Pipeline (triggered by Len)
      ├── AnalysisData  [src/Agent3/AnalysisData.py]   — kỹ thuật + OpenAI
      ├── ReadNews      [src/Agent3/ReadNews.py]        — Fear & Greed + OpenAI
      └── Orchestrator  [src/Agent3/orchestrator.py]   — tổng hợp + OpenAI
            └── Betting [src/Agent3/Betting.py]         — position management, no LLM
```

---

## File structure

```
AgentGoodForCrypto/
├── main.py                         — entry point, main loop
├── api_server.py                   — FastAPI server (xem data qua HTTP)
├── benchmark.py                    — benchmark indicators
├── requirements.txt
├── .env                            — credentials (không commit)
│
├── src/
│   ├── Agent3/
│   │   ├── Len.py                  — gatekeeper: RSI/MACD/Stoch/BB/divergence
│   │   ├── AnalysisData.py         — technical analysis + OpenAI interpretation
│   │   ├── ReadNews.py             — Fear & Greed index + OpenAI sentiment
│   │   ├── orchestrator.py         — synthesis + memory + OpenAI final verdict
│   │   └── Betting.py              — virtual futures position manager
│   │
│   ├── API/
│   │   ├── GetDataCrypto.py        — Binance REST API wrapper
│   │   ├── GetNews.py              — alternative.me Fear & Greed API
│   │   └── mock_data.py            — mock data cho development/testing
│   │
│   ├── SystemPrompt/
│   │   ├── analysis_data_prompt.txt
│   │   ├── read_news_prompt.txt
│   │   └── orchestrator_prompt.txt
│   │
│   └── tracker.py                  — CostTracker + AccuracyTracker
│
└── data/                           — auto-created, gitignored
    ├── memory.json                 — 10 rounds gần nhất (orchestrator memory)
    ├── cost_log.json               — chi phí OpenAI per round
    ├── accuracy_log.json           — lịch sử dự đoán và kết quả thực tế
    ├── betting.json                — trạng thái portfolio futures
    ├── len_signals.json            — log mọi signal Len đã fire (max 500)
    └── len_cooldown_state.json     — cooldown state (persist qua restart)
```

---

## How to run

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Copy và điền .env (xem mục Environment Variables bên dưới)

# Chạy bot chính
python main.py

# Chạy API server (port 8000)
uvicorn api_server:app --reload

# Mock data (không cần Binance API)
USE_MOCK_DATA=true python main.py
```

---

## Environment variables (.env)

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
SEND_TELEGRAM_MESSAGES=true       # false = chỉ print ra console

# Binance symbol
CRYPTO_SYMBOL=BTCUSDT

# Vòng lặp
LOOP_INTERVAL_MIN=10              # giây, min delay giữa Len scans
LOOP_INTERVAL_MAX=30              # giây, max delay (random giữa min và max)
LOOP_MAX_ITERATIONS=0             # 0 = chạy mãi mãi

# Tắt/bật Len gatekeeper
LEN_AGENT_ENABLED=true            # false = chạy 3-agent mỗi cycle (tốn tiền hơn)

# Mock data
USE_MOCK_DATA=false

# Len Agent cooldown
COOLDOWN_PERIOD=15                # phút suppress re-trigger cùng hướng

# OpenAI model config (per agent)
ANALYTICS_MODEL=gpt-3.5-turbo
NEWS_MODEL=gpt-3.5-turbo
ORCHESTRA_MODEL=gpt-3.5-turbo
ANALYTICS_MODEL_TEMPERATURE=0.7
ANALYTICS_MAX_TOKENS=1500
# (tương tự cho NEWS_ và ORCHESTRA_)

# Betting (futures simulation)
MONEY=100                         # USD vốn ban đầu ảo
FUTURE_LEVERAGE=5                 # đòn bẩy
NEUTRAL_CLOSE_MIN_HOLD_MINUTES=5  # giữ tối thiểu bao nhiêu phút trước khi close khi NEUTRAL
```

---

## Len Agent — signal priority

Không dùng LLM. Thuần indicator math trên 3 timeframes (1m, 5m, 15m) fetch song song.

| Priority | Tên | Điều kiện | Severity |
|----------|-----|-----------|----------|
| P1 | RSI Divergence | Price lower low + RSI higher low (bullish), hoặc ngược lại. Chỉ tính khi RSI ≤ 45 (bull) hoặc RSI ≥ 55 (bear) | MEDIUM/HIGH |
| P2 | MACD/Stoch cross + RSI near threshold | MACD hoặc Stochastic cross + RSI trong buffer 5pt của threshold | MEDIUM |
| P3 | ≥2 TF đồng thuận | RSI oversold/overbought trên 2+ timeframes | MEDIUM/HIGH |
| P4 | Extreme RSI | RSI ≤ 20 hoặc RSI ≥ 80. Nếu chỉ 1m: bắt buộc cần volume spike | CRITICAL |

**DIVERGENT_TF**: 1m và 15m chỉ hướng ngược nhau → log warning, KHÔNG trigger pipeline.

**Cooldown**: 15 phút per direction (oversold/overbought độc lập). Override cooldown nếu RSI deepens thêm 5pt.

### Hằng số quan trọng trong Len.py (có thể tune)

```python
COOLDOWN_PERIOD = 15          # phút
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_EXTREME_OVERSOLD = 20
RSI_EXTREME_OVERBOUGHT = 80
MACD_RSI_BUFFER = 5           # P2: RSI buffer
PIVOT_WINGS = 3               # Williams fractal wings (divergence)
DIVERGENCE_LOOKBACK = 20      # bars
RSI_VELOCITY_BARS = 3
RSI_DEEPENING_THRESHOLD = 5   # để override cooldown
VOLUME_SPIKE_RATIO = 2.0      # 2× average volume
BB_SQUEEZE_THRESHOLD = 0.02   # 2% bandwidth
KLINES_LIMIT = 150            # bars fetch per TF
MIN_KLINES = 60               # minimum để proceed
```

---

## 3-Agent pipeline

### AnalysisData
- Fetch klines 1m/5m/15m/30m từ Binance
- Tính SMA, RSI, MACD per timeframe
- Gọi OpenAI để interpret và build structured JSON (timeframes, scenarios, overall_trend)
- Fallback về rule-based nếu OpenAI trả non-JSON

### ReadNews
- Fetch Fear & Greed Index 10 ngày từ alternative.me
- Analyze trend (IMPROVING / DECLINING / RECOVERING / FALLING / STABLE)
- Gọi OpenAI để summarize sentiment
- Output signal: STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL

### Orchestrator
- Load memory 10 rounds gần nhất từ `data/memory.json`
- Evaluate accuracy của prediction trước (so sánh price_at_prediction với current_price)
- `synthesize_results()`: tính confidence_score từ technical trend + sentiment (rule-based, không LLM)
- `build_signal_summary()`: pre-compute bullish/bearish counts per TF để giảm LLM hallucination
- Gọi OpenAI để viết final verdict
- Hardcoded rule: nếu RSI 15m/30m < 20 mà technical = BEARISH → downgrade sang NEUTRAL (extreme oversold reversal guard)

---

## Betting Agent

Mô phỏng virtual futures, không có tiền thật.

- BULLISH/SLIGHTLY_BULLISH → LONG
- BEARISH/SLIGHTLY_BEARISH → SHORT
- NEUTRAL → close position (nếu đã hold đủ `NEUTRAL_CLOSE_MIN_HOLD_MINUTES`)
- Signal flip (LONG→BEARISH) → close + open ngược

State persist tại `data/betting.json`. PnL = `position_size × price_change_pct` (có leverage).

---

## Data layer

### API wrappers (`src/API/`)

| Function | Nguồn | Dùng ở đâu |
|----------|-------|-----------|
| `get_klines(symbol, interval, limit)` | Binance `/api/v3/klines` | Len + AnalysisData |
| `get_current_price(symbol)` | Binance `/api/v3/ticker/price` | AnalysisData |
| `get_ticker_24h(symbol)` | Binance `/api/v3/ticker/24hr` | AnalysisData |
| `get_fear_and_greed(limit)` | alternative.me API | ReadNews |

Tất cả có mock mode: `USE_MOCK_DATA=true` → dùng `src/API/mock_data.py`.

### Trackers (`src/tracker.py`)

**CostTracker**: ghi cost per OpenAI call, tổng per round, lưu vào `data/cost_log.json`.

**AccuracyTracker**: so sánh predicted_trend với actual price movement round sau. Logic:
- BULLISH → cần price đi UP hoặc FLAT
- BEARISH → cần price đi DOWN hoặc FLAT
- NEUTRAL → cần price không vượt `neutral_band_pct` (dynamic, từ 24h ATR)

---

## Key design decisions

1. **Len là gatekeeper** — tránh gọi OpenAI mỗi 10-30 giây, chỉ gọi khi market có signal thực sự. Tiết kiệm 90%+ API cost.

2. **Indicator tự tính, không dùng TA-Lib** — portable, không cần native dependencies. Trade-off: kém optimize hơn nhưng đủ dùng.

3. **`calculate_rsi_series()` là single-pass O(n)** — dùng chung cho divergence detection và velocity, tránh gọi `calculate_rsi()` nhiều lần.

4. **`_macd_cross_from_dicts()`** — tách sign-flip logic ra khỏi data fetching để `_scan_timeframe` chỉ gọi `calculate_macd()` 2 lần thay vì 3.

5. **Direction-aware cooldown** — oversold và overbought track riêng biệt, tránh block overbought signal khi vừa xong oversold cooldown.

6. **Memory 10 rounds** — orchestrator dùng lịch sử để detect reversal pattern (nếu 3 rounds liên tiếp BULLISH mà score âm → `reversal_signal = REVERSAL_DOWN`).

7. **`build_signal_summary()` pre-compute** — đếm bullish/bearish indicators trước khi gửi LLM để giảm hallucination và đảm bảo LLM đề cập mixed signals.

---

## Common tasks

**Thêm timeframe mới cho Len:**
Thêm vào `TIMEFRAME_CONFIG` trong `Len.py`. Conflict check DIVERGENT_TF hiện dùng cứng "1m" vs "15m" — cần update nếu thay đổi.

**Thêm symbol:**
Đổi `CRYPTO_SYMBOL` trong `.env`. Không cần thay đổi code.

**Reset betting portfolio:**
Xóa `data/betting.json`.

**Reset cooldown Len:**
Xóa `data/len_cooldown_state.json`.

**Xem signal history:**
Đọc `data/len_signals.json` (max 500 records, có `outcome` field để đánh giá false positive).

**Tune P1 divergence zone:**
Tìm `RSI_OVERSOLD + 15` và `RSI_OVERBOUGHT - 15` trong `Len.py:_detect_extreme()`. Buffer 15pt có thể tăng để strict hơn.
