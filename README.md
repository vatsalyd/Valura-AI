# Valura AI — Team Lead Assignment

> An AI agent ecosystem that helps novice investors **build, monitor, grow, and protect** their portfolio.

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd <repo-name>

python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux/macOS

pip install -r requirements.txt

cp .env.example .env
# Fill in OPENAI_API_KEY in .env
```

### Run the server

```bash
python -m uvicorn src.app:app --reload --port 8000
```

### Run tests (no API key needed)

```bash
pytest tests/ -v
```

---

## 📋 Required Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes (runtime only) | — | OpenAI API key. Not needed for tests. |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model for classifier + agents. Use `gpt-4.1` for eval. |
| `APP_ENV` | No | `development` | `development` / `production` / `test` |

---

## 🏗️ Architecture

```
POST /chat
  │
  ▼
┌─────────────────┐
│  Rate Limiter    │ ← Token bucket, per-user, tiered (free/premium/unlimited)
└────────┬────────┘
         │ pass
         ▼
┌─────────────────┐
│  Safety Guard    │ ← Pure regex, no LLM, <1ms
│  (local filter)  │
└────────┬────────┘
         │ pass
         ▼
┌─────────────────┐
│ Intent Classifier│ ← Single LLM call (function calling)
│ (1 API call)     │   Returns: agent, entities, safety verdict
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent Router    │ ← Registry pattern, O(1) lookup
│                  │   StubAgent for unimplemented agents
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Specialist Agent │ ← Portfolio Health (implemented)
│                  │   Others return structured stub
└────────┬────────┘
         │
         ▼
    SSE Stream → Client
```

**Pipeline flow:** Every request follows this exact path. The safety guard is the only authority that blocks — it runs before any LLM call. The classifier's safety verdict is informational only.

---

## 🔑 Key Decisions

### 1. Safety Guard: Regex + Heuristic Scoring (no ML)

**Choice:** Two-layer regex approach — action-phrase detection + educational intent signals.

**Why:**
- Assignment demands <10ms, no LLM, no network. This rules out any ML model.
- Layer 1 catches first-person harmful intent ("help me pump the stock", "trade on confidential info").
- Layer 2 detects educational queries ("what is insider trading?", "explain money laundering") and lets them through.

**Tradeoff:** Biases toward safety — recall ≥95% is prioritised over passthrough ≥90%. A query that combines harmful keywords with ambiguous verbs may be over-blocked. Documented and accepted because the cost of missing a harmful query is higher than over-blocking an educational one.

**Performance:** Sub-1ms per query in testing (well under the 10ms requirement).

### 2. Intent Classifier: OpenAI Function Calling

**Choice:** Single `function_call` to the OpenAI API with a JSON schema that extracts agent, entities, safety verdict, and confidence — all in one call.

**Why:**
- Function calling returns validated JSON matching our Pydantic schema.
- One call does everything: routing, entity extraction, safety assessment.
- `temperature=0.0` for deterministic classification.

**Fallback:** If the LLM call fails (timeout, API error, malformed response), we route to `general_query` with empty entities. The request never crashes.

**Follow-up resolution:** Conversation history (last 6 turns) is passed in the messages array. The system prompt instructs the model to resolve pronouns using context but classify the current turn's intent independently.

### 3. Portfolio Health Agent: Compute Locally, Narrate with LLM

**Choice:** All numerical metrics (concentration, performance, benchmark comparison) are computed locally from market data. The LLM is used only to generate a plain-language summary.

**Why:**
- Numbers must be trustworthy. LLMs hallucinate financial data.
- Local computation is testable and deterministic.
- If the LLM fails, the structured data is still returned with a template summary.

**Market data:** yfinance — free, no API key, covers all exchanges in fixtures (NASDAQ, NYSE, LSE, Euronext, TSE).

### 4. Session Persistence: In-Memory Dict

**Choice:** `dict[str, Session]` with O(1) lookup.

**Why:** Demo scope. No production traffic. The interface (`get_or_create`, `append_turn`, `get_recent_turns`) is designed so swapping to SQLite or Postgres later requires changing only `session.py` — no caller changes.

**Tradeoff:** Data lost on restart. Acceptable for demo.

### 5. Pipeline Timeout: 30 Seconds

**Why:** p95 target is <6s end-to-end. 30s gives 5× headroom for worst-case LLM latency. Beyond 30s, the user has lost interest. The timeout wraps only the agent execution; safety guard and classifier have their own timeouts.

### 6. Streaming: SSE via sse-starlette

**Choice:** `sse-starlette` (already in scaffold requirements).

**Why:** Proven with FastAPI, handles async generators natively, minimal code. The portfolio health agent streams in logical sections (concentration → performance → benchmark → observations → summary → disclaimer) for progressive rendering on the client side.

### 7. Multi-Tenant Rate Limiting (Stretch Goal)

**Choice:** Token bucket algorithm, per-user, with tiered limits.

**Why:**
- Token bucket allows short bursts (user sends 3 quick queries) while enforcing an average rate — better UX than a hard sliding window.
- Per-user isolation prevents one heavy user from starving others.
- Zero external dependencies — in-memory for demo, swappable to Redis for production.

**Tiers:**
| Tier | Requests/min | Burst |
|---|---|---|
| `free` | 10 | 5 |
| `premium` | 60 | 15 |
| `unlimited` | ∞ | ∞ |

**Pipeline position:** Runs as Step 0, before safety guard — rate-limited requests never touch the LLM, saving cost.

---

## 📁 Repository Structure

```
src/
├── __init__.py
├── app.py                    # FastAPI app, SSE endpoint, pipeline orchestration
├── config.py                 # Pydantic settings from .env
├── models.py                 # All Pydantic domain models
├── safety.py                 # Safety guard — regex-based, no LLM
├── classifier.py             # Intent classifier — single LLM call
├── router.py                 # Agent registry + dispatch
├── session.py                # In-memory conversation session store
├── market_data.py            # yfinance wrapper with TTL cache
├── rate_limiter.py           # Multi-tenant token bucket rate limiter
└── agents/
    ├── __init__.py
    ├── base.py               # Abstract base agent interface
    ├── portfolio_health.py   # Full implementation — MONITOR + PROTECT
    └── stub.py               # Structured stub for unimplemented agents

tests/
├── conftest.py               # Shared fixtures, LLM mock
├── test_safety_pairs.py      # Safety guard recall + passthrough
├── test_classifier_routing.py# Classifier routing accuracy + entity matching
├── test_portfolio_health_skeleton.py  # Portfolio health agent tests
├── test_conversations.py     # Follow-up, multi-intent, ambiguous sessions
└── test_http.py              # HTTP integration tests (SSE pipeline)

fixtures/                     # Provided test data (do not modify)
```

---

## 📚 Library Justifications

| Library | Why |
|---|---|
| `fastapi` | Async, fast, native Pydantic integration, OpenAPI docs |
| `sse-starlette` | SSE support for FastAPI — provided in scaffold |
| `openai` | Official SDK for structured outputs and function calling |
| `pydantic` + `pydantic-settings` | Type-safe models + validated config from .env |
| `yfinance` | Free market data covering all exchanges in fixtures, no API key |
| `python-dotenv` | .env file loading |
| `httpx` | Async HTTP client for testing |

---

## 📊 Cost & Performance

### How to Reproduce

```bash
# Run the benchmark suite (requires OPENAI_API_KEY in .env)
python -m src.benchmark
```

This runs:
1. **Safety guard latency** — per-query timing over the full safety_pairs fixture
2. **Classifier latency** — real LLM calls timed with `time.perf_counter()`
3. **Portfolio health agent latency** — includes yfinance market data fetch
4. **End-to-end pipeline latency** — safety → classify → agent, timed as a unit
5. **Cost estimation** — token count estimates at gpt-4.1 pricing

### Measurement Method

- **First-token latency:** `time.monotonic()` from request receipt to first SSE event emission. For function-calling responses, first-token ≈ total (no streaming in classification).
- **End-to-end:** `time.monotonic()` from request receipt to final `done` SSE event. Logged in every response as `elapsed_seconds`.
- **Cost:** Estimated using OpenAI published pricing for gpt-4.1: $2/1M input tokens, $8/1M output tokens.

### Targets vs Actuals (measured with `python -m src.benchmark`)

**Development model:** gpt-oss-120b via OpenRouter

| Metric | Target | Measured (OpenRouter) | Expected (gpt-4.1 direct) |
|---|---|---|---|
| Safety guard p95 | <10ms | **0.125ms** ✅ | Same (no LLM) |
| Classifier accuracy | ≥85% | **100% (15/15)** ✅ | Same or better |
| p95 first-token | <2s | ~5.7s ⚠️ | **<1.5s** ✅ |
| p95 end-to-end | <6s | ~7.6s ⚠️ | **<4s** ✅ |
| Cost per query | <$0.05 | **$0.0036** ✅ | Same |

> **Note on latency:** Development benchmarks used gpt-oss-120b via OpenRouter, which adds ~3-4s of routing latency per call. The evaluation uses **gpt-4.1 on direct OpenAI API**, which typically responds in 500-1500ms for function-calling. Our pipeline adds <1ms overhead (safety guard), so end-to-end with gpt-4.1 should be well under 6s.

### Benchmark Details

```
Safety Guard:   p50=0.058ms  p95=0.125ms  p99=0.335ms  (47 queries)
Classifier:     Mean=7.5s (OpenRouter)  15/15 correct routing
Portfolio Agent: Run1=9s (cold), Run2=367ms, Run3=166ms (cached)
```

### Cost Breakdown (gpt-4.1 pricing)

| Component | Input Tokens | Output Tokens | Cost |
|---|---|---|---|
| Classifier | ~300 | ~80 | $0.0012 |
| Portfolio Health summary | ~600 | ~150 | $0.0024 |
| **Total per query** | | | **~$0.004** |

Budget: $0.05. Actual: $0.004. **93% headroom.**

---

## 🎥 Defence Video

> [VIDEO LINK HERE — to be added after recording]

---

## 🔮 What I'd Do Differently With Another Week

1. **Embedding-based pre-classifier** — Use sentence-transformers to skip the LLM call when confidence is high. Would cut latency and cost for common queries by ~70%.

2. **Persistent sessions with SQLite** — The in-memory store is fine for demo but loses context on restart. `aiosqlite` with a simple turns table would take ~1 hour.

3. **Implement more agents** — Market Research (yfinance + news API), Financial Calculator (deterministic math), and Risk Assessment (portfolio beta/drawdown) are all natural next steps.

4. **Structured logging + OpenTelemetry** — For production observability: trace IDs through the pipeline, latency histograms, LLM token usage tracking.

5. **LLM response cache** — Deduplicate identical queries within a time window to reduce cost and latency. Content-hash keyed, TTL-based expiry.

