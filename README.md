# FinSight AI — Intelligent Portfolio Co-Pilot

> A real-time AI microservice that analyzes investment portfolios, classifies financial queries, and streams actionable insights — all protected by a sub-millisecond safety layer.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-30%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ What It Does

FinSight AI takes a plain-English financial question, understands the intent, and routes it to a specialist agent that computes real metrics from live market data — streamed back in real-time via SSE.

**Example:**
```
User: "How is my portfolio doing?"
→ Safety guard (0.1ms) → Intent classified as portfolio_health (99% confidence)
→ Live market data fetched → Concentration risk, returns, benchmark alpha computed
→ Streamed back as structured SSE events in ~5s
```

### Key Capabilities

| Feature | Description |
|---|---|
| 🛡️ **Safety Guard** | Two-layer regex filter blocks harmful financial queries in <1ms while passing educational ones through |
| 🧠 **Intent Classifier** | Single LLM tool-call classifies queries across 10 financial domains with 100% accuracy on test suite |
| 📊 **Portfolio Analysis** | Concentration risk, CAGR, benchmark alpha, sector/currency diversification — all computed locally from live data |
| 🌊 **SSE Streaming** | Progressive response delivery — clients render results as they arrive |
| 🔒 **Rate Limiting** | Per-user token bucket with tiered access (free/premium/unlimited) |
| 🔌 **Provider Agnostic** | Works with OpenAI, OpenRouter, Azure OpenAI, Ollama — any compatible provider via config |

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd finsight-ai

python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux/macOS

pip install -r requirements.txt

# Configure LLM provider (see Environment Variables below)
cp .env.example .env
```

### Run the server

```bash
python -m uvicorn src.app:app --reload --port 8000
# Open http://localhost:8000 for Swagger UI
```

### Run tests (no API key needed)

```bash
pytest tests/ -v
# All 30 tests pass without any external API calls
```

### Run benchmarks

```bash
python -m src.benchmark
# Measures safety guard latency, classifier accuracy, end-to-end pipeline
```

---

## 📋 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes (runtime) | — | API key for your LLM provider. Not needed for tests. |
| `OPENAI_BASE_URL` | No | — | Custom base URL for OpenAI-compatible providers (e.g. `https://openrouter.ai/api/v1`). Leave empty for direct OpenAI. |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model identifier. Examples: `gpt-4.1`, `openai/gpt-oss-120b` |
| `APP_ENV` | No | `development` | `development` / `production` / `test` |

### Provider Configuration Examples

```bash
# Direct OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1

# OpenRouter (free credits, no credit card)
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-oss-120b

# Local Ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3
```

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
│ Intent Classifier│ ← Single LLM call (tool calling)
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
│ Specialist Agent │ ← Portfolio Health (fully implemented)
│                  │   Others return structured stub
└────────┬────────┘
         │
         ▼
    SSE Stream → Client
```

Every request follows this exact path. The safety guard is the only authority that blocks — it runs before any LLM call. The classifier's safety verdict is informational only.

---

## 🔑 Design Decisions

### 1. Safety Guard: Two-Layer Regex with Educational Bypass

**Approach:** Action-phrase detection + educational intent signals, no ML or LLM.

**Why regex over ML:**
- Sub-millisecond execution (0.125ms p95 — 80x under the 10ms target)
- Zero external dependencies, zero network calls
- Deterministic — same input always produces same output

**The two layers:**
- **Layer 1 (action phrases):** Catches first-person harmful intent — "help me pump the stock", "trade on confidential info". Patterns match the *concept*, not exact phrasings.
- **Layer 2 (educational bypass):** Detects learning queries — "what is insider trading?", "explain how regulators detect fraud" — and passes them through even if they contain harmful keywords.

**Tradeoff:** Biases toward safety — recall ≥95% is prioritised over passthrough ≥90%. The cost of missing a harmful query (regulatory risk) outweighs over-blocking an educational one (minor UX friction).

### 2. Intent Classifier: Single LLM Tool Call

**Approach:** One `tools` call to the LLM with a JSON schema that extracts agent, entities, safety verdict, and confidence simultaneously.

**Why one call, not multiple:**
- Function/tool calling returns validated JSON matching our Pydantic schema
- Single call reduces latency and cost vs. chaining multiple prompts
- `temperature=0.0` for deterministic, reproducible classification

**Follow-up resolution:** The last 6 conversation turns are passed in the messages array. The system prompt instructs the model to resolve pronouns ("what about Apple?" after "tell me about Microsoft") but classify the current turn's intent independently.

**Fallback:** If the LLM call fails for any reason — timeout, API error, malformed response — we route to `general_query` with empty entities and confidence 0.0. The request never crashes.

### 3. Portfolio Health Agent: Local Math + LLM Narrative

**Approach:** All numerical metrics computed deterministically in Python. The LLM is used only for generating a plain-language summary.

**Why not ask the LLM to analyze the portfolio:**
- LLMs hallucinate numbers. A financial co-pilot with wrong numbers is worse than no answer.
- Local computation is deterministic and unit-testable without mocking LLM responses.
- If the LLM summary generation fails, structured data is still returned with a template fallback.

**What's computed locally:**
- Concentration risk (top position %, top 3 %, flag: low/moderate/high)
- Total return and annualized return (CAGR formula using actual purchase dates)
- Benchmark alpha (portfolio return vs user's preferred benchmark over same period)
- Sector and currency diversification breakdowns
- Context-aware observations (retiree income focus, FX risk, concentration warnings)

**Market data:** yfinance — free, no API key needed, covers NASDAQ, NYSE, LSE, Euronext, and TSE. Wrapped with a 5-minute TTL cache that reduces repeat queries from ~9s (cold) to ~166ms (cached).

### 4. Provider-Agnostic LLM Configuration

**Problem:** OpenAI's API requires a credit card for billing. OpenRouter provides free credits with an OpenAI-compatible API.

**Solution:** Three config fields — `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`. The classifier conditionally passes `base_url` to the OpenAI client only when set. This means the same codebase works with:
- OpenAI directly (no base URL needed)
- OpenRouter (`OPENAI_BASE_URL=https://openrouter.ai/api/v1`)
- Azure OpenAI, Ollama, vLLM — any compatible provider

Switching providers is a one-line `.env` change with **zero code modifications.**

### 5. Graceful Degradation at Every Layer

Every component has a defined failure mode — nothing crashes:

| Component | Failure | Fallback |
|---|---|---|
| LLM classifier | Timeout / API error | Route to `general_query`, confidence 0.0 |
| Market data (yfinance) | Ticker not found | Use `avg_cost` from user fixtures |
| LLM summary | Generation fails | Template-based summary from computed metrics |
| Rate limiter | User exhausts quota | Structured SSE error with `retry_after_seconds` |
| Agent not implemented | Route to missing agent | `StubAgent` returns classified intent + entities |
| Pipeline timeout | Agent takes >30s | SSE timeout event, connection closed cleanly |
| Empty portfolio | User has no positions | Personalized onboarding guidance based on risk profile |

### 6. Session Store: Interface-First Design

**Approach:** In-memory `dict[str, Session]` with O(1) lookup.

The interface — `get_or_create()`, `append_turn()`, `get_recent_turns()` — is designed as a clean abstraction boundary. Swapping to SQLite (`aiosqlite`) or Postgres (`asyncpg`) means changing only `session.py`. No callers need to change.

### 7. Rate Limiting: Token Bucket with Tiered Access

**Approach:** Per-user token bucket algorithm with three tiers.

**Why token bucket over sliding window:** Allows natural bursts — a user sending 3 quick queries doesn't get blocked — while enforcing the average rate. Better UX for interactive use.

| Tier | Requests/min | Burst |
|---|---|---|
| `free` | 10 | 5 |
| `premium` | 60 | 15 |
| `unlimited` | ∞ | ∞ |

**Pipeline position:** Runs as Step 0, before the safety guard — rate-limited requests never touch the LLM, saving cost.

---

## 📁 Project Structure

```
src/
├── __init__.py
├── app.py                    # FastAPI app, SSE endpoint, pipeline orchestration
├── config.py                 # Pydantic settings from .env
├── models.py                 # All Pydantic domain models (shared data contracts)
├── safety.py                 # Safety guard — regex-based, no LLM
├── classifier.py             # Intent classifier — single LLM tool call
├── router.py                 # Agent registry + dispatch
├── session.py                # In-memory conversation session store
├── market_data.py            # yfinance wrapper with TTL cache
├── rate_limiter.py           # Multi-tenant token bucket rate limiter
├── benchmark.py              # Performance measurement suite
└── agents/
    ├── __init__.py
    ├── base.py               # Abstract base agent interface
    ├── portfolio_health.py   # Full implementation — analysis + streaming
    └── stub.py               # Structured stub for unimplemented agents

tests/
├── conftest.py               # Shared fixtures, LLM mock
├── test_safety_pairs.py      # Safety guard recall + passthrough
├── test_classifier_routing.py# Classifier accuracy + entity extraction
├── test_portfolio_health_skeleton.py  # Agent computation tests
├── test_conversations.py     # Follow-up, multi-intent, ambiguous sessions
├── test_http.py              # HTTP integration tests (SSE pipeline)
└── test_rate_limiter.py      # Rate limiter burst, refill, tier tests

fixtures/                     # User profiles, test queries, safety pairs
```

---

## 📚 Technology Stack

| Library | Why |
|---|---|
| `fastapi` | Async-native, Pydantic integration, auto-generated OpenAPI docs |
| `sse-starlette` | Server-Sent Events for FastAPI — enables progressive streaming |
| `openai` | Official SDK — works with any OpenAI-compatible provider via `base_url` |
| `pydantic` + `pydantic-settings` | Type-safe models + validated config from `.env` at boot |
| `yfinance` | Free market data covering 5 exchanges, no API key required |
| `httpx` | Async HTTP client for integration testing |
| `pytest` + `pytest-asyncio` | Async test framework with fixture-based mocking |

---

## 📊 Performance & Cost

### Benchmarks (measured with `python -m src.benchmark`)

| Metric | Target | Measured | Status |
|---|---|---|---|
| Safety guard p95 | <10ms | **0.125ms** | ✅ 80x under limit |
| Classifier accuracy | ≥85% | **100% (15/15)** | ✅ |
| Cost per query | <$0.05 | **$0.0036** | ✅ 93% under budget |
| p95 end-to-end | <6s | ~5.3s (cached) | ✅ |

### Benchmark Details

```
Safety Guard:   p50=0.058ms  p95=0.125ms  p99=0.335ms  (47 queries)
Classifier:     15/15 correct routing across 10 agent categories
Portfolio Agent: Run1=9s (cold), Run2=367ms, Run3=166ms (TTL cached)
```

### Cost Breakdown

| Component | Input Tokens | Output Tokens | Cost |
|---|---|---|---|
| Classifier | ~300 | ~80 | $0.0012 |
| Portfolio Health summary | ~600 | ~150 | $0.0024 |
| **Total per query** | | | **~$0.004** |

> Pricing based on OpenAI gpt-4.1: $2/1M input, $8/1M output.

---

## 🗺️ Roadmap

- [ ] **Embedding-based pre-classifier** — sentence-transformers for common intents in <50ms, LLM fallback for ambiguous queries. Cuts latency and cost by ~70%.
- [ ] **Persistent sessions** — Migrate from in-memory dict to `aiosqlite`. Interface already designed for the swap.
- [ ] **More agents** — Market Research (yfinance + news), Financial Calculator (deterministic math), Risk Assessment (beta/drawdown).
- [ ] **Structured logging + OpenTelemetry** — Trace IDs through the pipeline, latency histograms, token usage tracking.
- [ ] **Response cache** — Content-hash keyed, TTL-based deduplication for identical queries.
- [ ] **WebSocket support** — Bidirectional streaming for richer client interactions.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/market-research-agent`)
3. Run tests (`pytest tests/ -v`) — all 30 must pass
4. Submit a pull request

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
