"""
Benchmark script — measures p95 latency, first-token time, and cost per query.

Run with a valid OPENAI_API_KEY:
    python -m src.benchmark

Outputs a table of metrics for the README.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv()

from src.classifier import classify
from src.safety import check as safety_check
from src.agents.portfolio_health import PortfolioHealthAgent
from src.models import UserProfile, ClassifierResult
from src.market_data import get_ticker_data, TickerData


# ── Test queries covering different agents ────────────────────────────────────

BENCHMARK_QUERIES = [
    "how is my portfolio doing?",
    "what's the price of AAPL right now?",
    "should I buy more NVDA?",
    "hello",
    "what is a mutual fund?",
    "calculate mortgage for 500k at 6.5% for 30 years",
    "how much should I save for retirement?",
    "is my portfolio well diversified?",
    "tell me about Tesla",
    "review my holdings",
    "what's my downside risk if markets drop 30%?",
    "recommend a dividend ETF",
    "portfolio summary",
    "compare AAPL and MSFT",
    "rebalance my portfolio",
]


def _load_test_user() -> UserProfile:
    """Load user_001 for benchmark testing."""
    path = Path(__file__).parent.parent / "fixtures" / "users" / "user_001_active_trader_us.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return UserProfile(**data)


def benchmark_safety_guard():
    """Benchmark safety guard latency."""
    print("\n" + "=" * 60)
    print("SAFETY GUARD BENCHMARK")
    print("=" * 60)

    # Load safety queries
    path = Path(__file__).parent.parent / "fixtures" / "test_queries" / "safety_pairs.json"
    with open(path, encoding="utf-8") as f:
        queries = json.load(f)["queries"]

    latencies = []
    for case in queries:
        start = time.perf_counter()
        safety_check(case["query"])
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    print(f"  Queries:     {len(latencies)}")
    print(f"  Mean:        {statistics.mean(latencies):.3f} ms")
    print(f"  p50:         {statistics.median(latencies):.3f} ms")
    print(f"  p95:         {sorted(latencies)[int(len(latencies) * 0.95)]:.3f} ms")
    print(f"  p99:         {sorted(latencies)[int(len(latencies) * 0.99)]:.3f} ms")
    print(f"  Max:         {max(latencies):.3f} ms")


def benchmark_classifier():
    """Benchmark classifier latency (requires OPENAI_API_KEY)."""
    print("\n" + "=" * 60)
    print("CLASSIFIER BENCHMARK")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-..."):
        print("  ⚠ Skipped — no valid OPENAI_API_KEY")
        return [], []

    first_token_latencies = []
    total_latencies = []

    for query in BENCHMARK_QUERIES:
        start = time.perf_counter()
        result = classify(query)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        # For function calling, first-token ≈ total (single response)
        first_token_latencies.append(elapsed)
        total_latencies.append(elapsed)

        print(f"  {query[:50]:50s} → {result.agent:25s} {elapsed:7.0f}ms")

    if total_latencies:
        sorted_ft = sorted(first_token_latencies)
        sorted_total = sorted(total_latencies)
        p95_idx = int(len(sorted_total) * 0.95)

        print(f"\n  Queries:     {len(total_latencies)}")
        print(f"  Mean:        {statistics.mean(total_latencies):.0f} ms")
        print(f"  p50:         {statistics.median(total_latencies):.0f} ms")
        print(f"  p95 first-token: {sorted_ft[p95_idx]:.0f} ms")
        print(f"  p95 total:   {sorted_total[p95_idx]:.0f} ms")
        print(f"  Max:         {max(total_latencies):.0f} ms")

    return first_token_latencies, total_latencies


def benchmark_portfolio_health():
    """Benchmark portfolio health agent (requires market data)."""
    print("\n" + "=" * 60)
    print("PORTFOLIO HEALTH AGENT BENCHMARK")
    print("=" * 60)

    user = _load_test_user()
    agent = PortfolioHealthAgent()

    latencies = []
    for i in range(3):
        start = time.perf_counter()
        result = asyncio.run(agent.run("how is my portfolio doing?", user, {}))
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.0f}ms")

    print(f"\n  Mean:        {statistics.mean(latencies):.0f} ms")
    print(f"  Note: First run includes yfinance cold-start; subsequent runs use cache")


def estimate_cost():
    """Estimate cost per query at gpt-4.1 pricing."""
    print("\n" + "=" * 60)
    print("COST ESTIMATE (gpt-4.1 pricing)")
    print("=" * 60)

    # gpt-4.1 pricing (as of 2025):
    # Input: $2.00 / 1M tokens
    # Output: $8.00 / 1M tokens
    input_price = 2.00 / 1_000_000
    output_price = 8.00 / 1_000_000

    # Classifier: ~300 input tokens (system prompt + query), ~80 output tokens
    classifier_input = 300
    classifier_output = 80
    classifier_cost = classifier_input * input_price + classifier_output * output_price

    # Portfolio Health summary: ~600 input tokens, ~150 output tokens
    agent_input = 600
    agent_output = 150
    agent_cost = agent_input * input_price + agent_output * output_price

    total = classifier_cost + agent_cost

    print(f"  Classifier:  ~{classifier_input} input + ~{classifier_output} output tokens = ${classifier_cost:.4f}")
    print(f"  Agent:       ~{agent_input} input + ~{agent_output} output tokens = ${agent_cost:.4f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Total:       ${total:.4f} per query")
    print(f"  Budget:      $0.05 per query")
    print(f"  Margin:      {((0.05 - total) / 0.05) * 100:.0f}% under budget")


def benchmark_end_to_end():
    """Benchmark the full pipeline (safety + classifier + agent)."""
    print("\n" + "=" * 60)
    print("END-TO-END PIPELINE BENCHMARK")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-..."):
        print("  ⚠ Skipped — no valid OPENAI_API_KEY")
        return

    user = _load_test_user()
    agent = PortfolioHealthAgent()
    latencies = []

    test_queries = [
        "how is my portfolio doing?",
        "portfolio summary",
        "is my portfolio well diversified?",
    ]

    for query in test_queries:
        start = time.perf_counter()

        # Safety
        verdict = safety_check(query)
        safety_time = time.perf_counter() - start

        # Classify
        result = classify(query)
        classify_time = time.perf_counter() - start

        # Agent
        agent_result = asyncio.run(agent.run(query, user, {}))
        total_time = time.perf_counter() - start

        total_ms = total_time * 1000
        latencies.append(total_ms)
        print(f"  {query[:45]:45s} → {total_ms:7.0f}ms (safety: {safety_time*1000:.1f}ms, classify: {classify_time*1000:.0f}ms)")

    if latencies:
        sorted_l = sorted(latencies)
        p95_idx = min(int(len(sorted_l) * 0.95), len(sorted_l) - 1)
        print(f"\n  p95 end-to-end: {sorted_l[p95_idx]:.0f} ms")
        print(f"  Target:         6000 ms")
        status = "✅ PASS" if sorted_l[p95_idx] < 6000 else "❌ FAIL"
        print(f"  Status:         {status}")


if __name__ == "__main__":
    print("Valura AI — Performance Benchmark")
    print("Model:", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    print("=" * 60)

    benchmark_safety_guard()
    benchmark_classifier()
    benchmark_portfolio_health()
    estimate_cost()
    benchmark_end_to_end()

    print("\n" + "=" * 60)
    print("DONE — Copy the above into your README under '📊 Cost & Performance'")
    print("=" * 60)
