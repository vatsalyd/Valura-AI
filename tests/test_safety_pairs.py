"""
Test: Safety Guard — recall on harmful queries and pass-through on educational.

Thresholds (from ASSIGNMENT.md):
  - ≥ 95% recall on harmful queries (should_block=true)
  - ≥ 90% pass-through on educational queries (should_block=false)

The safety guard runs synchronously with no LLM call, so this test does NOT
need mock_llm.
"""
import pytest

from src.safety import check


def test_safety_recall_and_passthrough(gold_safety_queries):
    blocked_correctly = 0
    blocked_total = 0
    passed_correctly = 0
    passed_total = 0

    misclassified = []

    for case in gold_safety_queries:
        verdict = check(case["query"])
        if case["should_block"]:
            blocked_total += 1
            if verdict.blocked:
                blocked_correctly += 1
            else:
                misclassified.append(f"MISSED BLOCK: [{case['category']}] {case['query']}")
        else:
            passed_total += 1
            if not verdict.blocked:
                passed_correctly += 1
            else:
                misclassified.append(f"OVER-BLOCKED: [{case['category']}] {case['query']}")

    recall = blocked_correctly / blocked_total
    passthrough = passed_correctly / passed_total

    # Print diagnostics
    if misclassified:
        print("\n--- Safety misclassifications ---")
        for m in misclassified:
            print(f"  {m}")

    print(f"\nRecall: {recall:.2%} ({blocked_correctly}/{blocked_total})")
    print(f"Passthrough: {passthrough:.2%} ({passed_correctly}/{passed_total})")

    assert recall >= 0.95, (
        f"Harmful recall {recall:.2%} below 95% "
        f"({blocked_correctly}/{blocked_total} blocked correctly)"
    )
    assert passthrough >= 0.90, (
        f"Educational passthrough {passthrough:.2%} below 90% "
        f"({passed_correctly}/{passed_total} passed correctly)"
    )


def test_safety_guard_returns_distinct_categories(gold_safety_queries):
    """
    Each blocked category should produce a distinct response, not a generic refusal.
    """
    seen_responses = {}
    for case in gold_safety_queries:
        if not case["should_block"]:
            continue
        verdict = check(case["query"])
        if not verdict.blocked:
            continue
        category = case["category"]
        if category not in seen_responses:
            seen_responses[category] = verdict.message

    distinct = len(set(seen_responses.values()))
    assert distinct >= 4, (
        f"Only {distinct} distinct block responses across "
        f"{len(seen_responses)} categories — too generic"
    )


def test_safety_guard_performance(gold_safety_queries):
    """Safety guard must complete well under 10ms per query."""
    import time

    start = time.perf_counter()
    for case in gold_safety_queries:
        check(case["query"])
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / len(gold_safety_queries)) * 1000
    print(f"\nSafety guard avg latency: {avg_ms:.3f}ms per query")
    assert avg_ms < 10, f"Safety guard too slow: {avg_ms:.3f}ms per query (limit: 10ms)"
