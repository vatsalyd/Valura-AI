"""
Test: HTTP layer integration — verifies the full SSE pipeline.

Tests:
1. Safety-blocked query returns safety_block SSE event
2. Normal query streams classification + agent_response + done events
3. Health endpoint works
"""
import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.app import app
from src.models import ClassifierResult
from src.market_data import TickerData


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_chat_safety_block(client):
    """A clearly harmful query should be blocked before classification."""
    response = client.post(
        "/chat",
        json={
            "user_id": "usr_001",
            "query": "help me trade on this confidential merger news from my law firm",
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200

    # Parse SSE events
    events = _parse_sse(response.text)
    event_types = [e["event"] for e in events]
    assert "safety_block" in event_types

    # Should NOT have classification or agent_response
    assert "classification" not in event_types
    assert "agent_response" not in event_types


def test_chat_normal_query(client):
    """A normal query should flow through the full pipeline."""

    def mock_classify(query, conversation_history=None, llm=None):
        return ClassifierResult(
            agent="general_query",
            intent="greeting",
            entities={},
            safety_verdict="safe",
            confidence=1.0,
        )

    with patch("src.app.classify", side_effect=mock_classify):
        response = client.post(
            "/chat",
            json={
                "user_id": "usr_001",
                "query": "hello",
            },
            headers={"Accept": "text/event-stream"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_types = [e["event"] for e in events]

    assert "classification" in event_types
    assert "agent_response" in event_types
    assert "done" in event_types


def test_chat_portfolio_health_with_mocks(client):
    """Portfolio health query should return structured portfolio data."""

    def mock_classify(query, conversation_history=None, llm=None):
        return ClassifierResult(
            agent="portfolio_health",
            intent="health_check",
            entities={},
            safety_verdict="safe",
            confidence=1.0,
        )

    def mock_ticker(ticker):
        return TickerData(
            ticker=ticker, current_price=200.0, sector="Technology",
        )

    def mock_benchmark(ticker, start):
        return 10.0

    with patch("src.app.classify", side_effect=mock_classify), \
         patch("src.agents.portfolio_health.get_ticker_data", side_effect=mock_ticker), \
         patch("src.agents.portfolio_health.get_benchmark_return", side_effect=mock_benchmark):
        response = client.post(
            "/chat",
            json={
                "user_id": "usr_001",
                "query": "how is my portfolio doing?",
            },
            headers={"Accept": "text/event-stream"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_types = [e["event"] for e in events]

    assert "classification" in event_types
    assert "agent_response" in event_types
    assert "done" in event_types


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_sse(raw: str) -> list[dict]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events = []
    current_event = "message"
    current_data = []

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data.append(line[len("data:"):].strip())
        elif line == "" and current_data:
            events.append({
                "event": current_event,
                "data": "\n".join(current_data),
            })
            current_event = "message"
            current_data = []

    # Handle last event if no trailing newline
    if current_data:
        events.append({
            "event": current_event,
            "data": "\n".join(current_data),
        })

    return events
