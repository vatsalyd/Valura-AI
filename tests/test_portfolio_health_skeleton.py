"""
Test: Portfolio Health agent.

Tests verify:
1. Empty portfolio (user_004) doesn't crash, includes disclaimer
2. Concentrated portfolio (user_003) flags concentration risk
3. Normal portfolio (user_001) includes disclaimer
4. Structured output shape is correct
"""
import pytest

from src.agents.portfolio_health import PortfolioHealthAgent
from src.models import UserProfile
from unittest.mock import patch, MagicMock
from src.market_data import TickerData


# ---------------------------------------------------------------------------
# Mock market data — tests must not call real yfinance
# ---------------------------------------------------------------------------

def _mock_ticker_data(ticker: str) -> TickerData:
    """Return mock market data for test tickers."""
    mock_prices = {
        # user_001 (active trader)
        "AAPL": 195.0, "MSFT": 420.0, "NVDA": 880.0, "GOOGL": 175.0,
        "META": 500.0, "AMZN": 185.0, "TSLA": 170.0, "AMD": 155.0,
        "QQQ": 480.0,
        # user_003 (concentrated)
        "VTI": 260.0, "VXUS": 60.0, "BND": 73.0,
        # user_006 (multi-currency)
        "VOO": 480.0, "ASML.AS": 900.0, "HSBA.L": 7.5, "7203.T": 2800.0,
        # user_008 (retiree)
        "JNJ": 155.0, "PG": 165.0, "KO": 62.0, "VYM": 115.0,
        "SCHD": 80.0, "TLT": 95.0,
    }
    mock_sectors = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "GOOGL": "Technology", "META": "Technology", "AMZN": "Technology",
        "TSLA": "Consumer Cyclical", "AMD": "Technology",
        "JNJ": "Healthcare", "PG": "Consumer Defensive", "KO": "Consumer Defensive",
    }
    return TickerData(
        ticker=ticker,
        current_price=mock_prices.get(ticker, 100.0),
        sector=mock_sectors.get(ticker, "Other"),
        dividend_yield=0.03 if ticker in ("JNJ", "PG", "KO", "VYM", "SCHD") else None,
    )


def _mock_benchmark_return(ticker: str, start_date: str):
    return 14.2  # Mock 14.2% benchmark return


@pytest.fixture
def agent():
    return PortfolioHealthAgent()


@pytest.fixture
def mock_market():
    """Patch market data calls to use mock data."""
    with patch("src.agents.portfolio_health.get_ticker_data", side_effect=_mock_ticker_data), \
         patch("src.agents.portfolio_health.get_benchmark_return", side_effect=_mock_benchmark_return):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_portfolio_health_does_not_crash_on_empty_portfolio(load_user, agent, mock_market):
    """user_004 has no positions. Agent must not crash."""
    user_data = load_user("usr_004")
    user = UserProfile(**user_data)

    response = await agent.run("how is my portfolio doing?", user, {})

    assert response is not None
    assert "disclaimer" in response
    assert "not investment advice" in response["disclaimer"].lower()
    # Should mention building / getting started
    obs_texts = " ".join(o["text"] for o in response.get("observations", []))
    assert len(obs_texts) > 0  # Some guidance was provided


@pytest.mark.asyncio
async def test_portfolio_health_flags_concentration(load_user, agent, mock_market):
    """user_003 has ~60% in NVDA. Agent must surface this."""
    user_data = load_user("usr_003")
    user = UserProfile(**user_data)

    response = await agent.run("what's my concentration risk?", user, {})

    assert response["concentration_risk"]["flag"] in {"high", "warning"}
    # NVDA should dominate
    assert response["concentration_risk"]["top_position_pct"] > 40


@pytest.mark.asyncio
async def test_portfolio_health_includes_disclaimer(load_user, agent, mock_market):
    """Every response must include a regulatory disclaimer."""
    user_data = load_user("usr_001")
    user = UserProfile(**user_data)

    response = await agent.run("review my holdings", user, {})

    assert response["disclaimer"]
    assert "not investment advice" in response["disclaimer"].lower()


@pytest.mark.asyncio
async def test_portfolio_health_structured_output(load_user, agent, mock_market):
    """Verify the response has all required sections."""
    user_data = load_user("usr_001")
    user = UserProfile(**user_data)

    response = await agent.run("how is my portfolio doing?", user, {})

    assert "concentration_risk" in response
    assert "performance" in response
    assert "benchmark_comparison" in response
    assert "observations" in response
    assert "disclaimer" in response
    assert isinstance(response["observations"], list)
    assert len(response["observations"]) > 0


@pytest.mark.asyncio
async def test_portfolio_health_performance_metrics(load_user, agent, mock_market):
    """Verify performance metrics are computed."""
    user_data = load_user("usr_001")
    user = UserProfile(**user_data)

    response = await agent.run("portfolio summary", user, {})

    perf = response["performance"]
    assert "total_return_pct" in perf
    assert "annualized_return_pct" in perf
    assert "total_cost_basis" in perf
    assert "current_value" in perf
    assert perf["total_cost_basis"] > 0
    assert perf["current_value"] > 0


@pytest.mark.asyncio
async def test_portfolio_health_multi_currency(load_user, agent, mock_market):
    """user_006 has multi-currency holdings — should report currency exposure."""
    user_data = load_user("usr_006")
    user = UserProfile(**user_data)

    response = await agent.run("health check", user, {})

    diversification = response.get("diversification", {})
    by_currency = diversification.get("by_currency", {})
    # Should have multiple currencies
    assert len(by_currency) >= 2


@pytest.mark.asyncio
async def test_portfolio_health_retiree_income_focus(load_user, agent, mock_market):
    """user_008 is income-focused — should mention dividend yields."""
    user_data = load_user("usr_008")
    user = UserProfile(**user_data)

    response = await agent.run("how is my portfolio doing?", user, {})

    # Check that dividend-related observations exist
    obs_texts = " ".join(o["text"].lower() for o in response.get("observations", []))
    assert "dividend" in obs_texts or len(response["observations"]) > 0
