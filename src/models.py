"""
Pydantic domain models shared across the entire pipeline.

WHY a single models module:
- Avoids circular imports between safety / classifier / agents
- Serves as the data contract — if a field changes, tests break immediately
- Makes the SSE event shapes explicit and testable
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Agent taxonomy (matches fixtures/test_queries/intent_classification.json) ──

class AgentType(str, Enum):
    PORTFOLIO_HEALTH = "portfolio_health"
    MARKET_RESEARCH = "market_research"
    INVESTMENT_STRATEGY = "investment_strategy"
    FINANCIAL_PLANNING = "financial_planning"
    FINANCIAL_CALCULATOR = "financial_calculator"
    RISK_ASSESSMENT = "risk_assessment"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    CUSTOMER_SUPPORT = "customer_support"
    GENERAL_QUERY = "general_query"


# ── Safety guard ──────────────────────────────────────────────────────────────

class SafetyVerdict(BaseModel):
    blocked: bool = False
    category: Optional[str] = None
    message: str = ""


# ── Classifier output ────────────────────────────────────────────────────────

class ClassifierResult(BaseModel):
    """Structured output returned by the intent classifier."""
    agent: str = Field(description="Target agent from the taxonomy")
    intent: str = Field(default="", description="Human-readable intent label")
    entities: dict[str, Any] = Field(default_factory=dict)
    safety_verdict: str = Field(
        default="safe",
        description="Informational only — does not block",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ── User / portfolio models ──────────────────────────────────────────────────

class Position(BaseModel):
    ticker: str
    exchange: str = ""
    quantity: float
    avg_cost: float
    currency: str = "USD"
    purchased_at: str = ""


class KYC(BaseModel):
    status: str = "pending"


class UserPreferences(BaseModel):
    preferred_benchmark: str = "S&P 500"
    reporting_currency: str = "USD"
    income_focus: bool = False


class UserProfile(BaseModel):
    user_id: str
    name: str = ""
    age: int = 0
    country: str = "US"
    base_currency: str = "USD"
    kyc: KYC = Field(default_factory=KYC)
    risk_profile: str = "moderate"
    positions: list[Position] = Field(default_factory=list)
    preferences: UserPreferences = Field(default_factory=UserPreferences)


# ── Portfolio health agent output ─────────────────────────────────────────────

class ConcentrationRisk(BaseModel):
    top_position_pct: float = 0.0
    top_3_positions_pct: float = 0.0
    flag: str = "low"  # low | moderate | high


class PerformanceMetrics(BaseModel):
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    total_cost_basis: float = 0.0
    current_value: float = 0.0


class BenchmarkComparison(BaseModel):
    benchmark: str = ""
    portfolio_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0


class Observation(BaseModel):
    severity: str = "info"  # info | warning | critical
    text: str = ""


class DiversificationBreakdown(BaseModel):
    by_sector: dict[str, float] = Field(default_factory=dict)
    by_currency: dict[str, float] = Field(default_factory=dict)
    number_of_holdings: int = 0


class PortfolioHealthResponse(BaseModel):
    concentration_risk: ConcentrationRisk = Field(default_factory=ConcentrationRisk)
    performance: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    benchmark_comparison: BenchmarkComparison = Field(default_factory=BenchmarkComparison)
    diversification: DiversificationBreakdown = Field(default_factory=DiversificationBreakdown)
    observations: list[Observation] = Field(default_factory=list)
    summary: str = ""
    disclaimer: str = (
        "This is not investment advice. The information provided is for "
        "educational and informational purposes only. Past performance does "
        "not guarantee future results. Please consult a qualified financial "
        "advisor before making investment decisions."
    )


# ── SSE event shapes ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    session_id: str = ""
    query: str
    # Optional: override user profile for testing
    user_profile: Optional[UserProfile] = None


class SSEEvent(BaseModel):
    event: str = "message"
    data: Any = None
