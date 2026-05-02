"""
Intent Classifier — single LLM call that drives the entire routing pipeline.

WHY structured function-calling:
- OpenAI function calling returns validated JSON matching our schema
- One call extracts: intent, entities, target agent, safety verdict — all at once
- Cheaper and faster than chaining multiple calls

FOLLOW-UP RESOLUTION:
- We pass the last N conversation turns as context to the LLM
- The system prompt instructs the model to resolve pronouns and references
  ("what about Apple?" after "tell me about Microsoft")
- Topic switches are handled by instructing the model to classify ONLY the
  current turn's intent, using context only for entity resolution

FALLBACK BEHAVIOUR:
- If the LLM call fails (timeout, API error, malformed response), we route
  to `general_query` with empty entities. This ensures the request never crashes.
- The error is logged for observability.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from openai import OpenAI

from src.config import get_settings
from src.models import ClassifierResult
from src.session import Turn

logger = logging.getLogger(__name__)

# ── Agent taxonomy — must match fixtures exactly ──────────────────────────────

AGENT_TAXONOMY = [
    "portfolio_health",
    "market_research",
    "investment_strategy",
    "financial_planning",
    "financial_calculator",
    "risk_assessment",
    "product_recommendation",
    "predictive_analysis",
    "customer_support",
    "general_query",
]

# ── System prompt for the classifier ─────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for a wealth management platform called Valura.
Your job is to analyze a user's query and return structured classification data.

## Agent Taxonomy (choose exactly one):
- portfolio_health: assessment of user's portfolio — concentration, performance, benchmarking, health check, diversification review
- market_research: factual info about instruments, sectors, market events, prices, news, comparisons
- investment_strategy: advice/strategy — buy/sell/rebalance, allocation guidance
- financial_planning: long-term planning — retirement, goals, savings rate, FIRE
- financial_calculator: deterministic computation — DCA, mortgage, tax, future value, FX conversion
- risk_assessment: risk metrics, exposure analysis, what-if scenarios, stress tests, beta, drawdown
- product_recommendation: recommend specific products/funds matching profile
- predictive_analysis: forward-looking analysis — forecasts, trend extrapolation
- customer_support: platform issues, account questions, how-to-use-app
- general_query: educational, conversational, definitions, greetings, thanks, gibberish

## Entity Extraction Rules:
- tickers: array of uppercase ticker symbols. Map company names to tickers (Apple→AAPL, Nvidia→NVDA, Microsoft→MSFT, Tesla→TSLA, Google→GOOGL, Amazon→AMZN, Meta→META, HSBC→HSBA.L, Barclays→BARC.L, Toyota→7203.T, ASML→ASML.AS). If user says just a ticker or company name with no other context, route to market_research.
- amount: number in the unit of currency
- currency: ISO 4217 (USD, EUR, GBP, JPY)
- rate: decimal (8% = 0.08)
- period_years: integer
- frequency: daily | weekly | monthly | yearly
- horizon: 6_months | 1_year | 5_years
- time_period: today | this_week | this_month | this_year
- topics: array of relevant topic strings
- sectors: array of sector strings (use lowercase: "technology", "healthcare", etc.)
- index: exact canonical name (S&P 500, FTSE 100, NIKKEI 225, MSCI World)
- action: buy | sell | hold | hedge | rebalance
- goal: retirement | education | house | FIRE | emergency_fund

## Rules:
1. Classify based on the PRIMARY intent of the current query.
2. If the query has multiple intents, pick the dominant one.
3. For follow-up queries (e.g. "what about Apple?" or "how much do I own?"), use conversation history to resolve entities but classify the CURRENT turn's intent independently.
4. For greetings (hi, hello, thanks, thx), always route to general_query with empty entities.
5. For gibberish or unclear input, route to general_query.
6. Only include entities you can confidently extract. Don't guess.
7. For queries about "my portfolio", "my holdings", "health check", "diversification", "am I beating the market" → portfolio_health.
8. A bare ticker (e.g. just "AAPL") with no other context should route to market_research.
9. "how is my portfolio doing and what should i sell?" → portfolio_health (primary intent), include action: "sell" in entities.

## Safety Verdict:
Return "safe" for normal queries, or a brief note like "mentions_insider_info" if the query touches sensitive financial topics. This is INFORMATIONAL ONLY — it does not block anything."""

# ── OpenAI function schema for structured output ─────────────────────────────

CLASSIFY_FUNCTION = {
    "name": "classify_intent",
    "description": "Classify a user query into an agent and extract entities",
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "enum": AGENT_TAXONOMY,
                "description": "The target agent to route this query to",
            },
            "intent": {
                "type": "string",
                "description": "Human-readable intent label (e.g. 'check_portfolio', 'get_price', 'greeting')",
            },
            "entities": {
                "type": "object",
                "description": "Extracted entities from the query",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "amount": {"type": "number"},
                    "currency": {"type": "string"},
                    "rate": {"type": "number"},
                    "period_years": {"type": "integer"},
                    "frequency": {"type": "string", "enum": ["daily", "weekly", "monthly", "yearly"]},
                    "horizon": {"type": "string"},
                    "time_period": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "sectors": {"type": "array", "items": {"type": "string"}},
                    "index": {"type": "string"},
                    "action": {"type": "string", "enum": ["buy", "sell", "hold", "hedge", "rebalance"]},
                    "goal": {"type": "string", "enum": ["retirement", "education", "house", "FIRE", "emergency_fund"]},
                },
                "additionalProperties": False,
            },
            "safety_verdict": {
                "type": "string",
                "description": "Informational safety assessment: 'safe' or a brief note",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0 and 1",
            },
        },
        "required": ["agent", "intent", "entities", "safety_verdict", "confidence"],
        "additionalProperties": False,
    },
}


def _build_messages(
    query: str,
    conversation_history: list[Turn] | None = None,
) -> list[dict[str, str]]:
    """Build the messages array for the OpenAI API call."""
    messages = [{"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT}]

    # Add conversation history for follow-up resolution
    if conversation_history:
        for turn in conversation_history:
            messages.append({"role": turn.role, "content": turn.content})

    messages.append({"role": "user", "content": query})
    return messages


def classify(
    query: str,
    conversation_history: list[Turn] | None = None,
    llm: Any = None,
) -> ClassifierResult:
    """
    Classify a user query into an agent + entities using a single LLM call.

    Args:
        query: The user's input text
        conversation_history: Prior turns for follow-up resolution
        llm: Injectable LLM client (for testing). If None, uses real OpenAI.

    Returns:
        ClassifierResult with agent, intent, entities, safety_verdict

    Fallback:
        On any failure → general_query with empty entities
    """
    # If an injected mock is provided, use it directly
    if llm is not None:
        try:
            result = llm(query, conversation_history)
            if isinstance(result, ClassifierResult):
                return result
            if isinstance(result, dict):
                return ClassifierResult(**result)
            return ClassifierResult(agent="general_query", intent="fallback", entities={})
        except Exception as e:
            logger.warning(f"Mock LLM failed: {e}")
            return ClassifierResult(agent="general_query", intent="fallback", entities={})

    # Real OpenAI call
    try:
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)
        messages = _build_messages(query, conversation_history)

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            functions=[CLASSIFY_FUNCTION],
            function_call={"name": "classify_intent"},
            temperature=0.0,  # Deterministic for consistent routing
            timeout=settings.llm_timeout_seconds,
        )

        # Extract function call result
        fn_call = response.choices[0].message.function_call
        if fn_call and fn_call.arguments:
            parsed = json.loads(fn_call.arguments)
            # Clean up entities — remove None/empty values
            entities = {k: v for k, v in parsed.get("entities", {}).items() if v is not None}
            return ClassifierResult(
                agent=parsed.get("agent", "general_query"),
                intent=parsed.get("intent", ""),
                entities=entities,
                safety_verdict=parsed.get("safety_verdict", "safe"),
                confidence=parsed.get("confidence", 1.0),
            )

    except Exception as e:
        logger.error(f"Classifier LLM call failed: {e}", exc_info=True)

    # Fallback — never crash
    return ClassifierResult(
        agent="general_query",
        intent="fallback",
        entities={},
        safety_verdict="safe",
        confidence=0.0,
    )
