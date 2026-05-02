"""
Agent Router — dispatches classified queries to the correct agent.

WHY a registry pattern:
- Adding a new agent = register it in AGENT_REGISTRY, done
- No if/elif chains that grow with every new agent
- StubAgent handles all unimplemented agents automatically
"""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.portfolio_health import PortfolioHealthAgent
from src.agents.stub import StubAgent
from src.models import AgentType

# ── Agent registry ────────────────────────────────────────────────────────────
# Only portfolio_health is fully implemented. Everything else gets a StubAgent.

_IMPLEMENTED_AGENTS: dict[str, BaseAgent] = {
    AgentType.PORTFOLIO_HEALTH.value: PortfolioHealthAgent(),
}


def get_agent(agent_name: str) -> BaseAgent:
    """
    Look up the agent by name. Returns the real agent if implemented,
    otherwise a StubAgent that returns a clean "not implemented" response.
    """
    if agent_name in _IMPLEMENTED_AGENTS:
        return _IMPLEMENTED_AGENTS[agent_name]
    return StubAgent(agent_name)
