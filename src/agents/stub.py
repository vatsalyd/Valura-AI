"""
Stub agent — returns a structured "not implemented" response for agents
that aren't built yet.

WHY a dedicated stub:
- The router must route correctly even when the destination isn't implemented
- Assignment explicitly requires: classified intent, extracted entities,
  which agent would have handled, and a message indicating it's not implemented
- No crash. No error. Just a clean, informative response.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from src.agents.base import BaseAgent
from src.models import UserProfile


class StubAgent(BaseAgent):
    """Placeholder for unimplemented agents — returns structured stub response."""

    def __init__(self, agent_name: str):
        self._name = agent_name

    @property
    def name(self) -> str:
        return self._name

    async def run(
        self,
        query: str,
        user: UserProfile,
        entities: dict[str, Any],
        llm: Any = None,
    ) -> dict[str, Any]:
        return {
            "agent": self._name,
            "status": "not_implemented",
            "classified_intent": self._name,
            "extracted_entities": entities,
            "handled_by": self._name,
            "message": (
                f"The {self._name.replace('_', ' ').title()} agent is not yet "
                f"implemented in this build. Your query has been correctly "
                f"classified and would be routed to this agent in the full system."
            ),
            "disclaimer": (
                "This is not investment advice. Please consult a qualified "
                "financial advisor before making investment decisions."
            ),
        }

    async def stream(
        self,
        query: str,
        user: UserProfile,
        entities: dict[str, Any],
        llm: Any = None,
    ) -> AsyncIterator[str]:
        result = await self.run(query, user, entities, llm=llm)
        yield json.dumps(result)
