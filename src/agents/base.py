"""
Base agent interface — all agents implement this contract.

WHY an abstract base:
- Enforces a uniform interface: every agent has `run()` and `stream()`
- Makes the router generic — it doesn't need to know about specific agents
- Adding a new agent = subclass BaseAgent + register in the router
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from src.models import UserProfile


class BaseAgent(ABC):
    """Abstract base for all specialist agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier matching the taxonomy."""
        ...

    @abstractmethod
    async def run(
        self,
        query: str,
        user: UserProfile,
        entities: dict[str, Any],
        llm: Any = None,
    ) -> dict[str, Any]:
        """
        Execute the agent and return structured output.

        Args:
            query: The user's original query
            user: The user's profile including portfolio
            entities: Extracted entities from the classifier
            llm: Injectable LLM client (for testing)

        Returns:
            Structured dict response
        """
        ...

    async def stream(
        self,
        query: str,
        user: UserProfile,
        entities: dict[str, Any],
        llm: Any = None,
    ) -> AsyncIterator[str]:
        """
        Stream the agent's response as SSE-compatible chunks.
        Default implementation: run() then yield the full result.
        Override for true streaming.
        """
        result = await self.run(query, user, entities, llm=llm)
        yield json.dumps(result, default=str)
