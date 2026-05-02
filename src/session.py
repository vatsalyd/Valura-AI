"""
In-memory session store for conversation history.

WHY in-memory:
- Assignment scope is a demo — no prod traffic, no persistence needed across restarts
- Dict-based store is O(1) lookup, zero dependencies, trivial to test
- The interface (get/append/clear) is designed so swapping to SQLite/Postgres
  later requires changing only this module — no caller changes needed

TRADEOFF:
- Data lost on restart. Acceptable for demo; unacceptable for production.
- For prod: swap to aiosqlite (lightweight) or asyncpg (scalable).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    """A single turn in a conversation."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class Session:
    """A conversation session with its history."""
    session_id: str
    user_id: str
    turns: list[Turn] = field(default_factory=list)


class SessionStore:
    """Thread-safe (GIL-protected) in-memory session store."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str, user_id: str) -> Session:
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(
                session_id=session_id, user_id=user_id,
            )
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.turns.append(Turn(role=role, content=content))

    def get_recent_turns(self, session_id: str, max_turns: int = 10) -> list[Turn]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.turns[-max_turns:]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# Global singleton — agents and the HTTP layer import this directly.
store = SessionStore()
