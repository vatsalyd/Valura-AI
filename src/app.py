"""
FastAPI HTTP layer — the single entry point for the AI microservice.

PIPELINE:
  1. Receive ChatRequest
  2. Safety Guard (local, <10ms) → if blocked, stream error SSE and stop
  3. Intent Classifier (1 LLM call) → structured ClassifierResult
  4. Agent Router → dispatches to the right agent (or StubAgent)
  5. Agent → streams response via SSE

SSE EVENT TYPES:
  - "safety_block": query was blocked by safety guard
  - "classification": classifier result (agent, entities, safety verdict)
  - "agent_response": streamed chunks from the agent
  - "done": pipeline complete
  - "error": structured error (never a stack trace)

TIMEOUT:
  30 seconds — chosen because:
  - p95 target is <6s end-to-end
  - 30s gives 5x headroom for worst-case LLM latency
  - Beyond 30s, the user has already lost interest
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from src.config import get_settings
from src.models import ChatRequest, UserProfile, SafetyVerdict
from src.safety import check as safety_check
from src.classifier import classify
from src.router import get_agent
from src.session import store as session_store
from src.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    _load_user_fixtures()
    logger.info(f"Loaded {len(_USER_CACHE)} user fixtures")
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Valura AI",
    description="AI co-investor microservice — build, monitor, grow, protect.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── User fixtures loader (for demo — in production this comes from a DB) ─────

_USER_CACHE: dict[str, UserProfile] = {}


def _load_user_fixtures() -> None:
    """Load user fixtures from disk into memory."""
    from pathlib import Path
    fixtures_dir = Path(__file__).parent.parent / "fixtures" / "users"
    if not fixtures_dir.exists():
        return
    for path in fixtures_dir.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                import json as _json
                data = _json.load(f)
                user = UserProfile(**data)
                _USER_CACHE[user.user_id] = user
        except Exception as e:
            logger.warning(f"Failed to load user fixture {path}: {e}")



def get_user(user_id: str) -> UserProfile:
    """Get user profile. Falls back to a minimal profile if not found."""
    if user_id in _USER_CACHE:
        return _USER_CACHE[user_id]
    return UserProfile(user_id=user_id, name="Unknown User")


# ── SSE pipeline ──────────────────────────────────────────────────────────────

PIPELINE_TIMEOUT = 30.0  # seconds


async def _pipeline_stream(request: ChatRequest) -> AsyncIterator[dict]:
    """
    The full pipeline as an async generator of SSE events.
    Safety → Classify → Route → Agent → Stream
    """
    start_time = time.monotonic()
    session_id = request.session_id or str(uuid.uuid4())

    # ── Step 0: Rate Limiting ─────────────────────────────────────────
    allowed, retry_after = rate_limiter.check(request.user_id)
    if not allowed:
        yield {
            "event": "error",
            "data": json.dumps({
                "error": "rate_limited",
                "message": f"Too many requests. Please retry after {retry_after}s.",
                "retry_after_seconds": retry_after,
                "tier": rate_limiter.get_user_tier(request.user_id),
            }),
        }
        return

    # Get user profile
    user = request.user_profile or get_user(request.user_id)

    # ── Step 1: Safety Guard ──────────────────────────────────────────
    verdict: SafetyVerdict = safety_check(request.query)
    if verdict.blocked:
        yield {
            "event": "safety_block",
            "data": json.dumps({
                "blocked": True,
                "category": verdict.category,
                "message": verdict.message,
            }),
        }
        return

    # ── Step 2: Intent Classifier ─────────────────────────────────────
    # Get conversation history for follow-up resolution
    session = session_store.get_or_create(session_id, request.user_id)
    history = session_store.get_recent_turns(session_id, max_turns=6)

    classification = classify(request.query, conversation_history=history)

    yield {
        "event": "classification",
        "data": json.dumps({
            "agent": classification.agent,
            "intent": classification.intent,
            "entities": classification.entities,
            "safety_verdict": classification.safety_verdict,
            "confidence": classification.confidence,
        }),
    }

    # Save this turn to session history
    session_store.append_turn(session_id, "user", request.query)

    # ── Step 3: Route to Agent ────────────────────────────────────────
    agent = get_agent(classification.agent)

    # ── Step 4: Stream Agent Response ─────────────────────────────────
    try:
        elapsed = time.monotonic() - start_time
        remaining = max(PIPELINE_TIMEOUT - elapsed, 5.0)
        deadline = time.monotonic() + remaining

        async for chunk in agent.stream(request.query, user, classification.entities):
            if time.monotonic() > deadline:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "error": "timeout",
                        "message": "The request timed out. Please try again.",
                    }),
                }
                return
            yield {
                "event": "agent_response",
                "data": chunk,
            }

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        yield {
            "event": "error",
            "data": json.dumps({
                "error": "agent_error",
                "message": "An error occurred while processing your request.",
            }),
        }
        return

    # ── Step 5: Done ──────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time
    yield {
        "event": "done",
        "data": json.dumps({
            "session_id": session_id,
            "elapsed_seconds": round(elapsed, 2),
        }),
    }


# ── HTTP endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    """
    Main endpoint — accepts a user query and runs the full pipeline.
    Response is streamed via SSE. No JSON fallback.
    """
    return EventSourceResponse(
        _pipeline_stream(request),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "valura-ai"}


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

