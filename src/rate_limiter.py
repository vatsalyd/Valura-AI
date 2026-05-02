"""
Multi-tenant rate limiter — token bucket algorithm, per-user.

WHY token bucket:
- Allows bursts (user sends 3 quick queries) while enforcing average rate
- Simple, memory-efficient, no external dependencies (no Redis needed for demo)
- Per-user isolation — one heavy user can't starve others

TIER SYSTEM:
- free: 10 requests/minute, burst of 5
- premium: 60 requests/minute, burst of 15
- unlimited: no rate limiting (internal/admin)

In production, tiers would come from a user database. Here we use a simple
config lookup with a default tier.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenBucket:
    """A single user's rate limit state."""
    capacity: float          # Max tokens (burst size)
    refill_rate: float       # Tokens added per second
    tokens: float = 0.0      # Current token count
    last_refill: float = 0.0 # Timestamp of last refill

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def try_consume(self, cost: float = 1.0) -> bool:
        """
        Try to consume `cost` tokens. Returns True if allowed, False if rate limited.
        Refills tokens based on elapsed time before checking.
        """
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    @property
    def retry_after_seconds(self) -> float:
        """How many seconds until at least 1 token is available."""
        if self.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self.tokens
        return deficit / self.refill_rate if self.refill_rate > 0 else 60.0


# ── Tier definitions ──────────────────────────────────────────────────────────

@dataclass
class RateLimitTier:
    name: str
    requests_per_minute: float
    burst_size: float


TIERS: dict[str, RateLimitTier] = {
    "free": RateLimitTier(name="free", requests_per_minute=10, burst_size=5),
    "premium": RateLimitTier(name="premium", requests_per_minute=60, burst_size=15),
    "unlimited": RateLimitTier(name="unlimited", requests_per_minute=1e9, burst_size=1e9),
}

DEFAULT_TIER = "free"


# ── Rate limiter store ────────────────────────────────────────────────────────

class RateLimiter:
    """Per-user rate limiter using token buckets."""

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._user_tiers: dict[str, str] = {}  # user_id → tier name

    def set_user_tier(self, user_id: str, tier: str) -> None:
        """Assign a tier to a user. Resets their bucket if tier changes."""
        old_tier = self._user_tiers.get(user_id)
        if old_tier != tier:
            self._user_tiers[user_id] = tier
            self._buckets.pop(user_id, None)  # Reset bucket on tier change

    def check(self, user_id: str) -> tuple[bool, Optional[float]]:
        """
        Check if a request from user_id is allowed.

        Returns:
            (allowed, retry_after_seconds)
            - (True, None) if allowed
            - (False, seconds) if rate limited
        """
        bucket = self._get_or_create_bucket(user_id)
        if bucket.try_consume():
            return True, None
        return False, round(bucket.retry_after_seconds, 1)

    def _get_or_create_bucket(self, user_id: str) -> TokenBucket:
        if user_id not in self._buckets:
            tier_name = self._user_tiers.get(user_id, DEFAULT_TIER)
            tier = TIERS.get(tier_name, TIERS[DEFAULT_TIER])
            self._buckets[user_id] = TokenBucket(
                capacity=tier.burst_size,
                refill_rate=tier.requests_per_minute / 60.0,
            )
        return self._buckets[user_id]

    def get_user_tier(self, user_id: str) -> str:
        return self._user_tiers.get(user_id, DEFAULT_TIER)


# Global singleton
rate_limiter = RateLimiter()
