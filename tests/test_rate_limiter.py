"""
Test: Multi-tenant rate limiter.

Tests:
1. Requests within limit are allowed
2. Burst is respected
3. Requests over limit are blocked with retry_after
4. Token refill works
5. Different tiers have different limits
"""
import time
from unittest.mock import patch

from src.rate_limiter import RateLimiter, TokenBucket, TIERS


def test_token_bucket_allows_within_capacity():
    bucket = TokenBucket(capacity=5, refill_rate=1.0)
    for _ in range(5):
        assert bucket.try_consume() is True


def test_token_bucket_blocks_over_capacity():
    bucket = TokenBucket(capacity=3, refill_rate=0.1)
    for _ in range(3):
        bucket.try_consume()
    assert bucket.try_consume() is False


def test_token_bucket_refills_over_time():
    bucket = TokenBucket(capacity=2, refill_rate=10.0)  # 10 tokens/sec
    bucket.try_consume()
    bucket.try_consume()
    assert bucket.try_consume() is False

    # Simulate time passing
    bucket.last_refill -= 1.0  # 1 second ago → 10 tokens refilled
    assert bucket.try_consume() is True


def test_token_bucket_retry_after():
    bucket = TokenBucket(capacity=1, refill_rate=1.0)
    bucket.try_consume()
    assert bucket.retry_after_seconds > 0


def test_rate_limiter_allows_requests():
    rl = RateLimiter()
    allowed, retry = rl.check("user_test")
    assert allowed is True
    assert retry is None


def test_rate_limiter_blocks_after_burst():
    rl = RateLimiter()
    rl.set_user_tier("user_test", "free")  # burst=5
    for _ in range(5):
        allowed, _ = rl.check("user_test")
        assert allowed is True

    allowed, retry = rl.check("user_test")
    assert allowed is False
    assert retry is not None
    assert retry > 0


def test_rate_limiter_premium_has_higher_limits():
    rl = RateLimiter()
    rl.set_user_tier("free_user", "free")      # burst=5
    rl.set_user_tier("premium_user", "premium")  # burst=15

    # Exhaust free user
    for _ in range(5):
        rl.check("free_user")
    free_allowed, _ = rl.check("free_user")

    # Premium should still have capacity
    for _ in range(5):
        rl.check("premium_user")
    premium_allowed, _ = rl.check("premium_user")

    assert free_allowed is False
    assert premium_allowed is True


def test_rate_limiter_unlimited_tier():
    rl = RateLimiter()
    rl.set_user_tier("admin", "unlimited")
    for _ in range(100):
        allowed, _ = rl.check("admin")
        assert allowed is True


def test_rate_limiter_tier_change_resets_bucket():
    rl = RateLimiter()
    rl.set_user_tier("user_test", "free")
    for _ in range(5):
        rl.check("user_test")

    # Upgrade to premium — bucket should reset
    rl.set_user_tier("user_test", "premium")
    allowed, _ = rl.check("user_test")
    assert allowed is True
