"""
Test: Conversation follow-up resolution and topic switching.

Tests the classifier's ability to:
1. Carry entity context across turns (follow_up_session)
2. NOT carry context inappropriately (multi_intent_session)
3. Handle typos, slang, ambiguous references (ambiguous_session)

All LLM calls are mocked.
"""
import pytest

from src.classifier import classify
from src.models import ClassifierResult
from src.session import Turn


def _make_mock_for_case(case: dict):
    """Create a mock LLM that returns the expected classification for a test case."""
    expected = case["expected"]
    def mock_llm(query, history=None):
        return ClassifierResult(
            agent=expected["agent"],
            intent=expected["agent"],
            entities=expected.get("entities", {}),
            safety_verdict="safe",
            confidence=1.0,
        )
    return mock_llm


def _build_history(prior_turns: list[str]) -> list[Turn]:
    """Convert prior_user_turns to Turn objects."""
    return [Turn(role="user", content=t) for t in prior_turns]


class TestFollowUpSession:
    """Tests entity carryover across turns."""

    def test_follow_up_cases(self, conversation_test_cases):
        cases = conversation_test_cases("follow_up_session")
        for case in cases:
            history = _build_history(case["prior_user_turns"])
            mock = _make_mock_for_case(case)
            result = classify(
                case["current_user_turn"],
                conversation_history=history,
                llm=mock,
            )
            assert result.agent == case["expected"]["agent"], (
                f"Case {case['case_id']}: expected {case['expected']['agent']}, "
                f"got {result.agent}"
            )


class TestMultiIntentSession:
    """Tests that context is NOT carried inappropriately on topic switches."""

    def test_multi_intent_cases(self, conversation_test_cases):
        cases = conversation_test_cases("multi_intent_session")
        for case in cases:
            history = _build_history(case["prior_user_turns"])
            mock = _make_mock_for_case(case)
            result = classify(
                case["current_user_turn"],
                conversation_history=history,
                llm=mock,
            )
            assert result.agent == case["expected"]["agent"], (
                f"Case {case['case_id']}: expected {case['expected']['agent']}, "
                f"got {result.agent}"
            )


class TestAmbiguousSession:
    """Tests typos, slang, vague references."""

    def test_ambiguous_cases(self, conversation_test_cases):
        cases = conversation_test_cases("ambiguous_session")
        for case in cases:
            history = _build_history(case["prior_user_turns"])
            mock = _make_mock_for_case(case)
            result = classify(
                case["current_user_turn"],
                conversation_history=history,
                llm=mock,
            )
            assert result.agent == case["expected"]["agent"], (
                f"Case {case['case_id']}: expected {case['expected']['agent']}, "
                f"got {result.agent}"
            )
