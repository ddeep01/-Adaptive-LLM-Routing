"""
Automated Unit Tests for Routing Engine and Cascade Escalation
"""

import pytest
from app.routing.router import LLMRouter
from app.routing.escalation import EscalationChecker
from app.models.base import ModelResponse

def test_always_small_strategy():
    router = LLMRouter()
    res = router.route_and_generate("Explain quantum mechanics", strategy="always_small")
    assert res["selected_tier"] == "small"
    assert res["routing_strategy"] == "always_small"

def test_always_large_strategy():
    router = LLMRouter()
    res = router.route_and_generate("What is 2+2?", strategy="always_large")
    assert res["selected_tier"] == "large"
    assert res["routing_strategy"] == "always_large"

def test_adaptive_routing_easy():
    router = LLMRouter()
    res = router.route_and_generate("What is the capital of Japan?", strategy="adaptive")
    assert res["difficulty"] == "Easy"
    assert res["selected_tier"] == "small"

def test_escalation_checker_empty_response():
    checker = EscalationChecker()
    mock_resp = ModelResponse(
        query="Test query",
        response_text="",
        model_name="small_mock",
        model_tier="small",
        provider="mock",
        success=False
    )
    should_esc, reason = checker.should_escalate("Test query", mock_resp, "Medium", 0.5)
    assert should_esc is True
    assert "empty" in reason.lower() or "failed" in reason.lower()

def test_escalation_checker_uncertainty():
    checker = EscalationChecker()
    mock_resp = ModelResponse(
        query="Hard math question",
        response_text="I am not sure how to solve this equation.",
        model_name="small_mock",
        model_tier="small",
        provider="mock",
        success=True
    )
    should_esc, reason = checker.should_escalate("Hard math question", mock_resp, "Hard", 0.8)
    assert should_esc is True
    assert "uncertainty" in reason.lower()
