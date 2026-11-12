"""
Automated Unit Tests for Quality Metrics, Latency Profiler, and Cost Calculator
"""

import pytest
from app.evaluation.quality_metrics import (
    compute_exact_match, compute_token_f1, compute_rouge_n, compute_rouge_l, evaluate_quality
)
from app.evaluation.llm_judge import LLMJudge
from app.monitoring.latency import LatencyTracker
from app.monitoring.cost_calculator import CostCalculator

def test_exact_match():
    assert compute_exact_match("Paris", "paris") == 1.0
    assert compute_exact_match("London", "Paris") == 0.0

def test_token_f1():
    score = compute_token_f1("The capital of France is Paris", "Paris is the capital of France")
    assert score == 1.0
    partial = compute_token_f1("Paris France", "Paris Germany")
    assert 0.0 < partial < 1.0

def test_rouge_metrics():
    r1 = compute_rouge_n("Machine learning is powerful", "Machine learning is useful", n=1)
    assert r1["f1"] > 0.5
    rl = compute_rouge_l("A B C D", "A B C D")
    assert rl["f1"] == 1.0

def test_llm_judge_scores():
    judge = LLMJudge()
    res = judge.score_response(
        query="What is HTML?",
        response_text="HTML stands for HyperText Markup Language, the standard formatting language for web pages.",
        reference_answer="HyperText Markup Language",
        difficulty="Easy"
    )
    assert 1.0 <= res["overall_score"] <= 5.0
    assert res["correctness"] >= 4.0
    assert res["relevance"] >= 4.0

def test_latency_tracker():
    latencies = [0.2, 0.4, 0.6, 0.8, 1.0]
    summary = LatencyTracker.compute_summary(latencies)
    assert summary["count"] == 5
    assert summary["p50"] == 0.6
    assert summary["mean"] == 0.6

def test_cost_calculator():
    calc = CostCalculator()
    costs = calc.compute_query_cost(
        model_tier="small",
        input_tokens=100,
        output_tokens=50,
        latency_seconds=0.25
    )
    assert costs["api_token_cost_usd"] > 0
    assert costs["energy_kwh"] > 0

    savings = CostCalculator.compute_comparative_savings(
        latency_large=2.0,
        latency_adaptive=1.0,
        cost_large=0.01,
        cost_adaptive=0.005,
        quality_large=4.5,
        quality_adaptive=4.4
    )
    assert savings["latency_savings_pct"] == 50.0
    assert savings["cost_savings_pct"] == 50.0
    assert savings["quality_retention_pct"] > 95.0
