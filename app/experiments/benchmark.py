"""
Core Benchmarking Harness for Adaptive LLM Routing
Executes test queries across specified strategies, collects telemetry, computes quality scores,
and records all results into the SQLite repository and CSV exports.
"""

import time
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional

from app.core.logger import logger
from app.routing.router import LLMRouter
from app.evaluation.quality_metrics import evaluate_quality
from app.evaluation.llm_judge import LLMJudge
from app.monitoring.cost_calculator import CostCalculator
from app.database.repository import ExperimentRepository

class BenchmarkHarness:
    """Executes controlled benchmark experiments on query datasets."""

    def __init__(
        self,
        router: Optional[LLMRouter] = None,
        repository: Optional[ExperimentRepository] = None,
        cost_calculator: Optional[CostCalculator] = None,
        judge: Optional[LLMJudge] = None
    ):
        self.router = router or LLMRouter()
        self.repo = repository or ExperimentRepository()
        self.cost_calc = cost_calculator or CostCalculator()
        self.judge = judge or LLMJudge()

    def run_strategy_benchmark(
        self,
        experiment_id: str,
        queries: List[Dict[str, Any]],
        strategy: str,
        description: str = "",
        save_db: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Runs a list of queries through a single routing strategy.
        Records routing decisions, response telemetry, costs, and quality scores.
        """
        logger.info(f"Starting benchmark '{experiment_id}' for strategy '{strategy}' on {len(queries)} queries.")
        
        results = []
        for i, q_item in enumerate(queries):
            qid = q_item.get("query_id", f"Q{i+1:03d}")
            prompt = q_item.get("query", "")
            cat = q_item.get("category", "Medium")
            ref_ans = q_item.get("reference_answer", "")

            # Execute router
            route_out = self.router.route_and_generate(query=prompt, strategy=strategy)

            # Compute cost
            costs = self.cost_calc.compute_query_cost(
                model_tier=route_out["selected_tier"],
                input_tokens=route_out["input_tokens"],
                output_tokens=route_out["output_tokens"],
                latency_seconds=route_out["total_latency"]
            )

            # Compute quality metrics
            qual = evaluate_quality(prediction=route_out["response"], reference=ref_ans)
            
            # Compute judge score
            judge_res = self.judge.score_response(
                query=prompt,
                response_text=route_out["response"],
                reference_answer=ref_ans,
                difficulty=cat
            )

            row = {
                "experiment_id": experiment_id,
                "query_id": qid,
                "query": prompt,
                "category": cat,
                "strategy": strategy,
                "predicted_difficulty": route_out["difficulty"],
                "difficulty_score": route_out["difficulty_score"],
                "confidence": route_out["confidence"],
                "selected_tier": route_out["selected_tier"],
                "selected_model": route_out["selected_model"],
                "routing_reason": route_out["routing_reason"],
                "escalated": route_out["escalated"],
                "escalation_reason": route_out["escalation_reason"],
                "response_text": route_out["response"],
                "total_latency": route_out["total_latency"],
                "input_tokens": route_out["input_tokens"],
                "output_tokens": route_out["output_tokens"],
                "tokens_per_second": route_out["tokens_per_second"],
                "cpu_pct": route_out["cpu_utilization_pct"],
                "ram_mb": route_out["ram_used_mb"],
                "api_cost_usd": costs["api_token_cost_usd"],
                "energy_cost_usd": costs["energy_cost_usd"],
                "total_cost_usd": costs["total_estimated_cost_usd"],
                "exact_match": qual["exact_match"],
                "token_f1": qual["token_f1"],
                "rouge1_f1": qual["rouge1_f1"],
                "rouge2_f1": qual["rouge2_f1"],
                "rougeL_f1": qual["rougeL_f1"],
                "overall_judge_score": judge_res["overall_score"],
                "correctness": judge_res["correctness"],
                "relevance": judge_res["relevance"],
                "completeness": judge_res["completeness"],
                "clarity": judge_res["clarity"],
                "reasoning_quality": judge_res["reasoning_quality"],
                "instruction_following": judge_res["instruction_following"]
            }

            if save_db:
                self.repo.insert_routing_decision(row)
                self.repo.insert_response(row)
                self.repo.insert_evaluation(row)

            results.append(row)

        return results
