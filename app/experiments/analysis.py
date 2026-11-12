"""
Experimental Results Analysis and Metrics Aggregator
Extracts experimental records from SQLite and produces publication-ready statistical tables.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from app.database.repository import ExperimentRepository
from app.monitoring.cost_calculator import CostCalculator

class ResultsAnalyzer:
    """Computes statistical summaries, Pareto comparisons, and ablation insights."""

    def __init__(self, db_path: str = "results/experiments.db"):
        self.repo = ExperimentRepository(db_path)

    def analyze_experiment_1(self) -> Dict[str, Any]:
        """Summarizes small vs large model baseline."""
        df = self.repo.get_all_results_df("exp_001_small_vs_large")
        if df.empty:
            return {}

        grp = df.groupby("model_tier").agg(
            mean_latency=("total_latency", "mean"),
            p95_latency=("total_latency", lambda x: np.percentile(x, 95)),
            mean_cost=("api_cost_usd", "mean"),
            mean_judge_score=("overall_judge_score", "mean"),
            mean_token_f1=("token_f1", "mean"),
            mean_tps=("tokens_per_second", "mean")
        ).round(4).to_dict(orient="index")

        return grp

    def analyze_experiment_2(self) -> pd.DataFrame:
        """Summarizes 6 strategies comparison."""
        df = self.repo.get_all_results_df("exp_002_strategy_comparison")
        if df.empty:
            return pd.DataFrame()

        summary = df.groupby("strategy").agg(
            avg_latency=("total_latency", "mean"),
            p95_latency=("total_latency", lambda x: np.percentile(x, 95)),
            avg_cost=("api_cost_usd", "mean"),
            avg_quality=("overall_judge_score", "mean"),
            avg_f1=("token_f1", "mean"),
            large_model_pct=("model_tier", lambda x: (x == "large").mean() * 100.0)
        ).round(4).reset_index()

        return summary

    def compute_pareto_savings(self) -> Dict[str, Any]:
        """Calculates Latency Savings %, Cost Savings %, and Quality Retention %."""
        df = self.repo.get_all_results_df()
        if df.empty:
            return {}

        strat_grp = df.groupby("strategy").agg(
            latency=("total_latency", "mean"),
            cost=("api_cost_usd", "mean"),
            quality=("overall_judge_score", "mean")
        ).to_dict(orient="index")

        large_stats = strat_grp.get("always_large", {})
        adaptive_stats = strat_grp.get("adaptive", {})

        if not large_stats or not adaptive_stats:
            return {}

        savings = CostCalculator.compute_comparative_savings(
            latency_large=large_stats["latency"],
            latency_adaptive=adaptive_stats["latency"],
            cost_large=large_stats["cost"],
            cost_adaptive=adaptive_stats["cost"],
            quality_large=large_stats["quality"],
            quality_adaptive=adaptive_stats["quality"]
        )
        return {
            "large_baseline": large_stats,
            "adaptive_results": adaptive_stats,
            "comparative_savings": savings
        }
