"""
Automated Experiment Suite Runner
Executes all 7 planned research experiments with reproducible seeds and telemetry capture.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from app.core.logger import logger
from app.experiments.benchmark import BenchmarkHarness
from app.database.repository import ExperimentRepository

class ExperimentRunner:
    """Coordinates and executes the standardized 7 research experiments."""

    def __init__(self, data_path: str = "data/queries.csv", output_dir: str = "results"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.repo = ExperimentRepository()
        self.harness = BenchmarkHarness(repository=self.repo)
        self.queries = self._load_queries()

    def _load_queries(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Queries dataset missing at {self.data_path}")
        df = pd.read_csv(self.data_path)
        queries = df.to_dict(orient="records")
        # Ensure queries are in DB
        self.repo.insert_queries(queries)
        return queries

    def get_stratified_sample(self, n_per_category: int = 20) -> List[Dict[str, Any]]:
        """Extracts a balanced stratified sample of queries for rapid benchmarking."""
        df = pd.DataFrame(self.queries)
        sampled = df.groupby("category", group_keys=False).apply(
            lambda x: x.sample(min(len(x), n_per_category), random_state=42)
        )
        return sampled.to_dict(orient="records")

    def run_experiment_1_small_vs_large(self, test_queries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Experiment 1: Small vs. Large Model Performance Baseline
        Direct head-to-head comparison on identical queries across Quality, Latency, and Cost.
        """
        exp_id = "exp_001_small_vs_large"
        queries = test_queries or self.get_stratified_sample(n_per_category=25)
        
        self.repo.insert_experiment(
            experiment_id=exp_id,
            name="Small vs Large Model Baseline",
            description="Direct head-to-head comparison of small vs large LLM on identical queries.",
            config={"n_queries": len(queries)}
        )

        res_small = self.harness.run_strategy_benchmark(
            experiment_id=exp_id, queries=queries, strategy="always_small"
        )
        res_large = self.harness.run_strategy_benchmark(
            experiment_id=exp_id, queries=queries, strategy="always_large"
        )

        # Export CSVs
        self.repo.export_experiment_to_csv(exp_id, output_dir=self.output_dir)
        logger.info(f"Completed Experiment 1 with {len(queries)} queries per model.")
        return {"experiment_id": exp_id, "small_count": len(res_small), "large_count": len(res_large)}

    def run_experiment_2_strategy_comparison(self, test_queries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Experiment 2: Routing Strategy Comparison
        Compares all 6 strategies: Always Small, Always Large, Rule-Based, ML Adaptive, Confidence-Based, Cascade.
        """
        exp_id = "exp_002_strategy_comparison"
        queries = test_queries or self.get_stratified_sample(n_per_category=20)

        self.repo.insert_experiment(
            experiment_id=exp_id,
            name="Comprehensive Strategy Comparison",
            description="Benchmark of all 6 routing strategies across balanced queries.",
            config={"strategies": ["always_small", "always_large", "rule_based", "adaptive", "confidence_based", "cascade"]}
        )

        strategies = ["always_small", "always_large", "rule_based", "adaptive", "confidence_based", "cascade"]
        counts = {}
        for strat in strategies:
            res = self.harness.run_strategy_benchmark(
                experiment_id=exp_id, queries=queries, strategy=strat
            )
            counts[strat] = len(res)

        self.repo.export_experiment_to_csv(exp_id, output_dir=self.output_dir)
        logger.info("Completed Experiment 2 for all 6 strategies.")
        return {"experiment_id": exp_id, "counts": counts}

    def run_experiment_3_difficulty_stratification(self, test_queries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Experiment 3: Difficulty-Wise Performance Analysis
        Evaluates performance separately for Easy, Medium, and Hard queries.
        """
        exp_id = "exp_003_difficulty_stratification"
        queries = test_queries or self.get_stratified_sample(n_per_category=25)

        self.repo.insert_experiment(
            experiment_id=exp_id,
            name="Difficulty-Stratified Performance Analysis",
            description="Granular performance and quality breakdown across Easy, Medium, and Hard queries.",
            config={"categories": ["Easy", "Medium", "Hard"]}
        )

        # Run on Adaptive and Cascade strategies
        self.harness.run_strategy_benchmark(exp_id, queries, "adaptive")
        self.harness.run_strategy_benchmark(exp_id, queries, "cascade")

        self.repo.export_experiment_to_csv(exp_id, output_dir=self.output_dir)
        logger.info("Completed Experiment 3.")
        return {"experiment_id": exp_id}

    def run_experiment_4_quality_vs_cost(self, test_queries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Experiment 4: Quality vs. Cost Pareto Trade-Off
        Measures quality retention percentage alongside latency and financial savings.
        """
        exp_id = "exp_004_quality_cost_pareto"
        queries = test_queries or self.get_stratified_sample(n_per_category=20)

        self.repo.insert_experiment(
            experiment_id=exp_id,
            name="Quality vs Cost Pareto Trade-Off",
            description="Quantifies quality retention versus cost and latency reduction.",
            config={"queries_count": len(queries)}
        )

        self.harness.run_strategy_benchmark(exp_id, queries, "always_small")
        self.harness.run_strategy_benchmark(exp_id, queries, "always_large")
        self.harness.run_strategy_benchmark(exp_id, queries, "adaptive")

        self.repo.export_experiment_to_csv(exp_id, output_dir=self.output_dir)
        logger.info("Completed Experiment 4.")
        return {"experiment_id": exp_id}

    def run_experiment_5_threshold_sweep(self, test_queries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Experiment 5: Routing Threshold Analysis
        Sweeps confidence thresholds tau in [0.20, 0.40, 0.60, 0.75, 0.85].
        """
        exp_id = "exp_005_threshold_sweep"
        queries = test_queries or self.get_stratified_sample(n_per_category=15)
        thresholds = [0.20, 0.40, 0.60, 0.75, 0.85]

        self.repo.insert_experiment(
            experiment_id=exp_id,
            name="Routing Threshold Sensitivity Sweep",
            description="Investigates sensitivity of cost, latency, and large-model usage to decision thresholds.",
            config={"thresholds": thresholds}
        )

        for tau in thresholds:
            # Dynamically adjust router threshold
            self.harness.router.conf_threshold = tau
            self.harness.run_strategy_benchmark(
                experiment_id=exp_id,
                queries=queries,
                strategy="confidence_based",
                description=f"tau={tau}"
            )

        self.repo.export_experiment_to_csv(exp_id, output_dir=self.output_dir)
        logger.info("Completed Experiment 5 threshold sweep.")
        return {"experiment_id": exp_id, "thresholds": thresholds}

    def run_experiment_6_escalation_analysis(self, test_queries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Experiment 6: Escalation Analysis
        Analyzes cascade failure rates, additional latency overhead, and quality recovery.
        """
        exp_id = "exp_006_escalation_analysis"
        queries = test_queries or self.get_stratified_sample(n_per_category=20)

        self.repo.insert_experiment(
            experiment_id=exp_id,
            name="Cascade Escalation Dynamics",
            description="Measures escalation rate, small model failure modes, and post-escalation quality gain.",
            config={"queries_count": len(queries)}
        )

        self.harness.run_strategy_benchmark(exp_id, queries, "cascade")
        self.repo.export_experiment_to_csv(exp_id, output_dir=self.output_dir)
        logger.info("Completed Experiment 6.")
        return {"experiment_id": exp_id}

    def run_experiment_7_ablation_study(self, test_queries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Experiment 7: Component Ablation Study
        Compares complete adaptive system against ablated versions without ML, confidence, or escalation.
        """
        exp_id = "exp_007_ablation_study"
        queries = test_queries or self.get_stratified_sample(n_per_category=20)

        self.repo.insert_experiment(
            experiment_id=exp_id,
            name="Architectural Component Ablation Study",
            description="Evaluates contribution of ML classifier, confidence weighting, and escalation checks.",
            config={"variants": ["full_adaptive", "no_ml_rule_based", "no_escalation_small_only"]}
        )

        # Full system
        self.harness.run_strategy_benchmark(exp_id, queries, "adaptive")
        # Ablation 1: No ML (Rule-based only)
        self.harness.run_strategy_benchmark(exp_id, queries, "rule_based")
        # Ablation 2: No escalation (always small)
        self.harness.run_strategy_benchmark(exp_id, queries, "always_small")

        self.repo.export_experiment_to_csv(exp_id, output_dir=self.output_dir)
        logger.info("Completed Experiment 7.")
        return {"experiment_id": exp_id}

    def run_all(self, n_per_category: int = 15) -> Dict[str, Any]:
        """Runs the complete suite of all 7 experiments systematically."""
        logger.info("================ STARTING COMPLETE EXPERIMENTAL BENCHMARK ================")
        sample = self.get_stratified_sample(n_per_category=n_per_category)

        r1 = self.run_experiment_1_small_vs_large(sample)
        r2 = self.run_experiment_2_strategy_comparison(sample)
        r3 = self.run_experiment_3_difficulty_stratification(sample)
        r4 = self.run_experiment_4_quality_vs_cost(sample)
        r5 = self.run_experiment_5_threshold_sweep(sample)
        r6 = self.run_experiment_6_escalation_analysis(sample)
        r7 = self.run_experiment_7_ablation_study(sample)

        summary = {
            "status": "completed",
            "experiments_run": 7,
            "sample_size_per_stratum": n_per_category,
            "total_benchmark_queries": len(sample),
            "results": {
                "exp_1": r1, "exp_2": r2, "exp_3": r3, "exp_4": r4,
                "exp_5": r5, "exp_6": r6, "exp_7": r7
            }
        }
        with open(os.path.join(self.output_dir, "benchmark_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("================ COMPLETE BENCHMARK SUITE EXECUTED SUCCESSFULLY ================")
        return summary
