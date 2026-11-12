"""
Command-line Benchmark Runner
Executes the full suite of 7 research experiments and persists all results.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from app.experiments.experiment_runner import ExperimentRunner
from app.experiments.analysis import ResultsAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Adaptive LLM Routing Benchmark Runner")
    parser.add_argument("--samples-per-class", type=int, default=15, help="Number of queries per difficulty category (default: 15, total 45)")
    parser.add_argument("--data-path", type=str, default="data/queries.csv", help="Path to queries CSV")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output CSVs and summaries")
    args = parser.parse_args()

    print(f"=== INITIALIZING EXPERIMENT RUNNER ({args.samples_per_class} samples/class) ===")
    runner = ExperimentRunner(data_path=args.data_path, output_dir=args.output_dir)
    summary = runner.run_all(n_per_category=args.samples_per_class)

    print("\n=== BENCHMARK SUITE COMPLETED ===")
    print(f"Total experiments executed: {summary['experiments_run']}")
    print(f"Summary saved to {args.output_dir}/benchmark_summary.json")

    # Run analytical pass
    analyzer = ResultsAnalyzer()
    pareto = analyzer.compute_pareto_savings()
    if pareto:
        sav = pareto.get("comparative_savings", {})
        print("\n=== MEASURED COMPARATIVE FINDINGS ===")
        print(f"Latency Savings:   {sav.get('latency_savings_pct')}%")
        print(f"Cost Savings:      {sav.get('cost_savings_pct')}%")
        print(f"Quality Retention: {sav.get('quality_retention_pct')}%")

if __name__ == "__main__":
    main()
