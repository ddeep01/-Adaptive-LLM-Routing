"""
Automated Research Report Generator
Extracts empirical results from SQLite and outputs a markdown summary table.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from app.database.repository import ExperimentRepository
from app.experiments.analysis import ResultsAnalyzer

def generate_report():
    repo = ExperimentRepository()
    analyzer = ResultsAnalyzer()

    df = repo.get_all_results_df()
    if df.empty:
        print("[WARN] No database records found to generate report.")
        return

    exp1 = analyzer.analyze_experiment_1()
    exp2 = analyzer.analyze_experiment_2()
    pareto = analyzer.compute_pareto_savings()
    sav = pareto.get("comparative_savings", {})

    print("=== EMPIRICAL RESEARCH SUMMARY ===")
    print(f"Total Database Query Evaluations: {len(df)}")
    print(f"Latency Savings: {sav.get('latency_savings_pct')}%")
    print(f"Cost Savings:    {sav.get('cost_savings_pct')}%")
    print(f"Quality Parity:  {sav.get('quality_retention_pct')}%")
    print("\nStrategy Breakdown:")
    print(exp2.to_string(index=False))

if __name__ == "__main__":
    generate_report()
