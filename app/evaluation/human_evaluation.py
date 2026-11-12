"""
Human Evaluation Framework
Generates structured templates for blind human evaluation and aggregates human scores.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional

def generate_human_eval_template(
    records: List[Dict[str, Any]],
    output_path: str = "results/human_evaluation_template.csv"
) -> str:
    """
    Creates a CSV template for human evaluators assessing Small vs Large vs Adaptive responses.
    Candidate responses can be anonymized to prevent brand or model bias.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for r in records:
        rows.append({
            "query_id": r.get("query_id", ""),
            "query": r.get("query", ""),
            "category": r.get("category", ""),
            "model_a_response": r.get("small_response", ""),
            "model_b_response": r.get("large_response", ""),
            "adaptive_response": r.get("adaptive_response", ""),
            "correctness_score_1_to_5": "",
            "relevance_score_1_to_5": "",
            "completeness_score_1_to_5": "",
            "clarity_score_1_to_5": "",
            "overall_preference": "",  # "Model A", "Model B", "Tie"
            "evaluator_notes": ""
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path

def load_human_ratings(filepath: str) -> pd.DataFrame:
    """Loads populated human evaluation spreadsheet."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def summarize_human_ratings(filepath: str) -> Dict[str, Any]:
    """Computes mean scores and win rates from human ratings."""
    df = load_human_ratings(filepath)
    # Filter for non-empty ratings
    valid = df[df["overall_preference"].notna() & (df["overall_preference"] != "")]
    if valid.empty:
        return {"status": "pending", "completed_count": 0, "total_records": len(df)}

    win_counts = valid["overall_preference"].value_counts().to_dict()
    return {
        "status": "completed",
        "completed_count": len(valid),
        "total_records": len(df),
        "preferences": win_counts,
        "mean_correctness": round(float(pd.to_numeric(valid["correctness_score_1_to_5"], errors="coerce").mean()), 2)
    }
