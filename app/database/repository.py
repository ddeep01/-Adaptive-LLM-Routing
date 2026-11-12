"""
Database Repository Module
Encapsulates CRUD operations, experiment persistence, and CSV export functionality.
"""

import os
import json
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional

from app.database.db import get_connection, init_db, DEFAULT_DB_PATH

class ExperimentRepository:
    """Provides methods to insert, query, and export experiment data."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn = init_db(db_path)

    def insert_experiment(self, experiment_id: str, name: str, description: str, config: Dict[str, Any]):
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO experiments (experiment_id, name, description, config_json, status)
                VALUES (?, ?, ?, ?, 'completed')
                """,
                (experiment_id, name, description, json.dumps(config))
            )

    def insert_queries(self, queries: List[Dict[str, Any]]):
        with self.conn:
            for q in queries:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO queries 
                    (query_id, query, category, difficulty_score, domain, expected_capability, reference_answer)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        q.get("query_id"), q.get("query"), q.get("category"),
                        q.get("difficulty_score"), q.get("domain"),
                        q.get("expected_capability"), q.get("reference_answer")
                    )
                )

    def insert_routing_decision(self, record: Dict[str, Any]):
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO routing_decisions
                (experiment_id, query_id, strategy, predicted_difficulty, difficulty_score,
                 confidence, selected_tier, selected_model, routing_reason, escalated, escalation_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("experiment_id"), record.get("query_id"), record.get("strategy"),
                    record.get("predicted_difficulty"), record.get("difficulty_score"),
                    record.get("confidence"), record.get("selected_tier"),
                    record.get("selected_model"), record.get("routing_reason"),
                    1 if record.get("escalated") else 0, record.get("escalation_reason")
                )
            )

    def insert_response(self, record: Dict[str, Any]):
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO responses
                (experiment_id, query_id, strategy, model_tier, model_name, response_text,
                 total_latency, input_tokens, output_tokens, tokens_per_second, cpu_pct, ram_mb,
                 api_cost_usd, energy_cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("experiment_id"), record.get("query_id"), record.get("strategy"),
                    record.get("model_tier"), record.get("model_name"), record.get("response_text"),
                    record.get("total_latency"), record.get("input_tokens"), record.get("output_tokens"),
                    record.get("tokens_per_second"), record.get("cpu_pct"), record.get("ram_mb"),
                    record.get("api_cost_usd"), record.get("energy_cost_usd")
                )
            )

    def insert_evaluation(self, record: Dict[str, Any]):
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO evaluation_scores
                (experiment_id, query_id, strategy, model_tier, exact_match, token_f1,
                 rouge1_f1, rouge2_f1, rougeL_f1, overall_judge_score, correctness,
                 relevance, completeness, clarity, reasoning_quality, instruction_following)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("experiment_id"), record.get("query_id"), record.get("strategy"),
                    record.get("model_tier"), record.get("exact_match"), record.get("token_f1"),
                    record.get("rouge1_f1"), record.get("rouge2_f1"), record.get("rougeL_f1"),
                    record.get("overall_judge_score"), record.get("correctness"),
                    record.get("relevance"), record.get("completeness"), record.get("clarity"),
                    record.get("reasoning_quality"), record.get("instruction_following")
                )
            )

    def get_experiments(self) -> List[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

    def get_responses_by_experiment(self, experiment_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            """
            SELECT r.*, q.query, q.category, q.difficulty_score as true_difficulty
            FROM responses r
            LEFT JOIN queries q ON r.query_id = q.query_id
            WHERE r.experiment_id = ?
            """,
            (experiment_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_all_results_df(self, experiment_id: Optional[str] = None) -> pd.DataFrame:
        query = """
        SELECT 
            r.experiment_id,
            r.query_id,
            q.query,
            q.category as true_category,
            q.domain,
            r.strategy,
            r.model_tier,
            rd.selected_tier,
            r.model_name,
            rd.predicted_difficulty,
            rd.difficulty_score as predicted_difficulty_score,
            rd.confidence as classifier_confidence,
            rd.escalated,
            r.total_latency,
            r.tokens_per_second,
            r.api_cost_usd,
            r.energy_cost_usd,
            ev.exact_match,
            ev.token_f1,
            ev.rougeL_f1,
            ev.overall_judge_score,
            ev.correctness,
            ev.completeness
        FROM responses r
        LEFT JOIN queries q ON r.query_id = q.query_id
        LEFT JOIN routing_decisions rd ON (r.experiment_id = rd.experiment_id AND r.query_id = rd.query_id AND r.strategy = rd.strategy)
        LEFT JOIN evaluation_scores ev ON (r.experiment_id = ev.experiment_id AND r.query_id = ev.query_id AND r.strategy = ev.strategy)
        """
        params = ()
        if experiment_id:
            query += " WHERE r.experiment_id = ?"
            params = (experiment_id,)
        
        return pd.read_sql_query(query, self.conn, params=params)

    def export_experiment_to_csv(self, experiment_id: str, output_dir: str = "results") -> Dict[str, str]:
        """Exports experiment records to CSV files."""
        exp_dir = os.path.join(output_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)

        tables = {
            "responses": f"SELECT * FROM responses WHERE experiment_id = '{experiment_id}'",
            "routing_decisions": f"SELECT * FROM routing_decisions WHERE experiment_id = '{experiment_id}'",
            "evaluation_scores": f"SELECT * FROM evaluation_scores WHERE experiment_id = '{experiment_id}'"
        }

        paths = {}
        for name, sql in tables.items():
            df = pd.read_sql_query(sql, self.conn)
            csv_path = os.path.join(exp_dir, f"{name}.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            paths[name] = csv_path

        return paths
