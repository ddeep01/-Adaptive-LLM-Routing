"""
Automated Unit Tests for SQLite Database and Repository
"""

import os
import pytest
from app.database.repository import ExperimentRepository

TEST_DB_PATH = "results/test_experiments.db"

@pytest.fixture
def repo():
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
    r = ExperimentRepository(db_path=TEST_DB_PATH)
    yield r
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass

def test_experiment_crud(repo):
    repo.insert_experiment(
        experiment_id="test_exp_01",
        name="Unit Test Experiment",
        description="Testing repository insert and read",
        config={"param": "value"}
    )
    exps = repo.get_experiments()
    assert len(exps) == 1
    assert exps[0]["experiment_id"] == "test_exp_01"

def test_queries_insertion(repo):
    queries = [
        {"query_id": "T01", "query": "Test query 1", "category": "Easy", "difficulty_score": 0.1},
        {"query_id": "T02", "query": "Test query 2", "category": "Hard", "difficulty_score": 0.8}
    ]
    repo.insert_queries(queries)
    cursor = repo.conn.execute("SELECT COUNT(*) as cnt FROM queries")
    count = cursor.fetchone()["cnt"]
    assert count == 2

def test_routing_and_response_insertion(repo):
    repo.insert_experiment("exp_demo", "Demo", "Demo", {})
    repo.insert_queries([{"query_id": "Q_DEMO", "query": "Demo Q", "category": "Easy"}])

    repo.insert_routing_decision({
        "experiment_id": "exp_demo",
        "query_id": "Q_DEMO",
        "strategy": "adaptive",
        "predicted_difficulty": "Easy",
        "difficulty_score": 0.15,
        "confidence": 0.95,
        "selected_tier": "small",
        "selected_model": "qwen2.5:3b",
        "routing_reason": "Low difficulty",
        "escalated": False
    })

    repo.insert_response({
        "experiment_id": "exp_demo",
        "query_id": "Q_DEMO",
        "strategy": "adaptive",
        "model_tier": "small",
        "model_name": "qwen2.5:3b",
        "response_text": "Demo response",
        "total_latency": 0.22,
        "input_tokens": 10,
        "output_tokens": 20,
        "tokens_per_second": 45.0,
        "cpu_pct": 15.0,
        "ram_mb": 2500.0,
        "api_cost_usd": 0.00005,
        "energy_cost_usd": 0.00001
    })

    df = repo.get_all_results_df("exp_demo")
    assert not df.empty
    assert df.iloc[0]["query_id"] == "Q_DEMO"
    assert df.iloc[0]["selected_tier"] == "small"
