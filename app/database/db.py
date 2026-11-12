"""
SQLite Database Connection and Schema Management
"""

import os
import sqlite3
from typing import Optional

DEFAULT_DB_PATH = "results/experiments.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    config_json TEXT,
    status TEXT DEFAULT 'completed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    category TEXT NOT NULL,
    difficulty_score REAL,
    domain TEXT,
    expected_capability TEXT,
    reference_answer TEXT
);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    query_id TEXT,
    strategy TEXT NOT NULL,
    predicted_difficulty TEXT,
    difficulty_score REAL,
    confidence REAL,
    selected_tier TEXT,
    selected_model TEXT,
    routing_reason TEXT,
    escalated INTEGER DEFAULT 0,
    escalation_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
    FOREIGN KEY(query_id) REFERENCES queries(query_id)
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    query_id TEXT,
    strategy TEXT,
    model_tier TEXT,
    model_name TEXT,
    response_text TEXT,
    total_latency REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    tokens_per_second REAL,
    cpu_pct REAL,
    ram_mb REAL,
    api_cost_usd REAL,
    energy_cost_usd REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
    FOREIGN KEY(query_id) REFERENCES queries(query_id)
);

CREATE TABLE IF NOT EXISTS evaluation_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    query_id TEXT,
    strategy TEXT,
    model_tier TEXT,
    exact_match REAL,
    token_f1 REAL,
    rouge1_f1 REAL,
    rouge2_f1 REAL,
    rougeL_f1 REAL,
    overall_judge_score REAL,
    correctness REAL,
    relevance REAL,
    completeness REAL,
    clarity REAL,
    reasoning_quality REAL,
    instruction_following REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
    FOREIGN KEY(query_id) REFERENCES queries(query_id)
);
"""

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with row factory configured."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initializes tables in the SQLite database."""
    conn = get_connection(db_path)
    with conn:
        conn.executescript(SCHEMA_SQL)
    return conn
