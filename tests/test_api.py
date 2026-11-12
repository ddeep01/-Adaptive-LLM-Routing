"""
Automated Unit Tests for FastAPI Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "cpu_utilization_pct" in data

def test_models_endpoint(client):
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert "small_model" in data
    assert "large_model" in data

def test_route_endpoint(client):
    r = client.post("/route", json={"query": "What is the capital of Italy?", "strategy": "adaptive"})
    assert r.status_code == 200
    data = r.json()
    assert data["selected_tier"] == "small"
    assert data["predicted_difficulty"] == "Easy"

def test_query_endpoint(client):
    r = client.post("/query", json={"query": "What is RAM in computing?", "strategy": "always_small"})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert len(data["response"]) > 0
    assert data["selected_tier"] == "small"

def test_evaluate_endpoint(client):
    r = client.post("/evaluate", json={
        "prediction": "Paris is the capital of France.",
        "reference": "Paris",
        "query": "What is the capital of France?",
        "difficulty": "Easy"
    })
    assert r.status_code == 200
    data = r.json()
    assert data["exact_match"] in [0.0, 1.0]
    assert 1.0 <= data["overall_judge_score"] <= 5.0
