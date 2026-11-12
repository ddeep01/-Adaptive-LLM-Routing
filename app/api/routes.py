"""
FastAPI REST Routes for Adaptive LLM Routing
"""

import os
import psutil
from fastapi import APIRouter, HTTPException, Query as QueryParam
from typing import Optional, Dict, Any, List

from app.api.schemas import (
    HealthResponse, ModelsResponse, QueryRequest, QueryResponse,
    RouteRequest, RouteResponse, EvaluateRequest, EvaluateResponse,
    RunExperimentRequest, RunExperimentResponse, ExperimentsListResponse
)
from app.routing.router import LLMRouter
from app.models.model_manager import ModelManager
from app.classifier.predict_difficulty import DifficultyPredictor
from app.evaluation.quality_metrics import evaluate_quality
from app.evaluation.llm_judge import LLMJudge
from app.database.repository import ExperimentRepository
from app.experiments.experiment_runner import ExperimentRunner
from app.experiments.analysis import ResultsAnalyzer
from app.monitoring.resource_monitor import ResourceMonitor

router = APIRouter()

# Instantiate shared dependencies
model_mgr = ModelManager()
predictor = DifficultyPredictor()
llm_router = LLMRouter(model_manager=model_mgr, difficulty_predictor=predictor)
judge = LLMJudge()
repo = ExperimentRepository()
analyzer = ResultsAnalyzer()

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns system status, active version, CPU and RAM metrics."""
    snap = ResourceMonitor.get_current_snapshot()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment="production",
        ram_available_mb=snap["ram_available_mb"],
        cpu_utilization_pct=snap["cpu_utilization_pct"]
    )

@router.get("/models", response_model=ModelsResponse, tags=["Models"])
def get_models():
    """Returns configured Small and Large LLM details and availability."""
    st = model_mgr.get_status()
    return ModelsResponse(
        small_model=st["small_model"],
        large_model=st["large_model"]
    )

@router.post("/query", response_model=QueryResponse, tags=["Inference"])
def process_query(req: QueryRequest):
    """Routes query, executes appropriate LLM, evaluates cascade checks, and returns result."""
    try:
        res = llm_router.route_and_generate(
            query=req.query,
            strategy=req.strategy,
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
        return QueryResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/route", response_model=RouteResponse, tags=["Routing"])
def route_query(req: RouteRequest):
    """Evaluates query complexity and returns predicted tier without running model generation."""
    pred = predictor.predict(req.query)
    cat = pred["predicted_difficulty"]
    score = pred["difficulty_score"]
    conf = pred["confidence"]
    
    # Strategy evaluation
    from app.routing.strategies import RoutingStrategyRegistry
    strat = req.strategy.lower()
    if strat == "always_small":
        tier, reason = RoutingStrategyRegistry.always_small(req.query, pred)
    elif strat == "always_large":
        tier, reason = RoutingStrategyRegistry.always_large(req.query, pred)
    elif strat == "rule_based":
        tier, reason = RoutingStrategyRegistry.rule_based(req.query, pred)
    elif strat == "confidence_based":
        tier, reason = RoutingStrategyRegistry.confidence_based(req.query, pred)
    else:
        tier, reason = RoutingStrategyRegistry.ml_adaptive(req.query, pred)

    return RouteResponse(
        query=req.query,
        predicted_difficulty=cat,
        difficulty_score=score,
        confidence=conf,
        classifier=pred["predicted_by"],
        selected_tier=tier,
        routing_reason=reason
    )

@router.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
def evaluate_response(req: EvaluateRequest):
    """Calculates Exact Match, Token F1, ROUGE-1/2/L, and LLM-as-a-Judge scores."""
    metrics = evaluate_quality(prediction=req.prediction, reference=req.reference)
    judge_res = judge.score_response(
        query=req.query,
        response_text=req.prediction,
        reference_answer=req.reference,
        difficulty=req.difficulty
    )
    return EvaluateResponse(
        exact_match=metrics["exact_match"],
        token_f1=metrics["token_f1"],
        rouge1_f1=metrics["rouge1_f1"],
        rouge2_f1=metrics["rouge2_f1"],
        rougeL_f1=metrics["rougeL_f1"],
        overall_judge_score=judge_res["overall_score"],
        correctness=judge_res["correctness"],
        relevance=judge_res["relevance"],
        completeness=judge_res["completeness"],
        clarity=judge_res["clarity"],
        reasoning_quality=judge_res["reasoning_quality"],
        instruction_following=judge_res["instruction_following"]
    )

@router.post("/experiments/run", response_model=RunExperimentResponse, tags=["Experiments"])
def trigger_experiments(req: RunExperimentRequest):
    """Triggers execution of the complete 7-experiment benchmarking suite."""
    try:
        runner = ExperimentRunner()
        summary = runner.run_all(n_per_category=req.samples_per_category)
        return RunExperimentResponse(
            status="completed",
            experiments_run=summary["experiments_run"],
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments", response_model=ExperimentsListResponse, tags=["Experiments"])
def list_experiments():
    """Lists all completed benchmark experiment runs."""
    exps = repo.get_experiments()
    return ExperimentsListResponse(experiments=exps)

@router.get("/experiments/{experiment_id}", tags=["Experiments"])
def get_experiment_details(experiment_id: str):
    """Returns detailed records for a specific experiment ID."""
    df = repo.get_all_results_df(experiment_id)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return {
        "experiment_id": experiment_id,
        "record_count": len(df),
        "records": df.to_dict(orient="records")
    }

@router.get("/results", tags=["Analytics"])
def get_all_results(limit: int = 100):
    """Retrieves aggregated query response and evaluation results across experiments."""
    df = repo.get_all_results_df()
    return {
        "total_results": len(df),
        "limit": limit,
        "results": df.head(limit).to_dict(orient="records")
    }

@router.get("/metrics", tags=["Analytics"])
def get_comparative_metrics():
    """Returns comparative summary: Latency Savings %, Cost Savings %, and Quality Retention %."""
    return analyzer.compute_pareto_savings()

@router.get("/export", tags=["Analytics"])
def export_results(experiment_id: Optional[str] = None):
    """Exports results to CSV files and returns file paths."""
    exp_id = experiment_id or "exp_002_strategy_comparison"
    paths = repo.export_experiment_to_csv(exp_id)
    return {
        "experiment_id": exp_id,
        "exported_files": paths
    }
