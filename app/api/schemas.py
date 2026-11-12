"""
Pydantic Schemas for FastAPI Request and Response Models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    ram_available_mb: float
    cpu_utilization_pct: float

class ModelInfo(BaseModel):
    name: str
    tier: str
    provider: str
    available: bool
    warmed_up: bool

class ModelsResponse(BaseModel):
    small_model: ModelInfo
    large_model: ModelInfo

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language prompt")
    strategy: Optional[str] = Field("adaptive", description="Routing strategy: adaptive, always_small, always_large, rule_based, confidence_based, cascade")
    temperature: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(512, ge=1, le=2048)

class QueryResponse(BaseModel):
    query: str
    difficulty: str
    difficulty_score: float
    confidence: float
    classifier_used: str
    routing_strategy: str
    routing_reason: str
    selected_model: str
    selected_tier: str
    response: str
    escalated: bool
    escalation_reason: Optional[str] = None
    total_latency: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    success: bool
    error_message: Optional[str] = None

class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1)
    strategy: Optional[str] = "adaptive"

class RouteResponse(BaseModel):
    query: str
    predicted_difficulty: str
    difficulty_score: float
    confidence: float
    classifier: str
    selected_tier: str
    routing_reason: str

class EvaluateRequest(BaseModel):
    prediction: str = Field(..., description="Generated model response")
    reference: str = Field(..., description="Ground truth reference answer")
    query: Optional[str] = ""
    difficulty: Optional[str] = "Medium"

class EvaluateResponse(BaseModel):
    exact_match: float
    token_f1: float
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    overall_judge_score: float
    correctness: float
    relevance: float
    completeness: float
    clarity: float
    reasoning_quality: float
    instruction_following: float

class RunExperimentRequest(BaseModel):
    samples_per_category: Optional[int] = Field(15, ge=5, le=100)

class RunExperimentResponse(BaseModel):
    status: str
    experiments_run: int
    summary: Dict[str, Any]

class ExperimentItem(BaseModel):
    experiment_id: str
    name: str
    description: Optional[str] = None
    config_json: Optional[str] = None
    status: str
    created_at: str

class ExperimentsListResponse(BaseModel):
    experiments: List[ExperimentItem]
