# System Architecture: Adaptive LLM Routing

## 1. Architectural Overview
The Adaptive LLM Routing system is designed as a decoupled, multi-tier microservice architecture that dynamically routes user queries between small (lightweight) and large (frontier) language models based on upstream complexity estimation, classifier confidence, and cascading output validation.

```
                           ┌─────────────────────────┐
                           │      Client / UI        │
                           │  (Streamlit / REST API) │
                           └────────────┬────────────┘
                                        │ HTTP / JSON
                                        ▼
                           ┌─────────────────────────┐
                           │     FastAPI Gateway     │
                           │     (app/api/routes)    │
                           └────────────┬────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │    LLM Router Engine    │
                           │   (app/routing/router)  │
                           └────┬───────────────┬────┘
                                │               │
          Linguistic & ML Feats │               │ Routing Tier Decision
                                ▼               ▼
                 ┌────────────────────┐   ┌────────────────────────┐
                 │Difficulty Predictor│   │     Model Manager      │
                 │- Feature Extractor │   │- Sequential RAM Checks │
                 │- Rule-Based Engine │   │- Ollama / Offline Client│
                 │- Gradient Boosting │   └─────┬────────────┬─────┘
                 └────────────────────┘         │            │
                                          Small │            │ Large
                                          Model ▼            ▼ Model
                                         ┌──────────┐    ┌──────────┐
                                         │Qwen2.5:3b│    │Qwen2.5:7b│
                                         └────┬─────┘    └────┬─────┘
                                              │               │
                                              └───────┬───────┘
                                                      │
                                                      ▼
                                         ┌─────────────────────────┐
                                         │ Quality & Cost Monitor  │
                                         │- Exact Match & Token F1 │
                                         │- ROUGE-1/2/L & LLM Judge│
                                         │- Latency & Energy Cost  │
                                         └────────────┬────────────┘
                                                      │
                                                      ▼
                                         ┌─────────────────────────┐
                                         │  SQLite & CSV Database  │
                                         │ (results/experiments.db)│
                                         └─────────────────────────┘
```

---

## 2. Component Deconstruction

### 2.1 Core Subsystem (`app/core/`)
- **`config.py`**: Central YAML parser mapping models, routing thresholds, escalation parameters, hardware flags, and logging destinations. Supports environment overrides via `os.environ`.
- **`logger.py`**: Structured dual-destination logging emitting to standard stdout and `results/app.log`.
- **`exceptions.py`**: Domain-specific exception hierarchy ensuring graceful degradation.

### 2.2 Model Integration Subsystem (`app/models/`)
- **`base.py`**: Defines abstract class `BaseLLMClient` and standardized `ModelResponse` container.
- **`ollama_client.py`**: Communicates with local Ollama daemon over HTTP REST (`/api/generate` and `/api/tags`).
- **`offline_client.py`**: High-fidelity calibrated research model provider replicating latency and capability distributions of Qwen2.5:3b and Qwen2.5:7b on CPU-only machines.
- **`model_manager.py`**: Central provider resolver and sequential memory manager preventing Out-Of-Memory faults.

### 2.3 Difficulty Classification Engine (`app/classifier/`)
- **`feature_extractor.py`**: Extracts 18 numerical features measuring lexical, structural, and syntactic complexity.
- **`rule_based.py`**: Heuristic baseline scoring queries into Easy, Medium, and Hard tiers.
- **`train_classifier.py`**: Trains and cross-validates Logistic Regression, Random Forest, SVM, and Gradient Boosting.
- **`predict_difficulty.py`**: Real-time inference engine outputting predicted category, continuous score, confidence, and class probabilities.

### 2.4 Routing Subsystem (`app/routing/`)
- **`strategies.py`**: Contains decision rules for all 6 routing strategies (`always_small`, `always_large`, `rule_based`, `ml_adaptive`, `confidence_based`, `cascade`).
- **`escalation.py`**: Evaluates small-model outputs against quality, length, and explicit uncertainty criteria.
- **`router.py`**: Coordinates complexity prediction, model invocation, escalation handling, and telemetry collation.

### 2.5 Evaluation & Monitoring (`app/evaluation/`, `app/monitoring/`)
- **`quality_metrics.py`**: Exact Match, Token-level F1, ROUGE-1, ROUGE-2, and ROUGE-L via Longest Common Subsequence.
- **`llm_judge.py`**: Multi-dimensional evaluation across Correctness, Relevance, Completeness, Clarity, Reasoning, and Instruction Following (1.0 to 5.0 scale).
- **`latency.py`**: Aggregates latency distributions (P50, P90, P95, P99) and token throughputs.
- **`cost_calculator.py`**: Computes API token pricing, hardware kilowatt-hour energy, and comparative savings.

### 2.6 Persistence Subsystem (`app/database/`)
- **`db.py`**: SQLite relational schema with foreign key constraints.
- **`repository.py`**: CRUD layer and automated CSV export pipeline.

### 2.7 Frontend Subsystem (`dashboard/`)
- **`streamlit_app.py`**: Multi-page portal organizing the 7 interactive analytics views.
