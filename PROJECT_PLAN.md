# PROJECT PLAN: Adaptive LLM Routing

## Title
**Adaptive LLM Routing: An Experimental Study of Response Quality, Latency, GPU Utilization, and Computational Cost**

---

## 1. Project Objectives
Modern Large Language Model (LLM) deployments face a fundamental tension: frontier/large models achieve superior reasoning and comprehension on challenging queries, but impose severe latency bottlenecks, VRAM footprints, and dollar/compute costs. Conversely, lightweight small models (e.g., 3B parameters) provide sub-second latency and minimal overhead, yet degrade on complex multi-step reasoning, coding, and mathematical deduction.

The objective of this project is to build, experimentally benchmark, and productionize an **Adaptive LLM Routing System** that estimates query complexity upstream and dynamically directs requests to the optimal model (Small vs. Large). 

The experimental study addresses seven core research questions:
1. *Can a small LLM handle simple queries without sacrificing answer quality?*
2. *At what level of query complexity does the large model become necessary?*
3. *How much latency can be reduced through intelligent routing?*
4. *How much computational resource consumption can be saved?*
5. *What is the difference in response quality between small and large models?*
6. *Can we achieve an optimal Pareto balance between answer quality and computational cost?*
7. *What routing strategy provides the best quality-cost trade-off across real-world workloads?*

---

## 2. Technical Architecture
The system follows a clean, decoupled, production-grade micro-architecture:

```
                  ┌────────────────────────────────────────┐
                  │          Client / Streamlit UI         │
                  └───────────────────┬────────────────────┘
                                      │ HTTP / JSON
                                      ▼
                  ┌────────────────────────────────────────┐
                  │            FastAPI Gateway             │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       Adaptive Routing Engine          │
                  │   (Router, Strategies, Escalation)     │
                  └───────┬──────────────────────┬─────────┘
                          │                      │
        Feature Extractor │                      │ Routing Decision
        & Difficulty ML   ▼                      ▼
    ┌──────────────────────────┐       ┌───────────────────────────┐
    │ Query Difficulty Engine  │       │       Model Manager       │
    │  - Rule-Based Heuristics │       │  (Ollama / API / Local)   │
    │  - ML Classifiers        │       └──────┬─────────────┬──────┘
    │  - Feature Vectorizer    │              │             │
    └──────────────────────────┘        Small │             │ Large
                                        Model ▼             ▼ Model
                                   ┌───────────┐       ┌───────────┐
                                   │ Qwen2.5:3b│       │ Qwen2.5:7b│
                                   └─────┬─────┘       └─────┬─────┘
                                         │                   │
                                         └─────────┬─────────┘
                                                   │
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │    Evaluation & Monitor     │
                                    │ - Latency & Resource Tracker│
                                    │ - Quality (ROUGE/F1/Judge)  │
                                    │ - Cost Accounting Engine    │
                                    └──────────────┬──────────────┘
                                                   │
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │  SQLite Database & CSV Store│
                                    └─────────────────────────────┘
```

### Core Subsystems:
1. **Core & Configuration (`app/core/`)**: YAML configuration, logging, custom exceptions.
2. **Model Management (`app/models/`)**: Unified interface supporting Ollama local inference, external LLM APIs, and simulated/mock backends for CPU/resource-constrained offline testing.
3. **Query Difficulty Estimation Engine (`app/classifier/`)**: Feature extraction (linguistic, syntactic, lexical, complexity indicators) + Rule-based scoring + Supervised ML classifiers (Logistic Regression, Random Forest, SVM, Gradient Boosting) with cross-validation and calibration.
4. **Adaptive Routing Engine (`app/routing/`)**: Six distinct routing strategies: Always Small, Always Large, Rule-Based, ML-Based Adaptive, Confidence-Based, and Cascade/Escalation with quality checks.
5. **Evaluation & Monitoring (`app/evaluation/`, `app/monitoring/`)**: Exact Match, Token-F1, ROUGE-1/2/L, LLM-as-a-Judge, latency tracking (TTFT, total inference, warmup/cold-start), resource profiling (CPU, RAM, GPU/VRAM via psutil/torch), and cost calculation (local vs API token pricing).
6. **Persistence & Benchmarking (`app/database/`, `app/experiments/`)**: SQLite relational storage + CSV export pipeline; automated experiment runner with reproducible seeds.
7. **FastAPI Backend (`app/api/`)**: REST API with Pydantic v2 schemas for health, routing, live query, evaluation, and experiment runs.
8. **Streamlit Research Dashboard (`dashboard/`)**: 7-page interactive UI (Overview, Live Query, Model Comparison, Routing Analytics, Quality vs Cost, Experiment Analysis, Research Findings).

---

## 3. Implementation Phases

- **Phase 1: Environment & Planning**
  - Verify system hardware, RAM, CPU cores, GPU availability.
  - Set up directory structure and `__init__.py` files.
  - Create `PROJECT_PLAN.md`, `IMPLEMENTATION_LOG.md`, `config.yaml`, `requirements.txt`.
  - Implement `scripts/check_environment.py`.

- **Phase 2: Dataset Creation & Query Classification**
  - Build balanced 300+ query dataset across Easy, Medium, Hard categories (100 Easy, 100 Medium, 100 Hard).
  - Store in `data/queries.csv` and `data/queries.json`.
  - Implement linguistic & lexical feature extractor (`app/classifier/feature_extractor.py`).
  - Implement rule-based baseline estimator (`app/classifier/rule_based.py`).
  - Implement ML classifier trainer (`app/classifier/train_classifier.py`) evaluating Logistic Regression, Random Forest, SVM, Gradient Boosting.
  - Save best trained model pipeline artifact to `models/difficulty_classifier.joblib`.
  - Implement prediction module (`app/classifier/predict_difficulty.py`).

- **Phase 3: LLM Integration & Model Manager**
  - Define unified base class `BaseLLMClient` with metrics collection.
  - Implement `OllamaClient` for local models (e.g. Qwen2.5:3b and Qwen2.5:7b).
  - Implement `OfflineResearchClient` for reproducible local CPU testing with calibrated latency and response traits.
  - Implement `ModelManager` supporting model switching, sequential loading to protect 8GB RAM, and health checks.

- **Phase 4: Adaptive Routing Engine & Strategies**
  - Implement 6 routing strategies:
    1. Always Small
    2. Always Large
    3. Rule-Based
    4. ML-Based Adaptive
    5. Confidence-Based
    6. Cascade / Escalation (Small model first -> validation check -> escalate to Large if failed)
  - Implement validation checks: emptiness, explicit uncertainty, minimum length, keyword coverage.

- **Phase 5: Evaluation Framework & Resource Monitoring**
  - Quality metrics: Exact Match, Token-F1, ROUGE-L, Heuristic/LLM Judge scoring (1-5 scale).
  - Performance monitoring: latency (loading time, inference time, tokens/sec), memory/CPU delta via `psutil`.
  - Cost calculations: local compute energy/hardware amortized cost vs API token pricing.
  - SQLite database schemas (`experiments`, `queries`, `routing_decisions`, `responses`, `metrics`).

- **Phase 6: Systematic Experimental Benchmarking**
  - Execute Experiment 1: Small vs Large baseline on identical test queries.
  - Execute Experiment 2: Routing Strategy Comparison (6 strategies).
  - Execute Experiment 3: Difficulty-wise performance breakdown (Easy, Medium, Hard).
  - Execute Experiment 4: Quality vs Cost Pareto trade-off.
  - Execute Experiment 5: Confidence/Difficulty threshold sensitivity analysis.
  - Execute Experiment 6: Cascade escalation failure and recovery analysis.
  - Execute Experiment 7: Ablation study of individual system components.
  - Export all results to SQLite and CSV.

- **Phase 7: Backend API (FastAPI)**
  - REST endpoints: `/health`, `/models`, `/query`, `/route`, `/evaluate`, `/experiments/run`, `/results`, `/metrics`, `/export`.
  - Pydantic models for request/response validation.

- **Phase 8: Streamlit Interactive Research Dashboard**
  - 7 pages: Overview, Live Query Interface, Model Comparison, Routing Analytics, Quality vs Cost, Experiment Analysis, Research Findings.
  - Live charts using Plotly and Seaborn.

- **Phase 9: Automated Testing & Reproducibility**
  - Comprehensive unit and integration test suite (`tests/`).
  - Verification of all endpoints, classifier, router, database, and metrics.

- **Phase 10: Research Documentation & Final Report**
  - Complete `IMPLEMENTATION_LOG.md` with full chronological logs.
  - Create `docs/ARCHITECTURE.md`, `docs/METHODOLOGY.md`, `docs/EXPERIMENTS.md`, `docs/METRICS.md`, `docs/FINDINGS.md`.
  - Author comprehensive `FINAL_REPORT.md` and `README.md`.

---

## 4. Required Dependencies
- Python 3.10+ (Current runtime: Python 3.14.6 AMD64)
- Core Data & ML: `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.3.0`, `joblib>=1.3.0`
- Web & API: `fastapi>=0.100.0`, `uvicorn>=0.23.0`, `pydantic>=2.0.0`, `httpx>=0.24.0`
- Dashboard & Plotting: `streamlit>=1.28.0`, `plotly>=5.15.0`, `seaborn>=0.12.0`, `matplotlib>=3.7.0`
- System Profiling: `psutil>=5.9.0`
- Testing: `pytest>=7.4.0`
- Configuration & I/O: `PyYAML>=6.0`, `requests>=2.31.0`

---

## 5. Planned Experiments
1. **Exp 1: Small vs. Large Model Performance Baseline**: Direct head-to-head comparison on latency, quality, memory, and cost.
2. **Exp 2: Strategy Benchmarking**: Compare all 6 routing strategies on the entire benchmark set.
3. **Exp 3: Difficulty-Stratified Analysis**: Granular analysis of response quality and latency per query level (Easy, Medium, Hard).
4. **Exp 4: Quality-Cost Pareto Optimization**: Quantitative measurement of Quality Retention (%) vs. Cost & Latency Savings (%).
5. **Exp 5: Threshold Parameter Sweep**: Varying difficulty/confidence thresholds ($\tau \in [0.2, 0.8]$) to map the frontier curve.
6. **Exp 6: Cascade Escalation Dynamics**: Quantifying escalation rate, second-pass latency overhead, and quality recovery.
7. **Exp 7: Component Ablation Study**: Evaluating contribution of ML classifier, confidence weighting, and escalation checks.

---

## 6. Expected Outputs
- `data/queries.csv` and `data/queries.json` (300 categorized queries).
- `models/difficulty_classifier.joblib` (Trained TF-IDF + Classifier pipeline).
- `models/classifier_metrics.json` (Cross-validation accuracy, precision, recall, F1, confusion matrix).
- SQLite Database: `results/experiments.db`.
- Exported CSV files in `results/` for all benchmark experiments.
- Live FastAPI service with Swagger documentation at `/docs`.
- Multi-page Streamlit Dashboard.
- Complete documentation suite and `FINAL_REPORT.md`.

---

## 7. Testing Strategy
- **Unit Tests**:
  - `test_classifier.py`: Feature extraction consistency, rule-based output range, ML model predictions and probability shapes.
  - `test_router.py`: Routing decisions for all 6 strategies, escalation triggers on failure states.
  - `test_metrics.py`: Correct computation of Exact Match, Token-F1, ROUGE, Latency savings, Cost savings, and Quality retention.
  - `test_database.py`: SQLite initialization, record insertion, querying, and CSV exports.
  - `test_api.py`: FastAPI endpoints status codes, request validation, error responses.
- **Integration Tests**:
  - End-to-end pipeline run from raw query -> feature extraction -> routing -> inference -> evaluation -> database storage.
