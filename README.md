# Adaptive LLM Routing

### An Experimental Study of Response Quality, Latency, GPU Utilization, and Computational Cost

Adaptive LLM Routing is an end-to-end research and production framework designed to dynamically route natural language queries between lightweight small language models (e.g., Qwen 2.5:3B) and heavyweight large language models (e.g., Qwen 2.5:7B) based on predicted query complexity, classifier confidence, and cascading quality validation checks.

---

## Key Features

1. **Intelligent Query Difficulty Engine**:
   - Feature extraction covering 18+ lexical, syntactic, and structural indicators.
   - Dual-tier classification: Rule-Based heuristic scoring alongside trained ML classifiers (Logistic Regression, Random Forest, Support Vector Machine, Gradient Boosting).
   - Probability calibration and confidence scoring.

2. **Six Comprehensive Routing Strategies**:
   - `Always Small`: Baseline low-cost, low-latency strategy.
   - `Always Large`: Baseline maximum-capability frontier strategy.
   - `Rule-Based Routing`: Fast keyword and syntax thresholding.
   - `ML-Based Adaptive Routing`: Supervised difficulty classification.
   - `Confidence-Based Routing`: Decision-boundary routing with confidence thresholds.
   - `Cascade / Escalation Routing`: Fast optimistic small-model pass followed by automated validation and on-demand large-model escalation.

3. **Multi-Faceted Evaluation Suite**:
   - Quality metrics: Exact Match, Token-level F1, ROUGE-1/2/L, and heuristic LLM-as-a-Judge grading (1–5 scale).
   - Performance metrics: Total latency, Time-to-First-Token (TTFT), tokens/sec throughput, cold vs warm timing.
   - Resource metrics: CPU percent, RAM peak consumption, GPU utilization & VRAM tracking.
   - Cost accounting: Token-based API pricing vs amortized hardware watt-hour compute cost.

4. **Reproducible Experimental Benchmarks**:
   - 7 standardized experiments covering model head-to-head comparison, strategy benchmarking, difficulty stratification, Pareto frontier analysis, threshold sensitivity, cascade dynamics, and component ablation.
   - SQLite relational storage (`results/experiments.db`) and automated CSV dataset exports.

5. **Full-Stack Application**:
   - **FastAPI Gateway**: High-performance REST endpoints with Pydantic v2 validation.
   - **Streamlit Research Dashboard**: 7 interactive analytics pages for live querying, model head-to-head comparison, routing analytics, Pareto frontiers, and experimental inspection.

---

## Directory Structure

```text
adaptive-llm-routing/
├── app/
│   ├── main.py                  # FastAPI application entrypoint
│   ├── api/                     # REST routes & Pydantic schemas
│   ├── core/                    # Central configuration, logger, exceptions
│   ├── models/                  # BaseLLMClient, Ollama, API, Offline mock provider
│   ├── classifier/              # Feature extractor, rule-based, ML trainer, predictor
│   ├── routing/                 # Central router, 6 strategies, escalation checks
│   ├── evaluation/              # Quality metrics, LLM judge, human eval template
│   ├── monitoring/              # Latency, resource monitor, cost calculator
│   ├── experiments/             # Benchmarks, experiment runner, ablation analysis
│   └── database/                # SQLite engine & repository
├── dashboard/
│   ├── streamlit_app.py         # Multi-page Streamlit portal
│   └── pages/                   # 7 analytical dashboard views
├── data/
│   ├── queries.csv              # 300 curated benchmark queries (100 Easy, 100 Med, 100 Hard)
│   ├── queries.json             # Structured JSON benchmark dataset
│   └── README.md                # Dataset documentation & provenance
├── models/                      # Serialized ML models & vectorizer artifacts
├── results/                     # SQLite database, benchmark logs, and CSV exports
├── tests/                       # Automated pytest suite
├── scripts/                     # Environment check, dataset generator, runner scripts
├── docs/                        # Architecture, Methodology, Experiments, Findings
├── config.yaml                  # System configuration
├── IMPLEMENTATION_LOG.md        # Living engineering log
└── FINAL_REPORT.md              # Complete scientific research report
```

---

## Quickstart

### 1. Installation
```powershell
python -m pip install -r requirements.txt
```

### 2. Verify Hardware Environment
```powershell
python scripts/check_environment.py
```

### 3. Generate Benchmark Dataset & Train Classifier
```powershell
python scripts/generate_dataset.py
python -m app.classifier.train_classifier
```

### 4. Run Benchmarks
```powershell
python scripts/run_experiments.py
```

### 5. Launch FastAPI Backend
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Swagger UI available at `http://127.0.0.1:8000/docs`.

### 6. Launch Streamlit Dashboard
```powershell
python -m streamlit run dashboard/streamlit_app.py --server.port 8501
```
Dashboard available at `http://localhost:8501`.

---

## Research Documentation
- [PROJECT_PLAN.md](file:///e:/Mlops/PROJECT_PLAN.md)
- [IMPLEMENTATION_LOG.md](file:///e:/Mlops/IMPLEMENTATION_LOG.md)
- [FINAL_REPORT.md](file:///e:/Mlops/FINAL_REPORT.md)
