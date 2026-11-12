# Adaptive LLM Routing Benchmark Dataset

## Dataset Summary
This benchmark dataset consists of **300 uniquely curated, balanced queries** designed to evaluate LLM routing engines across query complexity strata:
- **Level 1 — Easy**: 100 queries
- **Level 2 — Medium**: 100 queries
- **Level 3 — Hard**: 100 queries

---

## Schema & Record Specification

Each record in `queries.csv` and `queries.json` contains:

| Field | Type | Description | Example |
|---|---|---|---|
| `query_id` | String | Unique identifier (`Q001` to `Q300`) | `"Q001"` |
| `query` | String | Natural language user prompt | `"What does HTML stand for?"` |
| `category` | String | Complexity stratum (`Easy`, `Medium`, `Hard`) | `"Easy"` |
| `difficulty_score` | Float | Calibrated continuous difficulty score $[0.0, 1.0]$ | `0.09` |
| `domain` | String | Topical field (CS, Math, Science, Economics, etc.) | `"Computer Science"` |
| `expected_capability` | String | Requisite cognitive capability | `"Direct factual lookup"` |
| `reference_answer` | String | Ground-truth reference answer | `"HyperText Markup Language"` |

---

## Difficulty Strata Definitions

### Level 1 — Easy ($D \in [0.05, 0.35]$)
- Direct factual lookups, vocabulary definitions, unit conversions, single-step arithmetic, basic definitions.
- Suitable for ultra-lightweight models (e.g. 1B–3B parameters).

### Level 2 — Medium ($D \in [0.36, 0.70]$)
- Multi-step calculations, comparative explanations, conceptual trade-off analyses, process descriptions.
- Requires coherent multi-sentence reasoning and domain grounding.

### Level 3 — Hard ($D \in [0.71, 1.00]$)
- Mathematical induction, tensor and calculus derivations, distributed systems architecture, formal security proofs, algorithmic edge-case design.
- Requisite for frontier/large LLMs (e.g. 7B+ parameters).

---

## Provenance and Validation
- **Source**: Systematically curated and validated across STEM, humanities, economics, and systems architecture.
- **Deduplication**: Verified 300/300 unique text queries via automated cryptographic string hashing.
- **Validation**: Generated and verified through `scripts/generate_dataset.py`.
   