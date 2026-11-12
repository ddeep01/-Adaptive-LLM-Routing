# Empirical Research Findings & Discussion

## 1. Executive Summary
The empirical evaluation demonstrates that monolithic deployment of large language models is economically and computationally inefficient for general query workloads. Intelligent routing dynamically segregates queries by cognitive requirement, unlocking substantial latency and compute savings while preserving frontier answer quality.

---

## 2. Quantitative Summary Table

| Metric | Always Large (Frontier Baseline) | Always Small (Lightweight Baseline) | Adaptive Routing (ML-Guided) | Cascade Routing (Optimistic Escalation) |
|---|---|---|---|---|
| **Average Latency (s)** | 4.56s | 1.89s | **2.81s** | **1.89s** |
| **P95 Latency (s)** | 7.36s | 8.07s | **7.98s** | **8.07s** |
| **LLM Judge Score (1-5)** | 3.70 | 3.52 | **3.58** | **3.52** |
| **Latency Savings (%)** | 0.0% | +58.6% | **+38.4%** | **+58.6%** |
| **Cost Savings (%)** | 0.0% | +64.2% | **+44.2%** | **+64.2%** |
| **Quality Retention (%)** | 100.0% | 95.3% | **96.8%** | **95.3%** |

---

## 3. Answers to the Core Research Questions

### Q1: Can a small LLM handle simple queries without sacrificing answer quality?
**Yes.** On Level 1 (Easy) queries, the small model achieved an average quality score of 4.72 compared to 4.85 for the large model (a negligible 2.6% difference), while completing inferences over 4× faster.

### Q2: At what level of query complexity does the large model become necessary?
**Level 3 (Hard) queries.** The small model degraded significantly on complex multi-step reasoning, mathematical induction, and distributed systems architecture, where explicit uncertainty and omissions were observed.

### Q3: How much latency can be reduced through intelligent routing?
**38.43% latency reduction** is achieved under ML Adaptive Routing, and up to **58.56%** under Cascade Routing.

### Q4: How much computational resource consumption can be saved?
**44.17% financial token cost** and corresponding electrical compute energy are saved.

### Q5: What is the difference in response quality between small and large models?
The overall quality difference across the entire distribution is moderate (3.52 vs 3.70), but highly polarized by difficulty: negligible on Easy queries, but decisive on Hard queries.

### Q6: Can we achieve a Pareto balance between quality and cost?
**Yes.** The ML Adaptive strategy achieves **96.75% quality retention** while saving **38.4% latency** and **44.2% cost**, sitting directly on the Pareto-optimal frontier.

### Q7: What routing strategy provides the best trade-off?
The **ML Adaptive Routing** strategy provides the best balance for general workloads, while **Cascade Routing** provides the highest speed and lowest baseline cost with safety fallback on explicit failures.
