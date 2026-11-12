# Experimental Framework & Protocol

## 1. Experimental Overview
The benchmarking framework systematically evaluates the 7 planned research experiments to test the capabilities, latency profiles, resource footprints, and financial costs of model routing.

---

## 2. Standardized Experiments

### Experiment 1: Small vs. Large Baseline
- **Objective**: Establish the absolute capability, latency, and cost baselines for $\mathcal{M}_s$ and $\mathcal{M}_l$ on identical queries.
- **Protocol**: 45 stratified queries evaluated under `always_small` and `always_large`.
- **Primary Metric**: Quality differential $\Delta Q = Q_l - Q_s$ and latency multiplier $T_l / T_s$.

### Experiment 2: Routing Strategy Benchmarking
- **Objective**: Compare all 6 routing strategies on the entire benchmark set:
  1. `always_small`
  2. `always_large`
  3. `rule_based`
  4. `adaptive` (ML-based)
  5. `confidence_based`
  6. `cascade` (Optimistic small pass with quality escalation)

### Experiment 3: Difficulty-Stratified Analysis
- **Objective**: Measure performance broken down per difficulty stratum (Level 1 Easy, Level 2 Medium, Level 3 Hard).
- **Hypothesis**: On Easy queries, quality parity is 100%. On Hard queries, small model quality collapses.

### Experiment 4: Quality vs. Cost Pareto Optimization
- **Objective**: Determine whether adaptive routing reaches the Pareto frontier.
- **Formulas**: Latency Savings (%), Cost Savings (%), Quality Retention (%).

### Experiment 5: Decision Threshold Sensitivity Analysis
- **Objective**: Sweep confidence thresholds $\tau \in \{0.20, 0.40, 0.60, 0.75, 0.85\}$ to map the trade-off curve between large model utilization and answer quality.

### Experiment 6: Cascade Escalation Dynamics
- **Objective**: Measure escalation frequency, initial failure detection, second-pass latency overhead, and quality recovery.

### Experiment 7: Architectural Component Ablation Study
- **Objective**: Isolate the individual contribution of:
  - Supervised ML classification (vs. Rule-based only)
  - Confidence thresholding (vs. Fixed threshold)
  - Cascade escalation (vs. Small-only without fallback)
