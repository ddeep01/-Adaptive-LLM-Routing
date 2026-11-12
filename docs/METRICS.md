# Metrics Specification: Performance, Quality, and Cost

## 1. Quality Evaluation Metrics

### 1.1 Exact Match (EM)
Binary metric measuring exact string equality after lowercase normalization and punctuation stripping:
$$\text{EM}(y, \hat{y}) = \begin{cases} 1 & \text{if } \text{norm}(y) = \text{norm}(\hat{y}) \\ 0 & \text{otherwise} \end{cases}$$

### 1.2 Token-Level F1 Score
Harmonic mean of precision and recall over token multisets:
$$\text{Precision} = \frac{|T_{\text{pred}} \cap T_{\text{ref}}|}{|T_{\text{pred}}|}, \quad \text{Recall} = \frac{|T_{\text{pred}} \cap T_{\text{ref}}|}{|T_{\text{ref}}|}$$
$$\text{Token-F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 1.3 ROUGE-L (Longest Common Subsequence)
Measures sequence preservation between candidate and reference:
$$R_{\text{LCS}} = \frac{\text{LCS}(\text{pred}, \text{ref})}{m}, \quad P_{\text{LCS}} = \frac{\text{LCS}(\text{pred}, \text{ref})}{n}$$
$$\text{ROUGE-L} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}$$

### 1.4 LLM-as-a-Judge Composite Score
Multi-dimensional evaluation across 6 rubrics on a $[1.0, 5.0]$ scale:
$$Q = 0.30 \cdot C_{\text{correct}} + 0.20 \cdot R_{\text{relevance}} + 0.20 \cdot C_{\text{complete}} + 0.10 \cdot C_{\text{clarity}} + 0.10 \cdot R_{\text{reason}} + 0.10 \cdot I_{\text{instruct}}$$

---

## 2. Latency Metrics
- **Total Latency ($T_{\text{total}}$)**: Wall-clock time from request dispatch to final token receipt.
- **Loading Time ($T_{\text{load}}$)**: Cold-start model weights deserialization and allocation latency.
- **Inference Time ($T_{\text{infer}}$)**: $T_{\text{total}} - T_{\text{load}}$.
- **Throughput**: $\text{Tokens/Sec} = \frac{N_{\text{out}}}{T_{\text{infer}}}$.
- **P95 Latency**: 95th percentile latency observation across the workload distribution.

---

## 3. Cost & Energy Metrics

### 3.1 API Token Pricing Model
$$C_{\text{api}} = \left(\frac{N_{\text{in}}}{1000} \times P_{\text{in}}\right) + \left(\frac{N_{\text{out}}}{1000} \times P_{\text{out}}\right)$$

### 3.2 Local Compute Energy Model
$$E_{\text{kWh}} = \left(\frac{T_{\text{seconds}}}{3600}\right) \times \left(\frac{W_{\text{estimated}}}{1000}\right)$$
$$C_{\text{energy}} = E_{\text{kWh}} \times \text{Rate}_{\$/\text{kWh}}$$

### 3.3 Comparative Savings Equations
$$\text{Latency Savings (\%)} = \frac{T_{\text{large}} - T_{\text{adaptive}}}{T_{\text{large}}} \times 100$$
$$\text{Cost Savings (\%)} = \frac{C_{\text{large}} - C_{\text{adaptive}}}{C_{\text{large}}} \times 100$$
$$\text{Quality Retention (\%)} = \frac{Q_{\text{adaptive}}}{Q_{\text{large}}} \times 100$$
