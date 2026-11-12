# Methodology: Query Complexity Estimation and Adaptive Routing

## 1. Problem Formulation
Let $\mathcal{Q}$ denote the space of natural language queries. We possess two language models:
- A small, lightweight model $\mathcal{M}_s$ characterized by low latency $T_s$, low inference cost $C_s$, and high throughput, but bounded cognitive capability.
- A large, frontier model $\mathcal{M}_l$ characterized by elevated latency $T_l \gg T_s$, high inference cost $C_l \gg C_s$, and superior multi-step reasoning capabilities.

Our objective is to discover a routing policy $\pi: \mathcal{Q} \to \{\mathcal{M}_s, \mathcal{M}_l\}$ that maximizes answer quality $Q(\pi(q))$ while minimizing total latency $T(\pi(q))$ and financial cost $C(\pi(q))$.

---

## 2. Feature Extraction & Engineering
Each query $q \in \mathcal{Q}$ is mapped into an 18-dimensional numerical feature vector $\mathbf{x} = \phi(q)$ capturing:
1. **Lexical Dimensions**: Total characters, total words, average word length.
2. **Structural Dimensions**: Total sentences, words-per-sentence, punctuation density.
3. **Domain & Task Flags**:
   - $\mathbb{I}_{\text{math}}$: Presence of mathematical operators ($+, -, *, /, \int, \sum, \partial, \sqrt{}$).
   - $\mathbb{I}_{\text{code}}$: Count of programming keywords (`def`, `class`, `concurrency`, `pointer`, etc.).
   - $\mathbb{I}_{\text{reason}}$: Count of causal/deductive keywords (`derive`, `prove`, `induction`, `therefore`, etc.).
   - $\mathbb{I}_{\text{compare}}$: Presence of comparative markers (`difference between`, `versus`, `trade-offs`).
   - $\mathbb{I}_{\text{design}}$: Indicators of high-concurrency systems design (`rate limiter`, `consensus`, `raft`, `sharded`).
4. **Readability & Complexity Index**: Normalized inverse Flesch Reading Ease score.

---

## 3. Supervised Classification Pipeline
We train a pipeline combining sublinear TF-IDF character and word n-grams (1-2) with scaled linguistic features:
$$\Phi(q) = [\text{TF-IDF}(q) \parallel \text{StandardScaler}(\phi(q))]$$

Four candidate classifiers were trained using 5-fold Stratified Cross-Validation on the 300 benchmark queries:
- **Logistic Regression**: $L_2$ penalty, $C=1.5$.
- **Random Forest**: 150 estimators, max depth 12.
- **Support Vector Machine**: Calibrated RBF kernel, $C=2.0$.
- **Gradient Boosting**: 120 estimators, learning rate 0.08, max depth 4.

**Model Selection Outcome**:
Gradient Boosting achieved the highest generalization performance with **96.67% hold-out test accuracy** and **0.9667 macro F1**, correctly segmenting queries across difficulty boundaries.

---

## 4. Continuous Difficulty Calibration
Rather than forcing a hard discrete threshold, the continuous difficulty score $D(q) \in [0.05, 0.98]$ is formulated as the expectation over the calibrated class probability distribution:
$$D(q) = \sum_{c \in \{\text{Easy}, \text{Med}, \text{Hard}\}} P(Y=c \mid q) \cdot \mu_c$$
Where $\mu_{\text{Easy}} = 0.18$, $\mu_{\text{Med}} = 0.52$, and $\mu_{\text{Hard}} = 0.86$.

---

## 5. Cascade Escalation & Quality Verification
For cascade routing, the query is optimistically routed to $\mathcal{M}_s$. The generated response $R_s$ is subjected to automated validation checks:
1. **Emptiness Check**: Length of stripped response $< 1$.
2. **Minimum Length Check**: Length $< 25$ characters for Medium/Hard queries.
3. **Explicit Uncertainty Check**: Detection of refusal or uncertainty phrases (e.g., *"I am not sure"*, *"unable to determine"*, *"cannot guarantee"*).
4. **Structural Completeness Check**: Verifies substance on Hard queries.

If validation fails, the query is escalated to $\mathcal{M}_l$.
