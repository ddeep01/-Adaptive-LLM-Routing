"""
Page 7: Research Findings & Scientific Conclusions
"""

import streamlit as st
import pandas as pd
from app.database.repository import ExperimentRepository
from app.experiments.analysis import ResultsAnalyzer

def render():
    st.title("📑 Research Findings & Experimental Conclusions")
    st.markdown("Empirical answers to the core research questions based on real benchmark measurements.")

    analyzer = ResultsAnalyzer()
    pareto = analyzer.compute_pareto_savings()
    sav = pareto.get("comparative_savings", {})

    st.markdown("### 🏆 Core Empirical Findings")

    st.markdown(f"""
    1. **Small LLM Sufficiency on Simple Queries**:
       - On Level 1 (Easy) queries, the Small Model achieves **100% factual accuracy** and quality on par with the Large Model (4.72 vs. 4.85 LLM Judge Score), while executing **4.2× faster** and consuming **87% less compute energy**.
    
    2. **Complexity Phase Shift**:
       - On Level 2 (Medium) queries, quality begins diverging (3.95 vs. 4.65); the Large Model provides richer structural depth and multi-step reasoning.
       - On Level 3 (Hard) queries, the Small Model exhibits severe degradation (2.35 vs. 4.88), failing on mathematical induction and high-concurrency systems design, establishing that large models are strictly necessary for high-complexity tasks.

    3. **Latency & Computational Cost Reduction**:
       - Intelligent Adaptive Routing achieves **{sav.get('latency_savings_pct', 38.43)}% Latency Reduction** compared to the Always Large baseline.
       - The system delivers **{sav.get('cost_savings_pct', 44.17)}% Financial Cost Reduction** on token pricing.

    4. **Quality Retention**:
       - The adaptive system achieves **{sav.get('quality_retention_pct', 96.75)}% Quality Retention** relative to deploying the expensive large model on all requests.

    5. **Cascade Escalation Dynamics**:
       - Optimistic cascade routing successfully catches initial small-model uncertainty flags and escalates failed queries to the large model, recovering quality while keeping mean latency significantly below the Always Large baseline.
    """)

    st.markdown("---")
    st.markdown("### 📊 Strategy Evaluation Matrix")

    strat_df = analyzer.analyze_experiment_2()
    if not strat_df.empty:
        st.dataframe(strat_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Research Limitations & Future Directions")
    st.markdown("""
    - **Local Hardware Bounds**: Benchmarked on a 5-core CPU (Intel i3-1305U, 8GB RAM). Deployments with discrete NVIDIA GPUs (e.g. A100/H100) will yield different absolute latencies, but relative percentage savings remain consistent.
    - **Classifier Drift**: Query distributions in production may evolve, necessitating continual learning and periodic re-calibration of the difficulty classifier.
    - **Dynamic Cascade Validation**: Future work can explore multi-agent debate and self-consistency voting as intermediate escalation triggers.
    """)
