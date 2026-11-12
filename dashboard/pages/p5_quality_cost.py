"""
Page 5: Quality vs. Cost Pareto Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from app.database.repository import ExperimentRepository
from app.experiments.analysis import ResultsAnalyzer

def render():
    st.title("📈 Quality vs. Cost Pareto Trade-Off Analysis")
    st.markdown("Investigates whether intelligent adaptive routing achieves Pareto-optimal efficiency over monolithic large model deployments.")

    repo = ExperimentRepository()
    analyzer = ResultsAnalyzer()
    
    # Get strategy aggregated metrics
    strat_df = analyzer.analyze_experiment_2()
    pareto_res = analyzer.compute_pareto_savings()
    sav = pareto_res.get("comparative_savings", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latency Savings", f"{sav.get('latency_savings_pct', 0.0)}%", "Relative to Always Large")
    with col2:
        st.metric("Financial Cost Savings", f"{sav.get('cost_savings_pct', 0.0)}%", "Relative to Always Large")
    with col3:
        st.metric("Quality Retention", f"{sav.get('quality_retention_pct', 100.0)}%", "Parity with Always Large")

    st.markdown("---")

    if not strat_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Quality vs. Latency Pareto Space")
            fig_ql = px.scatter(
                strat_df,
                x="avg_latency",
                y="avg_quality",
                color="strategy",
                size="large_model_pct",
                text="strategy",
                labels={"avg_latency": "Average Latency (s)", "avg_quality": "Average Judge Quality (1-5)"},
                title="Quality vs. Latency (Size = % Large Model)"
            )
            fig_ql.update_traces(textposition="top center")
            st.plotly_chart(fig_ql, use_container_width=True)

        with c2:
            st.markdown("#### Quality vs. Cost Pareto Space")
            fig_qc = px.scatter(
                strat_df,
                x="avg_cost",
                y="avg_quality",
                color="strategy",
                size="large_model_pct",
                text="strategy",
                labels={"avg_cost": "Average Cost per Query ($)", "avg_quality": "Average Judge Quality (1-5)"},
                title="Quality vs. Financial Cost"
            )
            fig_qc.update_traces(textposition="top center")
            st.plotly_chart(fig_qc, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Strategy Performance Benchmarks")
        st.dataframe(strat_df, use_container_width=True)
    else:
        st.info("No strategy benchmark data available. Run `python scripts/run_experiments.py`.")
