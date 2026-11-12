"""
Page 3: Model Comparison (Small vs. Large)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from app.database.repository import ExperimentRepository

def render():
    st.title("⚖️ Model Head-to-Head Comparison")
    st.markdown("Empirical comparison of the Small Model (e.g. 3B) vs. Large Model (e.g. 7B) on identical benchmark queries.")

    repo = ExperimentRepository()
    df = repo.get_all_results_df("exp_001_small_vs_large")

    if df.empty:
        # Fallback to general results
        df = repo.get_all_results_df()

    if df.empty:
        st.warning("No benchmark records found. Please run the benchmark experiments.")
        return

    # Filter for Small and Large baselines
    base_df = df[df["strategy"].isin(["always_small", "always_large"])]
    if base_df.empty:
        base_df = df

    # Aggregated Summary Table
    summary = base_df.groupby("model_tier").agg(
        Mean_Latency=("total_latency", "mean"),
        P95_Latency=("total_latency", lambda x: x.quantile(0.95)),
        Mean_Tokens_Sec=("tokens_per_second", "mean"),
        Mean_Judge_Quality=("overall_judge_score", "mean"),
        Mean_Token_F1=("token_f1", "mean"),
        Mean_Cost_USD=("api_cost_usd", "mean")
    ).round(4).reset_index()

    st.markdown("### 📊 Benchmark Summary Table")
    st.dataframe(summary, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Latency Distribution by Model Tier")
        fig_lat = px.box(
            base_df,
            x="model_tier",
            y="total_latency",
            color="model_tier",
            labels={"model_tier": "Model Tier", "total_latency": "Latency (seconds)"},
            title="Inference Latency: Small vs. Large"
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    with c2:
        st.markdown("#### Quality (LLM Judge) by Difficulty Category")
        fig_qual = px.bar(
            base_df.groupby(["true_category", "model_tier"])["overall_judge_score"].mean().reset_index(),
            x="true_category",
            y="overall_judge_score",
            color="model_tier",
            barmode="group",
            labels={"overall_judge_score": "Mean Judge Score (1-5)", "true_category": "Difficulty Level"},
            title="Quality Retention across Difficulty Strata"
        )
        st.plotly_chart(fig_qual, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Token Generation Throughput (tokens/second)")
        fig_tps = px.histogram(
            base_df,
            x="tokens_per_second",
            color="model_tier",
            barmode="overlay",
            title="Generation Speed Distribution"
        )
        st.plotly_chart(fig_tps, use_container_width=True)

    with c4:
        st.markdown("#### Cost Profile per Query (USD)")
        fig_cost = px.bar(
            summary,
            x="model_tier",
            y="Mean_Cost_USD",
            color="model_tier",
            title="Average Financial Cost per Query ($)"
        )
        st.plotly_chart(fig_cost, use_container_width=True)
