"""
Page 6: Experiment Analysis & Data Inspector
"""

import streamlit as st
import pandas as pd
from app.database.repository import ExperimentRepository

def render():
    st.title("🔬 Detailed Experiment Inspector")
    st.markdown("Filter, inspect, and export granular query-level routing decisions, latencies, and evaluations.")

    repo = ExperimentRepository()
    exps = repo.get_experiments()

    if not exps:
        st.warning("No experiment records found in database.")
        return

    exp_ids = [e["experiment_id"] for e in exps]
    selected_exp = st.selectbox("Select Experiment Run:", exp_ids, index=0)

    df = repo.get_all_results_df(selected_exp)

    if df.empty:
        st.info("No records found for this experiment.")
        return

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        cat_filter = st.multiselect("Filter by Difficulty:", options=list(df["true_category"].dropna().unique()), default=list(df["true_category"].dropna().unique()))
    with c2:
        strat_filter = st.multiselect("Filter by Strategy:", options=list(df["strategy"].dropna().unique()), default=list(df["strategy"].dropna().unique()))
    with c3:
        tier_filter = st.multiselect("Filter by Model Tier:", options=list(df["model_tier"].dropna().unique()), default=list(df["model_tier"].dropna().unique()))

    filtered = df[
        df["true_category"].isin(cat_filter) &
        df["strategy"].isin(strat_filter) &
        df["model_tier"].isin(tier_filter)
    ]

    st.markdown(f"**Displaying {len(filtered)} / {len(df)} records**")
    
    st.dataframe(
        filtered[[
            "query_id", "query", "true_category", "strategy", "model_tier",
            "model_name", "total_latency", "overall_judge_score", "token_f1", "escalated"
        ]],
        use_container_width=True
    )

    # Detailed inspection of single query
    st.markdown("### 🔍 Query Deep-Dive")
    q_selected = st.selectbox("Select a Query ID to inspect response text & breakdown:", options=filtered["query_id"].unique())
    if q_selected:
        q_row = filtered[filtered["query_id"] == q_selected].iloc[0]
        st.markdown(f"**Query**: {q_row['query']}")
        st.markdown(f"**Strategy**: `{q_row['strategy']}` | **Model**: `{q_row['model_name']}` ({q_row['model_tier']})")
        st.markdown(f"**Latency**: {q_row['total_latency']:.3f}s | **Quality Score**: {q_row['overall_judge_score']} / 5.0 | **Escalated**: {bool(q_row['escalated'])}")
        
        # Pull full response from DB
        resp_records = repo.get_responses_by_experiment(selected_exp)
        matching = [r for r in resp_records if r["query_id"] == q_selected and r["strategy"] == q_row["strategy"]]
        if matching:
            st.markdown("#### Response Text:")
            st.markdown(matching[0]["response_text"])

    # Export to CSV
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered Table to CSV",
        data=csv_bytes,
        file_name=f"{selected_exp}_filtered.csv",
        mime="text/csv"
    )
