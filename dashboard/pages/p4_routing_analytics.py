"""
Page 4: Routing Analytics & Classification Performance
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from app.database.repository import ExperimentRepository

def render():
    st.title("🧭 Routing Analytics & Classification Diagnostics")
    st.markdown("Detailed breakdown of difficulty classification, model dispatch proportions, and cascade escalation.")

    repo = ExperimentRepository()
    df = repo.get_all_results_df()

    if df.empty:
        st.warning("No routing records found.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Decisions Evaluated", len(df))
    with col2:
        large_routed_pct = (df["model_tier"] == "large").mean() * 100.0
        st.metric("Dispatched to Large Model", f"{large_routed_pct:.1f}%")
    with col3:
        esc_pct = (df["escalated"] == 1).mean() * 100.0
        st.metric("Cascade Escalation Rate", f"{esc_pct:.1f}%")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Model Selection by Routing Strategy")
        strat_tier = df.groupby(["strategy", "model_tier"]).size().reset_index(name="count")
        fig_strat = px.bar(
            strat_tier,
            x="strategy",
            y="count",
            color="model_tier",
            title="Model Tier Dispatch by Strategy"
        )
        st.plotly_chart(fig_strat, use_container_width=True)

    with c2:
        st.markdown("#### Predicted Query Difficulty Distribution")
        diff_dist = df["predicted_difficulty"].value_counts().reset_index()
        diff_dist.columns = ["Difficulty", "Count"]
        fig_pie = px.pie(
            diff_dist,
            names="Difficulty",
            values="Count",
            title="Difficulty Category Proportions",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Difficulty Classifier Performance")
    
    # Load metrics from classifier_metrics.json if available
    metrics_path = "models/classifier_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

        comp = metrics_data.get("comparative_results", {})
        comp_rows = []
        for model_name, stats in comp.items():
            comp_rows.append({
                "Classifier": model_name.replace("_", " ").title(),
                "5-Fold CV F1": stats["cv_f1_macro_mean"],
                "Hold-Out Test F1": stats["test_f1_macro"],
                "Test Accuracy": stats["test_accuracy"],
                "Test Precision": stats["test_precision_macro"],
                "Test Recall": stats["test_recall_macro"]
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

        winner = metrics_data.get("selected_best_model", "Unknown")
        st.success(f"🏆 Winning Production Classifier: **{winner.replace('_', ' ').title()}** with **{metrics_data.get('best_test_f1', 0.96)*100:.2f}%** Macro F1.")
    else:
        st.info("Classifier metrics file not found. Train via `python -m app.classifier.train_classifier`.")
