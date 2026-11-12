"""
Adaptive LLM Routing - Master Streamlit Application
Multi-page research portal and interactive analytics dashboard.
"""

import sys
import os
# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

st.set_page_config(
    page_title="Adaptive LLM Routing Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("Adaptive LLM Routing")
st.sidebar.caption("Research & Evaluation Portal")

pages = {
    "📌 1. Project Overview": "p1_overview",
    "🚀 2. Live Query Interface": "p2_live_query",
    "⚖️ 3. Model Comparison": "p3_model_comparison",
    "🧭 4. Routing Analytics": "p4_routing_analytics",
    "📈 5. Quality vs. Cost": "p5_quality_cost",
    "🔬 6. Experiment Analysis": "p6_experiment_analysis",
    "📑 7. Research Findings": "p7_research_findings"
}

selection = st.sidebar.radio("Navigate Views:", list(pages.keys()))
page_module_name = pages[selection]

# Dynamically import and render selected page
if page_module_name == "p1_overview":
    from dashboard.pages import p1_overview
    p1_overview.render()
elif page_module_name == "p2_live_query":
    from dashboard.pages import p2_live_query
    p2_live_query.render()
elif page_module_name == "p3_model_comparison":
    from dashboard.pages import p3_model_comparison
    p3_model_comparison.render()
elif page_module_name == "p4_routing_analytics":
    from dashboard.pages import p4_routing_analytics
    p4_routing_analytics.render()
elif page_module_name == "p5_quality_cost":
    from dashboard.pages import p5_quality_cost
    p5_quality_cost.render()
elif page_module_name == "p6_experiment_analysis":
    from dashboard.pages import p6_experiment_analysis
    p6_experiment_analysis.render()
elif page_module_name == "p7_research_findings":
    from dashboard.pages import p7_research_findings
    p7_research_findings.render()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Hardware**: Intel i3-1305U (5 Cores, 8GB RAM)\n\n"
    "**Backend**: FastAPI + SQLite\n\n"
    "**ML Classifier**: Gradient Boosting (96.7% F1)"
)
