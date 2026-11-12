"""
Page 2: Live Query Interface
"""

import streamlit as st
import time
from app.routing.router import LLMRouter
from app.monitoring.cost_calculator import CostCalculator

def render():
    st.title(" Live Query & Routing Console")
    st.markdown("Test query complexity estimation, dynamic model routing, and cascade execution in real time.")

    router = LLMRouter()
    cost_calc = CostCalculator()

    col_input, col_ctrl = st.columns([3, 1])
    with col_input:
        default_q = "Explain the difference between supervised and unsupervised learning in machine learning."
        query_text = st.text_area("Enter your query:", value=default_q, height=100)

    with col_ctrl:
        strategy = st.selectbox(
            "Routing Strategy",
            ["adaptive", "cascade", "confidence_based", "rule_based", "always_small", "always_large"],
            index=0
        )
        compare_both = st.checkbox("Compare Small & Large Side-by-Side", value=False)
        run_btn = st.button("🚀 Route & Generate", type="primary", use_container_width=True)

    if run_btn and query_text.strip():
        with st.spinner("Classifying complexity and executing model..."):
            result = router.route_and_generate(query=query_text, strategy=strategy)

        # Telemetry metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            diff_color = "green" if result["difficulty"] == "Easy" else "orange" if result["difficulty"] == "Medium" else "red"
            st.metric("Predicted Difficulty", f"{result['difficulty']} ({result['difficulty_score']:.2f})")
        with c2:
            st.metric("Classifier Confidence", f"{result['confidence'] * 100:.1f}%", help=f"Estimated by {result['classifier_used']}")
        with c3:
            st.metric("Selected Model", f"{result['selected_model']} ({result['selected_tier'].upper()})")
        with c4:
            st.metric("Total Latency", f"{result['total_latency']:.3f} s", f"{result['tokens_per_second']:.1f} tok/s")

        st.info(f"**Routing Explanation**: {result['routing_reason']}")

        if result.get("escalated"):
            st.warning(f"⚠️ **Cascade Escalation Triggered!** Initial pass failed validation checks. Escalated to Large model.\n*Reason*: {result.get('escalation_reason')}")

        st.markdown("### 📝 Generated Response")
        st.markdown(result["response"])

        # Cost and resources expander
        with st.expander("📊 Resource & Cost Telemetry Details"):
            costs = cost_calc.compute_query_cost(
                model_tier=result["selected_tier"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                latency_seconds=result["total_latency"]
            )
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Input Tokens", result["input_tokens"])
            rc2.metric("Output Tokens", result["output_tokens"])
            rc3.metric("API Token Cost", f"${costs['api_token_cost_usd']:.6f}")
            rc4.metric("Hardware Energy Cost", f"${costs['energy_cost_usd']:.6f}")

        # Side-by-side comparison if requested
        if compare_both:
            st.markdown("---")
            st.markdown("### ⚖️ Side-by-Side Model Output Comparison")
            col_s, col_l = st.columns(2)
            with col_s:
                st.markdown("#### Small Model (e.g. Qwen 2.5:3B)")
                with st.spinner("Generating small model response..."):
                    res_s = router.model_manager.generate("small", query_text)
                st.info(f"Latency: {res_s.total_latency:.3f}s | Output: {res_s.output_tokens} tokens")
                st.markdown(res_s.response_text)

            with col_l:
                st.markdown("#### Large Model (e.g. Qwen 2.5:7B)")
                with st.spinner("Generating large model response..."):
                    res_l = router.model_manager.generate("large", query_text)
                st.success(f"Latency: {res_l.total_latency:.3f}s | Output: {res_l.output_tokens} tokens")
                st.markdown(res_l.response_text)
