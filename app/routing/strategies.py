"""
Routing Strategies Implementation
Provides decision logic for model tier selection across all 6 experimental strategies.
"""

from typing import Dict, Any, Tuple

class RoutingStrategyRegistry:
    """Implements routing policies mapping query characteristics to model tiers ('small' or 'large')."""

    @staticmethod
    def always_small(query: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        return "small", "Strategy 'always_small': Every query is assigned to the small model."

    @staticmethod
    def always_large(query: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        return "large", "Strategy 'always_large': Every query is assigned to the large model."

    @staticmethod
    def rule_based(query: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        score = metadata.get("difficulty_score", 0.5)
        if score < 0.40:
            return "small", f"Rule-based: Low complexity score ({score:.2f} < 0.40)."
        elif score < 0.70:
            return "small", f"Rule-based: Moderate complexity score ({score:.2f} in [0.40, 0.70))."
        else:
            return "large", f"Rule-based: High complexity score ({score:.2f} >= 0.70)."

    @staticmethod
    def ml_adaptive(query: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        cat = metadata.get("predicted_difficulty", "Medium")
        score = metadata.get("difficulty_score", 0.5)
        
        if cat == "Easy":
            return "small", f"ML Adaptive: Easy complexity query (score {score:.2f})."
        elif cat == "Hard":
            return "large", f"ML Adaptive: Hard complexity query requires frontier capabilities (score {score:.2f})."
        else: # Medium
            if score > 0.55:
                return "large", f"ML Adaptive: Upper-tier Medium query routed to large model (score {score:.2f})."
            else:
                return "small", f"ML Adaptive: Manageable Medium query routed to small model (score {score:.2f})."

    @staticmethod
    def confidence_based(query: str, metadata: Dict[str, Any], conf_threshold: float = 0.75) -> Tuple[str, str]:
        cat = metadata.get("predicted_difficulty", "Medium")
        conf = metadata.get("confidence", 0.80)
        score = metadata.get("difficulty_score", 0.5)

        if cat == "Easy":
            return "small", f"Confidence routing: Confirmed Easy query (conf {conf:.2f})."
        elif cat == "Hard":
            return "large", f"Confidence routing: Confirmed Hard query (conf {conf:.2f})."
        else: # Medium
            # If classifier confidence is high, model can trust small model routing;
            # if confidence is low, risk is high, escalate conservatively to large model
            if conf >= conf_threshold and score < 0.60:
                return "small", f"Confidence routing: Medium query with high confidence ({conf:.2f} >= {conf_threshold:.2f})."
            else:
                return "large", f"Confidence routing: Medium query with uncertainty ({conf:.2f} < {conf_threshold:.2f}) or elevated score."

    @staticmethod
    def cascade_initial(query: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        cat = metadata.get("predicted_difficulty", "Medium")
        if cat == "Easy":
            return "small", "Cascade: Initial pass allocated to small model for Easy query."
        elif cat == "Medium":
            return "small", "Cascade: Initial pass allocated to small model for Medium query."
        else:
            # If clearly Hard, we can still start small to test escalation, or route directly if hard
            return "small", "Cascade: Optimistic initial pass allocated to small model to evaluate sufficiency."
