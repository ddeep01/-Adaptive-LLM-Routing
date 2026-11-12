"""
Rule-Based Query Difficulty Estimator
Computes a heuristic complexity score based on measurable query features
and maps to difficulty tiers: Easy, Medium, Hard.
"""

from typing import Dict, Any
from app.classifier.feature_extractor import FeatureExtractor

class RuleBasedClassifier:
    """Heuristic rule-based query difficulty classifier."""

    def __init__(self):
        self.extractor = FeatureExtractor()

    def predict(self, query: str) -> Dict[str, Any]:
        """Estimate query difficulty using explicit linguistic and structural rules."""
        feats = self.extractor.extract_features(query)
        
        # Raw score accumulation
        score = 0.05  # Base level

        # Length contributions (longer queries often involve more context or problem framing)
        if feats["word_count"] > 35:
            score += 0.25
        elif feats["word_count"] > 18:
            score += 0.15
        elif feats["word_count"] > 10:
            score += 0.08

        # Proof, derivation, and rigorous mathematics
        if feats["proof_matches"] > 0:
            score += 0.35 + min(0.15, feats["proof_matches"] * 0.05)

        # Distributed system design
        if feats["design_matches"] > 0:
            score += 0.30 + min(0.15, feats["design_matches"] * 0.05)

        # Code keywords and programming constructs
        if feats["code_keyword_matches"] > 0:
            score += 0.15 + min(0.15, feats["code_keyword_matches"] * 0.03)

        # Reasoning keywords (why, because, deduce, derive)
        if feats["reasoning_matches"] > 0:
            score += 0.12 + min(0.10, feats["reasoning_matches"] * 0.03)

        # Comparison indicators (trade-offs, vs, differences)
        if feats["comparison_matches"] > 0:
            score += 0.15

        # Constraints and edge conditions
        if feats["constraint_matches"] > 0:
            score += 0.10

        # Mathematical operators and formulas
        if feats["has_math_operators"] > 0:
            score += 0.10 + min(0.10, feats["math_symbol_count"] * 0.02)

        # Syntactic complexity
        score += feats["readability_complexity"] * 0.12

        # Multiple question components
        if feats["component_count"] > 3:
            score += 0.08

        # Bound score to [0.05, 0.98]
        score = round(min(0.98, max(0.05, score)), 3)

        # Categorize
        if score < 0.36:
            predicted_category = "Easy"
            # Confidence decreases near the 0.36 boundary
            confidence = round(0.70 + (0.36 - score) * 0.8, 2)
        elif score < 0.71:
            predicted_category = "Medium"
            # Distance from boundaries
            dist_to_edge = min(score - 0.36, 0.71 - score)
            confidence = round(0.70 + dist_to_edge * 1.1, 2)
        else:
            predicted_category = "Hard"
            confidence = round(0.70 + (score - 0.71) * 0.9, 2)

        confidence = round(min(0.96, max(0.60, confidence)), 2)

        return {
            "query": query,
            "predicted_difficulty": predicted_category,
            "difficulty_score": score,
            "confidence": confidence,
            "predicted_by": "rule_based",
            "features": feats
        }
