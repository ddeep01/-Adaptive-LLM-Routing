"""
Difficulty Predictor Module
Loads trained difficulty classifier (or rule-based fallback) to classify queries
into Easy, Medium, or Hard with continuous difficulty score and confidence.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, Optional
from app.classifier.rule_based import RuleBasedClassifier

# Category score calibrations: maps discrete class to center range
CATEGORY_CENTERS = {
    "Easy": 0.18,
    "Medium": 0.52,
    "Hard": 0.86
}

class DifficultyPredictor:
    """Predicts query complexity using either trained ML model or rule-based heuristics."""

    def __init__(self, model_path: str = "models/difficulty_classifier.joblib"):
        self.model_path = model_path
        self.rule_based_fallback = RuleBasedClassifier()
        self.pipeline = None
        self.model_name = "rule_based"
        self.classes = ["Easy", "Medium", "Hard"]
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.pipeline = data.get("pipeline")
                self.model_name = data.get("model_name", "ml_classifier")
                self.classes = list(data.get("classes", ["Easy", "Medium", "Hard"]))
            except Exception as e:
                print(f"[WARN] Failed to load ML model from {self.model_path}: {e}. Falling back to rule-based.")
                self.pipeline = None
                self.model_name = "rule_based"
        else:
            self.model_name = "rule_based"

    def predict(self, query: str, force_strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict query difficulty.
        
        Args:
            query: The input text query.
            force_strategy: If 'rule_based', uses the heuristic classifier. Otherwise uses ML if available.
        """
        if not isinstance(query, str) or not query.strip():
            return {
                "query": query,
                "predicted_difficulty": "Easy",
                "difficulty_score": 0.10,
                "confidence": 0.95,
                "predicted_by": "default_empty"
            }

        # Handle explicit rule-based request or missing pipeline
        if force_strategy == "rule_based" or self.pipeline is None:
            return self.rule_based_fallback.predict(query)

        try:
            # Predict probabilities
            probs = self.pipeline.predict_proba([query])[0]
            pred_idx = int(np.argmax(probs))
            predicted_class = self.classes[pred_idx]
            confidence = float(probs[pred_idx])

            # Compute continuous difficulty score based on class probabilities
            # Weights: Easy -> 0.18, Medium -> 0.52, Hard -> 0.86
            prob_dict = {cls: float(prob) for cls, prob in zip(self.classes, probs)}
            
            p_easy = prob_dict.get("Easy", 0.0)
            p_med = prob_dict.get("Medium", 0.0)
            p_hard = prob_dict.get("Hard", 0.0)

            # Continuous difficulty score in [0.05, 0.98]
            diff_score = (p_easy * CATEGORY_CENTERS["Easy"]) + \
                         (p_med * CATEGORY_CENTERS["Medium"]) + \
                         (p_hard * CATEGORY_CENTERS["Hard"])
            diff_score = round(min(0.98, max(0.05, diff_score)), 3)

            return {
                "query": query,
                "predicted_difficulty": predicted_class,
                "difficulty_score": diff_score,
                "confidence": round(confidence, 3),
                "predicted_by": self.model_name,
                "class_probabilities": {k: round(v, 4) for k, v in prob_dict.items()}
            }
        except Exception as e:
            # Safe degradation to rule-based estimator
            res = self.rule_based_fallback.predict(query)
            res["fallback_reason"] = str(e)
            return res
