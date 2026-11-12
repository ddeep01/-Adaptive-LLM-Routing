"""
Automated Unit Tests for Feature Extractor and Difficulty Classifiers
"""

import os
import pytest
import pandas as pd
from app.classifier.feature_extractor import FeatureExtractor
from app.classifier.rule_based import RuleBasedClassifier
from app.classifier.predict_difficulty import DifficultyPredictor

def test_feature_extractor():
    fe = FeatureExtractor()
    feats = fe.extract_features("What is the capital of France?")
    assert feats["word_count"] > 0
    assert feats["char_length"] > 0
    assert "avg_word_length" in feats
    assert "has_math_operators" in feats

    # Test math detection
    math_feats = fe.extract_features("Calculate 25 * 4 = 100")
    assert math_feats["has_math_operators"] == 1.0

    # Test code keywords
    code_feats = fe.extract_features("def binary_search(arr): return -1")
    assert code_feats["code_keyword_matches"] > 0

def test_rule_based_classifier():
    rb = RuleBasedClassifier()
    # Easy query
    easy_res = rb.predict("What is HTML?")
    assert easy_res["predicted_difficulty"] == "Easy"
    assert easy_res["difficulty_score"] < 0.40

    # Hard query
    hard_res = rb.predict("Prove by induction that the sum of cubes equals the square of the sum for all n.")
    assert hard_res["predicted_difficulty"] == "Hard"
    assert hard_res["difficulty_score"] >= 0.70

def test_difficulty_predictor():
    predictor = DifficultyPredictor()
    pred = predictor.predict("What is RAM?")
    assert pred["predicted_difficulty"] in ["Easy", "Medium", "Hard"]
    assert 0.0 <= pred["difficulty_score"] <= 1.0
    assert 0.0 <= pred["confidence"] <= 1.0
    assert "class_probabilities" in pred or pred["predicted_by"] == "rule_based"

def test_dataset_integrity():
    assert os.path.exists("data/queries.csv"), "data/queries.csv missing"
    df = pd.read_csv("data/queries.csv")
    assert len(df) == 300, f"Expected 300 queries, found {len(df)}"
    assert df["category"].value_counts().to_dict() == {"Easy": 100, "Medium": 100, "Hard": 100}
    assert df["query"].nunique() == 300, "Found duplicate query texts"
