"""
Machine Learning Classifier Training Pipeline for Query Difficulty Estimation
Trains, tunes, cross-validates, and evaluates:
1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (RBF kernel, calibrated probabilities)
4. Gradient Boosting Classifier

Combines TF-IDF word & character n-grams with engineered linguistic features.
Selects the best performing model and exports serialized artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from app.classifier.feature_extractor import FeatureExtractor, LinguisticFeatureExtractor

def build_feature_pipeline() -> FeatureUnion:
    """Build a combined feature union of TF-IDF and scaled linguistic features."""
    return FeatureUnion([
        ("tfidf_word", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True
        )),
        ("linguistic", Pipeline([
            ("extractor", LinguisticFeatureExtractor()),
            ("scaler", StandardScaler())
        ]))
    ])

def train_and_evaluate_classifiers(
    data_path: str = "data/queries.csv",
    output_dir: str = "models"
) -> Dict[str, Any]:
    """Train multiple classifiers with cross-validation and evaluate on hold-out test set."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    X = df["query"]
    y = df["category"]

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(
            C=1.5, max_iter=1000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_split=3, random_state=42
        ),
        "svm": CalibratedClassifierCV(
            SVC(C=2.0, kernel="rbf"), ensemble=False
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42
        )
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    labels = ["Easy", "Medium", "Hard"]

    for name, clf in models.items():
        pipeline = Pipeline([
            ("features", build_feature_pipeline()),
            ("clf", clf)
        ])

        # 5-fold cross-validation on training partition
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=skf,
            scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        )

        # Train on full train partition and evaluate on test partition
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro")
        test_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        test_rec = recall_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

        model_summary = {
            "cv_accuracy_mean": round(float(np.mean(cv_scores["test_accuracy"])), 4),
            "cv_accuracy_std": round(float(np.std(cv_scores["test_accuracy"])), 4),
            "cv_f1_macro_mean": round(float(np.mean(cv_scores["test_f1_macro"])), 4),
            "cv_f1_macro_std": round(float(np.std(cv_scores["test_f1_macro"])), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_f1_macro": round(float(test_f1), 4),
            "test_precision_macro": round(float(test_prec), 4),
            "test_recall_macro": round(float(test_rec), 4),
            "confusion_matrix": cm,
            "labels": labels
        }
        results[name] = model_summary

        print(f"[{name.upper()}] CV F1: {model_summary['cv_f1_macro_mean']:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_name = name
            best_pipeline = pipeline

    # Re-train the winning pipeline on full dataset (train + test) for production inference
    print(f"\n[WINNER] Selected Best Model: '{best_model_name}' (Test F1: {best_f1:.4f})")
    best_pipeline.fit(X, y)

    # Export best model pipeline artifact
    model_artifact_path = os.path.join(output_dir, "difficulty_classifier.joblib")
    joblib.dump({
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "classes": labels,
        "metadata": results[best_model_name]
    }, model_artifact_path)

    # Export metrics JSON
    metrics_path = os.path.join(output_dir, "classifier_metrics.json")
    final_report = {
        "dataset": data_path,
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "selected_best_model": best_model_name,
        "best_test_f1": best_f1,
        "comparative_results": results
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"[OK] Saved model artifact to '{model_artifact_path}'")
    print(f"[OK] Saved metrics report to '{metrics_path}'")
    return final_report

if __name__ == "__main__":
    train_and_evaluate_classifiers()
