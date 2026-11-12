"""
Linguistic and Syntactic Feature Extractor for Query Difficulty Estimation
Extracts dense numerical features measuring lexical, structural, and semantic complexity.
"""

import re
import math
from typing import Dict, Any, List

# Keyword Lexicons for Feature Engineering
CODE_KEYWORDS = {
    "def", "class", "function", "return", "import", "for", "while", "lambda",
    "async", "await", "public", "private", "void", "int", "struct", "pointer",
    "algorithm", "array", "stack", "queue", "heap", "tree", "graph", "hash",
    "linked list", "recursion", "complexity", "o(n)", "o(log n)", "sql", "api",
    "docker", "kubernetes", "concurrency", "thread", "lock", "mutex", "cas",
    "ring buffer", "rust", "c++", "python", "java", "kafka", "redis", "sharding"
}

REASONING_KEYWORDS = {
    "why", "because", "deduce", "infer", "derive", "prove", "justification",
    "theoretical", "hypothesis", "implication", "induction", "causation",
    "therefore", "consequently", "step by step", "first principles", "mechanism"
}

COMPARISON_KEYWORDS = {
    "compare", "contrast", "difference between", "versus", "vs", "trade-off",
    "tradeoffs", "advantages and disadvantages", "pros and cons", "distinguish",
    "differ from", "relative to"
}

CONSTRAINT_KEYWORDS = {
    "without", "only", "must", "constraint", "assume", "suppose", "strictly",
    "bounded", "invariant", "guarantee", "zero-downtime", "fault-tolerant",
    "non-negative", "real-time"
}

PROOF_DERIVATION_KEYWORDS = {
    "prove", "derivation", "derive", "induction", "theorem", "lemma",
    "formula", "equation", "eigenvalue", "integral", "derivative", "calculus",
    "differential", "schrodinger", "navier-stokes", "fourier", "euler-lagrange",
    "laplace", "poisson", "central limit theorem", "cauchy-schwarz"
}

SYSTEM_DESIGN_KEYWORDS = {
    "architect", "design", "distributed", "rate limiter", "high-availability",
    "consensus", "raft", "paxos", "sharded", "replication", "event-driven",
    "microservices", "cqrs", "failover", "throughput", "latency"
}

MATH_OPERATORS = set("+-*/^=<>%√∫∑∏∂≤≥≠≈")

class FeatureExtractor:
    """Extracts linguistic, syntactic, and complexity features from text queries."""

    def __init__(self):
        pass

    def extract_features(self, query: str) -> Dict[str, float]:
        """Extract a dictionary of numerical features from a single query string."""
        if not isinstance(query, str):
            query = str(query or "")

        text = query.strip()
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        word_count = max(1, len(words))
        char_length = len(text)
        
        # Word lengths
        word_lens = [len(w) for w in words]
        avg_word_length = sum(word_lens) / word_count if word_lens else 0.0

        # Sentences and punctuation
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = max(1, len(sentences))

        # Mathematical and numerical indicators
        math_symbol_count = sum(1 for ch in text if ch in MATH_OPERATORS)
        digit_count = sum(1 for ch in text if ch.isdigit())
        has_math_operators = 1.0 if math_symbol_count > 0 else 0.0

        # Keyword matching
        code_matches = sum(1 for kw in CODE_KEYWORDS if kw in text_lower)
        reasoning_matches = sum(1 for kw in REASONING_KEYWORDS if kw in text_lower)
        comparison_matches = sum(1 for kw in COMPARISON_KEYWORDS if kw in text_lower)
        constraint_matches = sum(1 for kw in CONSTRAINT_KEYWORDS if kw in text_lower)
        proof_matches = sum(1 for kw in PROOF_DERIVATION_KEYWORDS if kw in text_lower)
        design_matches = sum(1 for kw in SYSTEM_DESIGN_KEYWORDS if kw in text_lower)

        # Question components
        question_marks = text.count("?")
        commas = text.count(",")
        semicolons = text.count(";")
        colons = text.count(":")
        component_count = question_marks + commas + (semicolons * 2) + colons

        # Structural & readability estimates
        syllable_est = sum(max(1, len(re.findall(r"[aeiouy]+", w))) for w in words)
        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_est / word_count

        # Flesch-Kincaid Reading Ease approximation (lower = more complex)
        flesch_score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        # Normalize readability complexity index (higher = harder)
        readability_complexity = max(0.0, min(1.0, (100.0 - flesch_score) / 100.0))

        # Uppercase / Acronym count
        raw_words = re.findall(r"\b[A-Za-z]+\b", text)
        uppercase_words = sum(1 for w in raw_words if len(w) > 1 and w.isupper())

        return {
            "char_length": float(char_length),
            "word_count": float(word_count),
            "avg_word_length": round(avg_word_length, 3),
            "sentence_count": float(sentence_count),
            "words_per_sentence": round(words_per_sentence, 3),
            "math_symbol_count": float(math_symbol_count),
            "has_math_operators": has_math_operators,
            "digit_count": float(digit_count),
            "code_keyword_matches": float(code_matches),
            "reasoning_matches": float(reasoning_matches),
            "comparison_matches": float(comparison_matches),
            "constraint_matches": float(constraint_matches),
            "proof_matches": float(proof_matches),
            "design_matches": float(design_matches),
            "component_count": float(component_count),
            "readability_complexity": round(readability_complexity, 3),
            "uppercase_words": float(uppercase_words),
            "question_marks": float(question_marks)
        }

    def get_feature_names(self) -> List[str]:
        return [
            "char_length", "word_count", "avg_word_length", "sentence_count",
            "words_per_sentence", "math_symbol_count", "has_math_operators",
            "digit_count", "code_keyword_matches", "reasoning_matches",
            "comparison_matches", "constraint_matches", "proof_matches",
            "design_matches", "component_count", "readability_complexity",
            "uppercase_words", "question_marks"
        ]

    def extract_feature_vector(self, query: str) -> List[float]:
        feat_dict = self.extract_features(query)
        return [feat_dict[name] for name in self.get_feature_names()]


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer extracting custom linguistic features."""

    def __init__(self):
        self.extractor = FeatureExtractor()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import pandas as pd
        if isinstance(X, pd.Series):
            X = X.values
        features_list = [self.extractor.extract_feature_vector(str(q)) for q in X]
        return np.array(features_list)

