"""
Quality Evaluation Metrics
Implements Exact Match, Token-level F1, and ROUGE-1/2/L metrics
without requiring heavy external dependencies.
"""

import re
from typing import Dict, Any, List, Set, Tuple

def normalize_text(text: str) -> str:
    """Lowercases text and strips punctuation and extra whitespaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_exact_match(prediction: str, reference: str) -> float:
    """Exact Match (EM) binary score (1.0 if identical normalized text, else 0.0)."""
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0

def compute_token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score between prediction and reference."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    # Count matching tokens with frequency
    pred_counts = {t: pred_tokens.count(t) for t in common}
    ref_counts = {t: ref_tokens.count(t) for t in common}
    overlap = sum(min(pred_counts[t], ref_counts[t]) for t in common)

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)

def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generates contiguous token n-grams."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

def compute_rouge_n(prediction: str, reference: str, n: int = 1) -> Dict[str, float]:
    """Computes ROUGE-N precision, recall, and f1."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if len(pred_tokens) < n or len(ref_tokens) < n:
        match = 1.0 if pred_tokens == ref_tokens and len(pred_tokens) > 0 else 0.0
        return {"precision": match, "recall": match, "f1": match}

    pred_ngrams = get_ngrams(pred_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    pred_counts = {}
    for g in pred_ngrams:
        pred_counts[g] = pred_counts.get(g, 0) + 1

    ref_counts = {}
    for g in ref_ngrams:
        ref_counts[g] = ref_counts.get(g, 0) + 1

    overlap = 0
    for g, count in pred_counts.items():
        if g in ref_counts:
            overlap += min(count, ref_counts[g])

    prec = overlap / len(pred_ngrams) if pred_ngrams else 0.0
    rec = overlap / len(ref_ngrams) if ref_ngrams else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

def compute_lcs_length(x: List[str], y: List[str]) -> int:
    """Computes the Longest Common Subsequence length."""
    m, n = len(x), len(y)
    # Space-optimized 2-row DP table
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = list(curr)
    return prev[n]

def compute_rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """Computes ROUGE-L precision, recall, and f1 based on Longest Common Subsequence."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        match = 1.0 if pred_tokens == ref_tokens else 0.0
        return {"precision": match, "recall": match, "f1": match}

    lcs = compute_lcs_length(pred_tokens, ref_tokens)
    prec = lcs / len(pred_tokens)
    rec = lcs / len(ref_tokens)
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

def evaluate_quality(prediction: str, reference: str) -> Dict[str, float]:
    """Comprehensive evaluation returning EM, Token-F1, ROUGE-1, ROUGE-2, and ROUGE-L."""
    r1 = compute_rouge_n(prediction, reference, n=1)
    r2 = compute_rouge_n(prediction, reference, n=2)
    rl = compute_rouge_l(prediction, reference)
    em = compute_exact_match(prediction, reference)
    f1 = compute_token_f1(prediction, reference)

    return {
        "exact_match": round(em, 4),
        "token_f1": round(f1, 4),
        "rouge1_f1": r1["f1"],
        "rouge2_f1": r2["f1"],
        "rougeL_f1": rl["f1"]
    }
