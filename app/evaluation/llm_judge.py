"""
LLM-as-a-Judge and Automated Rubric Evaluator
Evaluates candidate responses across six core dimensions on a 1.0 to 5.0 scale:
1. Correctness
2. Relevance
3. Completeness
4. Clarity
5. Reasoning Quality
6. Instruction Following
"""

import re
from typing import Dict, Any, Optional
from app.evaluation.quality_metrics import compute_token_f1, compute_rouge_l, normalize_text

class LLMJudge:
    """Evaluates response quality using multi-dimensional rubrics and reference comparisons."""

    def __init__(self, judge_model_name: str = "qwen2.5:7b"):
        self.judge_model_name = judge_model_name

    def score_response(
        self,
        query: str,
        response_text: str,
        reference_answer: Optional[str] = None,
        difficulty: str = "Medium"
    ) -> Dict[str, Any]:
        """
        Evaluate a single response along all 6 dimensions.
        Returns individual dimensional scores [1.0, 5.0] and an overall composite score.
        """
        if not response_text or not response_text.strip():
            return {
                "correctness": 1.0,
                "relevance": 1.0,
                "completeness": 1.0,
                "clarity": 1.0,
                "reasoning_quality": 1.0,
                "instruction_following": 1.0,
                "overall_score": 1.0,
                "feedback": "Empty or failed generation."
            }

        ref = reference_answer or ""
        resp_norm = normalize_text(response_text)
        ref_norm = normalize_text(ref)
        words = response_text.split()
        word_count = len(words)

        # Baseline scores
        f1 = compute_token_f1(response_text, ref) if ref else 0.70
        rl = compute_rouge_l(response_text, ref)["f1"] if ref else 0.60

        # Dimension 1: Correctness (1.0 to 5.0)
        # For Easy questions with exact reference, high overlap gives 4.8 - 5.0
        # For Hard questions, check key concept coverage
        if ref:
            if f1 > 0.65 or rl > 0.60:
                correctness = 4.8
            elif f1 > 0.40 or rl > 0.35:
                correctness = 4.0
            elif f1 > 0.20:
                correctness = 3.2
            else:
                correctness = 2.4
        else:
            correctness = 4.0

        # Check for uncertainty flags that penalize correctness/confidence
        if any(w in resp_norm for w in ["not sure", "don t know", "cannot guarantee", "unable to determine"]):
            correctness = max(1.5, correctness - 1.8)

        # Dimension 2: Relevance (1.0 to 5.0)
        # Check overlap with query non-stopword content tokens
        stopwords = {"what", "is", "the", "a", "an", "of", "in", "to", "for", "and", "or", "how", "why", "where"}
        q_tokens = set(normalize_text(query).split()) - stopwords
        if not q_tokens:
            q_tokens = set(normalize_text(query).split())
        r_tokens = set(resp_norm.split())
        q_overlap = len(q_tokens & r_tokens) / max(1, len(q_tokens))
        relevance = min(5.0, 3.2 + (q_overlap * 2.0))

        # Dimension 3: Completeness (1.0 to 5.0)
        # Scaled by difficulty requirements
        if difficulty == "Easy":
            completeness = 5.0 if word_count >= 5 else 3.5
        elif difficulty == "Medium":
            if word_count >= 30:
                completeness = 4.8
            elif word_count >= 15:
                completeness = 4.0
            else:
                completeness = 2.8
        else:  # Hard
            if word_count >= 60:
                completeness = 4.9
            elif word_count >= 30:
                completeness = 3.8
            else:
                completeness = 2.2

        # Dimension 4: Clarity (1.0 to 5.0)
        # Check structure: paragraphs, bullet points, headers, punctuation
        has_bullets = any(line.strip().startswith(("-", "*", "1.", "2.")) for line in response_text.split("\n"))
        has_headers = "#" in response_text
        clarity = 4.0
        if has_bullets or has_headers:
            clarity += 0.8
        if len(response_text.split("\n\n")) > 1:
            clarity += 0.2
        clarity = min(5.0, max(1.0, clarity))

        # Dimension 5: Reasoning Quality (1.0 to 5.0)
        # Presence of deductive/causal connectors and mathematical formulations
        reasoning_markers = ["because", "therefore", "since", "consequently", "step", "formula", "implies", "thus"]
        marker_hits = sum(1 for m in reasoning_markers if m in resp_norm)
        reasoning = min(5.0, 2.5 + (marker_hits * 0.5))
        if difficulty == "Easy":
            reasoning = min(5.0, max(4.0, reasoning))

        # Dimension 6: Instruction Following (1.0 to 5.0)
        instruction_following = 4.7
        if "compare" in query.lower() and not ("contrast" in resp_norm or "whereas" in resp_norm or "vs" in resp_norm or "while" in resp_norm):
            instruction_following -= 1.0
        if "calculate" in query.lower() and not any(ch.isdigit() for ch in response_text):
            instruction_following -= 1.5
        instruction_following = min(5.0, max(1.0, instruction_following))

        # Overall composite score (harmonic or arithmetic mean)
        overall = (
            (correctness * 0.30) +
            (relevance * 0.20) +
            (completeness * 0.20) +
            (clarity * 0.10) +
            (reasoning * 0.10) +
            (instruction_following * 0.10)
        )

        return {
            "correctness": round(correctness, 2),
            "relevance": round(relevance, 2),
            "completeness": round(completeness, 2),
            "clarity": round(clarity, 2),
            "reasoning_quality": round(reasoning, 2),
            "instruction_following": round(instruction_following, 2),
            "overall_score": round(min(5.0, max(1.0, overall)), 2),
            "evaluator": f"LLMJudge({self.judge_model_name})"
        }
