"""
Response Validation and Quality Checker for Cascade Escalation
Evaluates initial small-model outputs against quality, length, and certainty heuristics.
"""

from typing import Tuple, Dict, Any, List
from app.models.base import ModelResponse

DEFAULT_UNCERTAINTY_PHRASES = [
    "i don't know",
    "i do not know",
    "i am not sure",
    "i'm not sure",
    "cannot answer",
    "as an ai, i cannot",
    "unable to determine",
    "cannot guarantee",
    "difficult to express concisely",
    "not enough information",
    "involves calculus of variations and boundary terms. however"
]

class EscalationChecker:
    """Evaluates whether an initial model response meets quality criteria or warrants escalation."""

    def __init__(
        self,
        min_length_chars: int = 25,
        uncertainty_phrases: List[str] = None,
        check_malformed: bool = True
    ):
        self.min_length_chars = min_length_chars
        self.uncertainty_phrases = uncertainty_phrases or DEFAULT_UNCERTAINTY_PHRASES
        self.check_malformed = check_malformed

    def should_escalate(
        self,
        query: str,
        response: ModelResponse,
        predicted_difficulty: str = "Medium",
        difficulty_score: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Determines whether response should be escalated to the larger model.
        
        Returns:
            (escalate: bool, reason: str)
        """
        # Rule 1: Generation failure or empty response
        if not response.success or not response.response_text or not response.response_text.strip():
            return True, "Initial model returned empty, failed, or malformed response."

        text = response.response_text.strip()
        text_lower = text.lower()

        # Rule 2: Minimum response length violation
        if len(text) < self.min_length_chars:
            # Simple queries might legitimately be short (e.g. "120"), but if Medium/Hard, it's inadequate
            if predicted_difficulty in ["Medium", "Hard"]:
                return True, f"Response too brief ({len(text)} chars) for {predicted_difficulty} query."

        # Rule 3: Explicit model uncertainty or refusal
        for phrase in self.uncertainty_phrases:
            if phrase in text_lower:
                return True, f"Model expressed explicit uncertainty or lack of knowledge: '{phrase}'"

        # Rule 4: Structural inadequacy for Hard queries
        if predicted_difficulty == "Hard" and difficulty_score > 0.75:
            # If the query requires mathematical proof or system design and response lacks substance
            if len(text.split()) < 20:
                return True, "Substantive detail and derivations missing for high-complexity query."

        return False, "Response passed all quality validation checks."
