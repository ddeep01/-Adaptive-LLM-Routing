"""
Central LLM Routing Engine
Coordinates difficulty classification, strategy selection, model execution,
cascade quality validation, and execution telemetry collection.
"""

import time
from typing import Dict, Any, Optional

from app.core.config import load_config
from app.core.logger import logger
from app.classifier.predict_difficulty import DifficultyPredictor
from app.models.model_manager import ModelManager
from app.models.base import ModelResponse
from app.routing.strategies import RoutingStrategyRegistry
from app.routing.escalation import EscalationChecker

class LLMRouter:
    """Central router directing user queries to the appropriate LLM."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        model_manager: Optional[ModelManager] = None,
        difficulty_predictor: Optional[DifficultyPredictor] = None
    ):
        self.config = load_config(config_path)
        self.model_manager = model_manager or ModelManager(config_path)
        self.predictor = difficulty_predictor or DifficultyPredictor()
        
        routing_cfg = self.config.get("routing", {})
        self.default_strategy = routing_cfg.get("default_strategy", "adaptive")
        self.conf_threshold = routing_cfg.get("confidence_threshold", 0.75)
        self.allow_escalation = routing_cfg.get("allow_escalation", True)
        
        esc_cfg = self.config.get("escalation", {})
        self.escalation_checker = EscalationChecker(
            min_length_chars=esc_cfg.get("min_response_length", 20),
            uncertainty_phrases=esc_cfg.get("uncertainty_phrases")
        )

    def route_and_generate(
        self,
        query: str,
        strategy: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes end-to-end adaptive routing and generation for a query.

        Returns a structured dictionary with decision details, telemetry, and response text.
        """
        start_total = time.perf_counter()
        active_strategy = (strategy or self.default_strategy).lower().strip()

        # Step 1: Query Difficulty Estimation
        prediction = self.predictor.predict(query)
        cat = prediction["predicted_difficulty"]
        score = prediction["difficulty_score"]
        conf = prediction["confidence"]
        predictor_name = prediction["predicted_by"]

        # Step 2: Routing Strategy Decision
        selected_tier = "small"
        routing_reason = ""

        if active_strategy in ["always_small", "small"]:
            selected_tier, routing_reason = RoutingStrategyRegistry.always_small(query, prediction)
        elif active_strategy in ["always_large", "large"]:
            selected_tier, routing_reason = RoutingStrategyRegistry.always_large(query, prediction)
        elif active_strategy in ["rule_based", "rule"]:
            selected_tier, routing_reason = RoutingStrategyRegistry.rule_based(query, prediction)
        elif active_strategy in ["confidence_based", "confidence"]:
            selected_tier, routing_reason = RoutingStrategyRegistry.confidence_based(
                query, prediction, conf_threshold=self.conf_threshold
            )
        elif active_strategy in ["cascade", "escalation"]:
            selected_tier, routing_reason = RoutingStrategyRegistry.cascade_initial(query, prediction)
        elif active_strategy in ["adaptive", "ml_adaptive", "ml"]:
            selected_tier, routing_reason = RoutingStrategyRegistry.ml_adaptive(query, prediction)
        else:
            logger.warning(f"Unknown routing strategy '{active_strategy}'. Using ml_adaptive.")
            selected_tier, routing_reason = RoutingStrategyRegistry.ml_adaptive(query, prediction)

        # Step 3: Primary Model Inference
        primary_response = self.model_manager.generate(
            tier=selected_tier,
            prompt=query,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        escalated = False
        escalation_reason = None
        final_response = primary_response
        initial_response_data = None

        # Step 4: Cascade / Escalation Quality Checks
        is_cascade_strategy = active_strategy in ["cascade", "escalation"]
        can_escalate = (is_cascade_strategy or self.allow_escalation) and (selected_tier == "small")

        if can_escalate:
            should_esc, esc_reason = self.escalation_checker.should_escalate(
                query=query,
                response=primary_response,
                predicted_difficulty=cat,
                difficulty_score=score
            )
            if should_esc:
                logger.info(f"Escalating query to large model. Reason: {esc_reason}")
                escalated = True
                escalation_reason = esc_reason
                initial_response_data = primary_response.to_dict()

                # Execute Large Model
                secondary_response = self.model_manager.generate(
                    tier="large",
                    prompt=query,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                final_response = secondary_response

        total_elapsed = time.perf_counter() - start_total

        # Compute composite telemetry
        composite_latency = (
            primary_response.total_latency + (final_response.total_latency if escalated else 0.0)
        )
        total_tokens = primary_response.output_tokens + (final_response.output_tokens if escalated else 0.0)

        result = {
            "query": query,
            "difficulty": cat,
            "difficulty_score": score,
            "confidence": conf,
            "classifier_used": predictor_name,
            "routing_strategy": active_strategy,
            "routing_reason": routing_reason,
            "selected_model": final_response.model_name,
            "selected_tier": final_response.model_tier,
            "response": final_response.response_text,
            "escalated": escalated,
            "escalation_reason": escalation_reason,
            "total_latency": round(composite_latency, 4),
            "input_tokens": final_response.input_tokens,
            "output_tokens": total_tokens,
            "tokens_per_second": final_response.tokens_per_second,
            "cpu_utilization_pct": final_response.cpu_utilization_pct,
            "ram_used_mb": final_response.ram_used_mb,
            "success": final_response.success,
            "error_message": final_response.error_message,
            "initial_attempt": initial_response_data
        }
        return result
