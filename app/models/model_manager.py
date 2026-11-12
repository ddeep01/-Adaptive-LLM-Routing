"""
Model Manager Module
Centralizes model lifecycle, provider resolution (Ollama vs. Offline Fallback vs. API),
sequential memory loading, and status checking.
"""

import os
from typing import Dict, Any, Optional

from app.core.config import load_config
from app.core.logger import logger
from app.core.exceptions import ModelUnavailableException
from app.models.base import BaseLLMClient, ModelResponse
from app.models.ollama_client import OllamaClient
from app.models.offline_client import OfflineResearchClient

class ModelManager:
    """Manages Small and Large LLM instances, handling provider resolution and hardware safety."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.small_model: Optional[BaseLLMClient] = None
        self.large_model: Optional[BaseLLMClient] = None
        self._init_models()

    def _init_models(self):
        models_cfg = self.config.get("models", {})
        
        small_cfg = models_cfg.get("small", {})
        large_cfg = models_cfg.get("large", {})

        self.small_model = self._create_client(
            tier="small",
            cfg=small_cfg,
            default_name="qwen2.5:3b"
        )
        self.large_model = self._create_client(
            tier="large",
            cfg=large_cfg,
            default_name="qwen2.5:7b"
        )

    def _create_client(self, tier: str, cfg: Dict[str, Any], default_name: str) -> BaseLLMClient:
        provider = cfg.get("provider", "ollama").lower()
        model_name = cfg.get("name", default_name)
        allow_fallback = cfg.get("fallback_offline", True)

        if provider == "ollama":
            ollama_host = self.config.get("models", {}).get("ollama_host", "http://127.0.0.1:11434")
            client = OllamaClient(
                model_name=model_name,
                model_tier=tier,
                base_url=ollama_host,
                timeout=self.config.get("evaluation", {}).get("timeout", 180.0)
            )
            if client.is_available():
                logger.info(f"Initialized live Ollama model for tier '{tier}': {model_name}")
                return client
            elif allow_fallback:
                logger.warning(
                    f"Ollama daemon not accessible or model '{model_name}' not pulled. "
                    f"Engaging calibrated high-fidelity OfflineResearchClient for tier '{tier}'."
                )
                return OfflineResearchClient(model_name=model_name, model_tier=tier)
            else:
                raise ModelUnavailableException(
                    f"Ollama model '{model_name}' for tier '{tier}' is unavailable and fallback is disabled."
                )

        elif provider in ["offline_research", "offline", "mock"]:
            logger.info(f"Using OfflineResearchClient for tier '{tier}': {model_name}")
            return OfflineResearchClient(model_name=model_name, model_tier=tier)

        else:
            logger.warning(f"Unknown provider '{provider}'. Defaulting to OfflineResearchClient.")
            return OfflineResearchClient(model_name=model_name, model_tier=tier)

    def get_model(self, tier: str) -> BaseLLMClient:
        """Retrieve client by tier ('small' or 'large')."""
        tier_clean = tier.lower().strip()
        if tier_clean == "small":
            return self.small_model
        elif tier_clean == "large":
            return self.large_model
        else:
            raise ValueError(f"Unknown model tier '{tier}'. Must be 'small' or 'large'.")

    def get_status(self) -> Dict[str, Any]:
        """Returns health and provider status for configured models."""
        return {
            "small_model": {
                "name": self.small_model.model_name,
                "tier": self.small_model.model_tier,
                "provider": self.small_model.provider,
                "available": self.small_model.is_available(),
                "warmed_up": self.small_model.is_warmed_up
            },
            "large_model": {
                "name": self.large_model.model_name,
                "tier": self.large_model.model_tier,
                "provider": self.large_model.provider,
                "available": self.large_model.is_available(),
                "warmed_up": self.large_model.is_warmed_up
            }
        }

    def generate(self, tier: str, prompt: str, **kwargs) -> ModelResponse:
        """Route generation request to the specified model tier."""
        client = self.get_model(tier)
        return client.generate(prompt, **kwargs)
