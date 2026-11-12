"""
Central Configuration Loader for Adaptive LLM Routing
"""

import os
import yaml
from typing import Dict, Any

DEFAULT_CONFIG_PATH = "config.yaml"

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and parse YAML configuration file with environment variable overrides."""
    if not os.path.exists(config_path):
        # Fallback default configuration
        return {
            "models": {
                "small": {"provider": "ollama", "name": "qwen2.5:3b", "fallback_offline": True},
                "large": {"provider": "ollama", "name": "qwen2.5:7b", "fallback_offline": True}
            },
            "routing": {"default_strategy": "adaptive", "confidence_threshold": 0.75},
            "evaluation": {"temperature": 0.0, "max_tokens": 512, "timeout": 180},
            "logging": {"level": "INFO"}
        }

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Environment variable overrides
    if "OLLAMA_HOST" in os.environ:
        cfg.setdefault("models", {})
        cfg["models"]["ollama_host"] = os.environ["OLLAMA_HOST"]

    return cfg
