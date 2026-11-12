"""
Base Model Classes and Data Structures for LLM Inference
Defines the standard ModelResponse payload and BaseLLMClient interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

@dataclass
class ModelResponse:
    """Standardized response data structure returned by all LLM clients."""
    query: str
    response_text: str
    model_name: str
    model_tier: str  # "small" or "large"
    provider: str    # "ollama", "api", "offline_research"
    
    # Latency Telemetry (in seconds)
    total_latency: float = 0.0
    loading_time: float = 0.0
    inference_time: float = 0.0
    time_to_first_token: Optional[float] = None
    
    # Token Metrics
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    cold_start: bool = False
    
    # Resource Telemetry
    cpu_utilization_pct: Optional[float] = None
    ram_used_mb: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    vram_used_mb: Optional[float] = None
    
    # Execution Status
    success: bool = True
    error_message: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BaseLLMClient(ABC):
    """Abstract Base Class for all LLM inference providers."""

    def __init__(self, model_name: str, model_tier: str, provider: str = "base"):
        self.model_name = model_name
        self.model_tier = model_tier
        self.provider = provider
        self.is_warmed_up = False

    @abstractmethod
    def is_available(self) -> bool:
        """Verify whether the model backend is accessible and functional."""
        pass

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512, **kwargs) -> ModelResponse:
        """Generate response text and collect timing and resource telemetry."""
        pass

    def warm_up(self) -> bool:
        """Executes a small warm-up query to separate cold start from warm inference."""
        try:
            res = self.generate("Hello", temperature=0.0, max_tokens=10)
            self.is_warmed_up = res.success
            return self.is_warmed_up
        except Exception:
            return False
