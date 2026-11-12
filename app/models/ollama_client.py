"""
Ollama Local Inference Client
Interacts with local Ollama daemon via REST API (/api/generate and /api/tags).
"""

import time
import psutil
import httpx
from typing import Dict, Any, Optional

from app.models.base import BaseLLMClient, ModelResponse
from app.core.logger import logger

class OllamaClient(BaseLLMClient):
    """Client for local model execution using Ollama."""

    def __init__(
        self,
        model_name: str,
        model_tier: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 180.0
    ):
        super().__init__(model_name=model_name, model_tier=model_tier, provider="ollama")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama is running and has the target model pulled."""
        try:
            with httpx.Client(timeout=3.0) as client:
                r = client.get(f"{self.base_url}/api/tags")
                if r.status_code != 200:
                    return False
                models = [m.get("name", "") for m in r.json().get("models", [])]
                # Check for model match (e.g., qwen2.5:3b or qwen2.5:3b-instruct)
                for m in models:
                    if self.model_name in m or m.startswith(self.model_name.split(":")[0]):
                        return True
                return False
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs
    ) -> ModelResponse:
        """Execute inference against Ollama and record performance telemetry."""
        is_cold = not self.is_warmed_up
        start_time = time.perf_counter()
        
        cpu_before = psutil.cpu_percent(interval=None)
        ram_before = psutil.virtual_memory().used / (1024 * 1024)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()

            total_latency = time.perf_counter() - start_time
            self.is_warmed_up = True

            # Extract Ollama native high-precision timings (in nanoseconds)
            total_duration_ns = data.get("total_duration", 0)
            load_duration_ns = data.get("load_duration", 0)
            eval_duration_ns = data.get("eval_duration", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            eval_count = data.get("eval_count", 0)

            loading_time = load_duration_ns / 1e9 if load_duration_ns else 0.0
            inference_time = eval_duration_ns / 1e9 if eval_duration_ns else total_latency - loading_time
            tokens_per_sec = (eval_count / inference_time) if (inference_time > 0 and eval_count > 0) else 0.0

            cpu_after = psutil.cpu_percent(interval=None)
            ram_after = psutil.virtual_memory().used / (1024 * 1024)

            return ModelResponse(
                query=prompt,
                response_text=data.get("response", "").strip(),
                model_name=self.model_name,
                model_tier=self.model_tier,
                provider=self.provider,
                total_latency=round(total_latency, 4),
                loading_time=round(loading_time, 4),
                inference_time=round(inference_time, 4),
                input_tokens=prompt_eval_count,
                output_tokens=eval_count,
                tokens_per_second=round(tokens_per_sec, 2),
                cold_start=is_cold,
                cpu_utilization_pct=round((cpu_before + cpu_after) / 2.0, 1),
                ram_used_mb=round(max(ram_before, ram_after), 1),
                success=True,
                raw_metadata=data
            )
        except Exception as e:
            total_latency = time.perf_counter() - start_time
            logger.error(f"Ollama generation failed for {self.model_name}: {e}")
            return ModelResponse(
                query=prompt,
                response_text="",
                model_name=self.model_name,
                model_tier=self.model_tier,
                provider=self.provider,
                total_latency=round(total_latency, 4),
                cold_start=is_cold,
                success=False,
                error_message=str(e)
            )
