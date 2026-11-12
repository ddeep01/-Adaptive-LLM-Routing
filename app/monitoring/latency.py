"""
Latency Profiler and Statistical Aggregator
Computes Mean, Median (P50), P95 Latency, and Throughput Distributions.
"""

from typing import List, Dict, Any
import numpy as np

class LatencyTracker:
    """Calculates statistical percentiles and throughput metrics from latency observations."""

    @staticmethod
    def compute_summary(latencies: List[float], tokens: List[int] = None) -> Dict[str, float]:
        """Calculates latency summary statistics (in seconds)."""
        if not latencies:
            return {
                "count": 0, "mean": 0.0, "std": 0.0, "min": 0.0,
                "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0
            }

        arr = np.array(latencies)
        summary = {
            "count": int(len(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
            "max": round(float(np.max(arr)), 4)
        }

        if tokens and len(tokens) == len(latencies):
            total_tokens = sum(tokens)
            total_time = sum(latencies)
            summary["aggregate_tokens_per_second"] = round(
                total_tokens / total_time if total_time > 0 else 0.0, 2
            )

        return summary
