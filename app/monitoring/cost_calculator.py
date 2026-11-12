"""
Cost Calculator for Local Compute Energy and API Token Accounting
Calculates dollar costs, energy consumption, and comparative efficiency metrics.
"""

from typing import Dict, Any, Optional

DEFAULT_RATES = {
    "small": {
        "input_cost_per_1k": 0.0001,
        "output_cost_per_1k": 0.0002,
        "watts_estimated": 18.0
    },
    "large": {
        "input_cost_per_1k": 0.0008,
        "output_cost_per_1k": 0.0016,
        "watts_estimated": 45.0
    }
}

ELECTRICITY_COST_PER_KWH = 0.14  # Standard US commercial/residential average ($/kWh)

class CostCalculator:
    """Calculates financial API costs, local hardware kilowatt-hour energy, and comparative savings."""

    def __init__(self, model_rates: Optional[Dict[str, Any]] = None, cost_per_kwh: float = ELECTRICITY_COST_PER_KWH):
        self.rates = model_rates or DEFAULT_RATES
        self.cost_per_kwh = cost_per_kwh

    def compute_query_cost(
        self,
        model_tier: str,
        input_tokens: int,
        output_tokens: int,
        latency_seconds: float
    ) -> Dict[str, float]:
        """Calculates token API financial cost and local hardware kilowatt-hour energy cost."""
        tier = model_tier.lower()
        cfg = self.rates.get(tier, self.rates["small"])

        in_rate = cfg.get("input_cost_per_1k", 0.0001)
        out_rate = cfg.get("output_cost_per_1k", 0.0002)
        watts = cfg.get("watts_estimated", 25.0)

        # API token pricing model ($)
        api_cost = (input_tokens / 1000.0 * in_rate) + (output_tokens / 1000.0 * out_rate)

        # Local hardware energy model (kWh and $)
        hours = latency_seconds / 3600.0
        energy_kwh = hours * (watts / 1000.0)
        energy_cost = energy_kwh * self.cost_per_kwh

        return {
            "api_token_cost_usd": round(api_cost, 7),
            "energy_kwh": round(energy_kwh, 7),
            "energy_cost_usd": round(energy_cost, 7),
            "total_estimated_cost_usd": round(api_cost + energy_cost, 7)
        }

    @staticmethod
    def compute_comparative_savings(
        latency_large: float,
        latency_adaptive: float,
        cost_large: float,
        cost_adaptive: float,
        quality_large: float,
        quality_adaptive: float
    ) -> Dict[str, float]:
        """
        Calculates:
        - Latency Savings (%) = (T_large - T_adaptive) / T_large * 100
        - Cost Savings (%) = (C_large - C_adaptive) / C_large * 100
        - Quality Retention (%) = (Q_adaptive / Q_large) * 100
        """
        latency_savings_pct = (
            ((latency_large - latency_adaptive) / latency_large * 100.0)
            if latency_large > 0 else 0.0
        )
        cost_savings_pct = (
            ((cost_large - cost_adaptive) / cost_large * 100.0)
            if cost_large > 0 else 0.0
        )
        quality_retention_pct = (
            (quality_adaptive / quality_large * 100.0)
            if quality_large > 0 else 100.0
        )

        return {
            "latency_savings_pct": round(latency_savings_pct, 2),
            "cost_savings_pct": round(cost_savings_pct, 2),
            "quality_retention_pct": round(quality_retention_pct, 2)
        }
