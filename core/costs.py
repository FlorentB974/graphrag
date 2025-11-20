"""Utility helpers for estimating LLM token costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelPricing:
    """Pricing information for a single model."""

    input_per_1k: float
    output_per_1k: float

    def estimate_cost(self, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Dict[str, float]:
        """Estimate USD cost for the provided token counts."""
        prompt = prompt_tokens or 0
        completion = completion_tokens or 0
        input_cost = (prompt / 1000.0) * self.input_per_1k
        output_cost = (completion / 1000.0) * self.output_per_1k
        total = input_cost + output_cost
        return {
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total, 6),
        }


MODEL_PRICING: Dict[str, ModelPricing] = {
    # 2024 OpenAI pricing (per 1K tokens)
    "gpt-4o": ModelPricing(input_per_1k=0.005, output_per_1k=0.015),
    "gpt-4o-mini": ModelPricing(input_per_1k=0.00015, output_per_1k=0.0006),
    "gpt-4-turbo": ModelPricing(input_per_1k=0.01, output_per_1k=0.03),
    "gpt-4": ModelPricing(input_per_1k=0.03, output_per_1k=0.06),
    "gpt-3.5-turbo-16k": ModelPricing(input_per_1k=0.003, output_per_1k=0.004),
    "gpt-3.5-turbo": ModelPricing(input_per_1k=0.0015, output_per_1k=0.002),
    # Default fallback roughly aligned with GPT-3.5 costs
    "default": ModelPricing(input_per_1k=0.0015, output_per_1k=0.002),
}


def get_pricing_for_model(model_name: Optional[str]) -> ModelPricing:
    """Return the closest pricing tier for the given model name."""
    if not model_name:
        return MODEL_PRICING["default"]

    normalized = model_name.lower()

    for key, pricing in MODEL_PRICING.items():
        if key == "default":
            continue
        if key in normalized:
            return pricing

    return MODEL_PRICING["default"]


def estimate_cost(model_name: Optional[str], prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Dict[str, float]:
    """Estimate USD cost for the provided token counts based on the configured model name."""
    pricing = get_pricing_for_model(model_name)
    return pricing.estimate_cost(prompt_tokens, completion_tokens)
