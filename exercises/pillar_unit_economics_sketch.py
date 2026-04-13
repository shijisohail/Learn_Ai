#!/usr/bin/env python3
"""
Companion to: next/ai_product_building.html
Rough monthly LLM API cost from requests and token assumptions (no dependencies).
Edit PRICE_PER_1K_COMBINED to match your provider blended in+out pricing.
"""

def monthly_llm_cost(
    requests_per_month: int,
    avg_prompt_tokens: int,
    avg_completion_tokens: int,
    price_per_1k_tokens: float,
) -> float:
    """Return estimated monthly $ for generation calls only (excludes vector DB, etc.)."""
    tokens_per_request = avg_prompt_tokens + avg_completion_tokens
    return requests_per_month * (tokens_per_request / 1000.0) * price_per_1k_tokens


if __name__ == "__main__":
    # Example: 500k requests, 800 in + 400 out, $3 per 1M tokens → use 0.003 per 1K
    PRICE_PER_1K_COMBINED = 0.003
    usd = monthly_llm_cost(
        requests_per_month=500_000,
        avg_prompt_tokens=800,
        avg_completion_tokens=400,
        price_per_1k_tokens=PRICE_PER_1K_COMBINED,
    )
    print(f"Estimated monthly LLM API spend: ${usd:,.2f}")
    print("Adjust token averages by route (e.g. Haiku inner loop vs Sonnet outer).")
