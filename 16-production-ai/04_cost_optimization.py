"""
04_cost_optimization.py — The 5 levers that cut AI costs 90-95%
==================================================================

1. Token counting before every call (tiktoken)
2. Model routing (mini for simple, big for complex)
3. Prompt compression (remove filler words)
4. Response caching (identical or near-identical queries)
5. Batch API for bulk async work (50% discount)

SCENARIO
--------
10,000 support tickets/day. Cost calculation showing the difference between
naive ("all gpt-4o") and optimized ("mini + routing + cache + batch").
"""

import os
from dataclasses import dataclass

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


PRICES = {
    "gpt-4o-mini":   {"input": 0.15,  "output": 0.60,  "batch_discount": 0.5},
    "gpt-4o":        {"input": 2.50,  "output": 10.00, "batch_discount": 0.5},
    "gpt-4o-mini-cached": {"input": 0.075, "output": 0.60, "batch_discount": 0.5}, # 50% off cached input
}


@dataclass
class Volume:
    calls_per_day: int
    avg_input_tokens: int
    avg_output_tokens: int
    fraction_needing_big_model: float = 0.2   # only 20% actually need gpt-4o
    cache_hit_rate: float = 0.3               # 30% of calls are dupes/near-dupes
    batch_percent: float = 0.7                # 70% can run async via Batch API


def cost_per_day(vol: Volume, model: str, batch: bool = False, cached: bool = False) -> float:
    p = PRICES[model]
    input_price = p["input"] / 2 if cached else p["input"]
    output_price = p["output"]
    if batch:
        input_price *= p["batch_discount"]
        output_price *= p["batch_discount"]
    in_cost = vol.calls_per_day * vol.avg_input_tokens * input_price / 1_000_000
    out_cost = vol.calls_per_day * vol.avg_output_tokens * output_price / 1_000_000
    return in_cost + out_cost


def strategy_naive(vol: Volume) -> float:
    """All-gpt-4o, no optimizations."""
    return cost_per_day(vol, "gpt-4o", batch=False, cached=False)


def strategy_optimized(vol: Volume) -> float:
    """gpt-4o-mini for 80%, gpt-4o for 20%, + caching + batching."""
    total = 0.0
    v = Volume(**{**vars(vol)})

    # Fraction using mini
    v.calls_per_day = int(vol.calls_per_day * (1 - vol.fraction_needing_big_model))
    # Cache-hit portion is cheaper
    cached_calls = int(v.calls_per_day * vol.cache_hit_rate)
    fresh_calls = v.calls_per_day - cached_calls

    # Non-cached, batched portion
    batched_fresh = int(fresh_calls * vol.batch_percent)
    live_fresh = fresh_calls - batched_fresh
    vb = Volume(**{**vars(v), "calls_per_day": batched_fresh})
    vl = Volume(**{**vars(v), "calls_per_day": live_fresh})
    vc = Volume(**{**vars(v), "calls_per_day": cached_calls})

    total += cost_per_day(vb, "gpt-4o-mini", batch=True)
    total += cost_per_day(vl, "gpt-4o-mini", batch=False)
    total += cost_per_day(vc, "gpt-4o-mini-cached", batch=False)

    # 20% needing gpt-4o — assume no batching, but caching
    big = Volume(**{**vars(vol)})
    big.calls_per_day = int(vol.calls_per_day * vol.fraction_needing_big_model)
    big_cached = int(big.calls_per_day * vol.cache_hit_rate)
    big_fresh = big.calls_per_day - big_cached
    total += cost_per_day(
        Volume(**{**vars(big), "calls_per_day": big_cached}),
        "gpt-4o", batch=False, cached=True,
    )
    total += cost_per_day(
        Volume(**{**vars(big), "calls_per_day": big_fresh}),
        "gpt-4o", batch=False,
    )
    return total


if __name__ == "__main__":
    vol = Volume(calls_per_day=10_000, avg_input_tokens=800, avg_output_tokens=200)

    naive = strategy_naive(vol)
    opt = strategy_optimized(vol)

    print(f"Traffic: {vol.calls_per_day:,} calls/day, "
          f"{vol.avg_input_tokens}+{vol.avg_output_tokens} tok each")
    print()
    print(f"{'Strategy':<30}{'$/day':>10}{'$/month':>12}{'$/year':>14}")
    print("─" * 66)
    for name, cost in [("Naive (all gpt-4o)", naive), ("Optimized", opt)]:
        print(f"{name:<30}{cost:>10.2f}{cost*30:>12.2f}{cost*365:>14.2f}")

    savings_pct = (1 - opt / naive) * 100
    print(f"\nSavings: {savings_pct:.1f}% — ${(naive - opt) * 365:,.0f} per year")

    print("""

✅ The 5 cost levers in one place:

  1. Token count BEFORE every call → catch runaway prompts (Phase 2)
  2. Model routing — mini for 80% of tasks, big for the rest
  3. Prompt caching — repeated system prompts get 50% off input
  4. Response caching — Redis on the query hash (10-30% hit rate typical)
  5. Batch API — 50% off for async work you don't need <1s response

The single biggest win is model routing. Set it up first. Evaluate on your
golden set (Phase 14) to prove mini is good enough for that traffic slice.
""")
