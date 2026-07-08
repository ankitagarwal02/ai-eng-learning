"""
04_async_programming.py — Async I/O for parallel LLM calls
==========================================================

WHAT THIS FILE TEACHES
----------------------
• `async def` and `await` — the two syntactic anchors.
• `asyncio.gather()` — run N coroutines in parallel, wait for all.
• `asyncio.as_completed()` — process results as soon as each finishes.
• `asyncio.sleep()` — simulate latency without blocking.
• Why async matters: serial vs parallel LLM calls, with real timings.

HOW TO RUN
----------
    python 04_async_programming.py

REAL-WORLD SCENARIOS
--------------------
1. Query 3 LLM providers (GPT-4o, Claude, Gemini) SERIALLY vs in PARALLEL.
   Show the wall-clock difference.
2. Fetch data from 5 URLs concurrently, process each result as soon as it arrives.

Why this matters:
    Serial 3× 500ms  =  1500ms  (visible lag)
    Parallel via gather  ≈  500ms  (feels instant)
The whole difference is one function call: asyncio.gather.
"""

import os
import asyncio
import random
import time
from typing import Any

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Sync vs async — the mental model
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Sync vs async")
print("=" * 70)
print("""
Sync (blocking):
    call GPT-4o    ██████        (500ms — CPU idle waiting on network)
    call Claude          ██████  (500ms)
    call Gemini                ██████
    Total: 1500ms

Async (non-blocking):
    call GPT-4o    ██████
    call Claude    ██████   (all three started immediately;
    call Gemini    ██████    Python moves on while each waits on network)
    Total: ~500ms (slowest single call)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 2: A mock async LLM call
# ─────────────────────────────────────────────────────────────
async def mock_llm_call(provider: str, prompt: str) -> dict:
    """Simulate an LLM call. Latency is random 300-700ms so we can see parallelism."""
    latency = random.uniform(0.3, 0.7)
    await asyncio.sleep(latency)   # non-blocking sleep — key to concurrency
    return {
        "provider": provider,
        "prompt": prompt,
        "response": f"[MOCK {provider}] {prompt[:30]}...",
        "latency_ms": round(latency * 1000, 1),
    }


# ─────────────────────────────────────────────────────────────
# SECTION 3: Serial (one after another) — the slow way
# ─────────────────────────────────────────────────────────────
async def serial_multi_provider(prompt: str) -> list[dict]:
    """Call three providers ONE AT A TIME."""
    print("\n  [serial] starting...")
    start = time.perf_counter()
    results = []
    for provider in ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"]:
        results.append(await mock_llm_call(provider, prompt))
    total = (time.perf_counter() - start) * 1000
    print(f"  [serial] total wall-clock: {total:.1f} ms")
    return results


# ─────────────────────────────────────────────────────────────
# SECTION 4: Parallel with asyncio.gather() — the fast way
# ─────────────────────────────────────────────────────────────
async def parallel_multi_provider(prompt: str) -> list[dict]:
    """Fire all three requests SIMULTANEOUSLY, wait for all to finish."""
    print("\n  [parallel] starting...")
    start = time.perf_counter()
    results = await asyncio.gather(
        mock_llm_call("gpt-4o", prompt),
        mock_llm_call("claude-3-5-sonnet", prompt),
        mock_llm_call("gemini-1.5-pro", prompt),
    )
    total = (time.perf_counter() - start) * 1000
    print(f"  [parallel] total wall-clock: {total:.1f} ms")
    return results


# ─────────────────────────────────────────────────────────────
# SECTION 5: Process results as they finish — asyncio.as_completed
# ─────────────────────────────────────────────────────────────
async def stream_results_as_they_arrive(prompt: str) -> None:
    """Sometimes you want to display each result the moment it's ready
    (e.g. update a UI progressively). Use as_completed for this."""
    print("\n  [as_completed] streaming individual results...")
    coros = [mock_llm_call(p, prompt) for p in ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"]]

    for coro in asyncio.as_completed(coros):
        result = await coro
        print(f"    ← {result['provider']} finished in {result['latency_ms']} ms")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Handling failures — return_exceptions=True
# ─────────────────────────────────────────────────────────────
async def flaky_provider(name: str) -> dict:
    """A provider that sometimes fails."""
    await asyncio.sleep(0.2)
    if name == "flaky-2":
        raise RuntimeError(f"{name} rate-limited")
    return {"provider": name, "ok": True}

async def parallel_with_failures() -> None:
    """Real world: if one provider fails, still collect the others."""
    print("\n  [robust gather] one provider fails; others complete...")
    results = await asyncio.gather(
        flaky_provider("flaky-1"),
        flaky_provider("flaky-2"),   # will raise
        flaky_provider("flaky-3"),
        return_exceptions=True,      # exceptions become return values
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"    ✗ error: {type(r).__name__}: {r}")
        else:
            print(f"    ✓ {r}")


# ─────────────────────────────────────────────────────────────
# SECTION 7: Concurrency limits — semaphore for rate-limited APIs
# ─────────────────────────────────────────────────────────────
async def bounded_batch_processing(items: list[str], concurrency: int = 3) -> list[dict]:
    """Process 10 items but only run 3 at a time — respects provider rate limits.
    This is the pattern for embedding a batch of docs, translating a corpus, etc."""
    print(f"\n  [semaphore] {len(items)} items, max {concurrency} in flight...")
    sem = asyncio.Semaphore(concurrency)

    async def bounded(item):
        async with sem:                       # blocks if 3 already in flight
            return await mock_llm_call("gpt-4o-mini", item)

    return await asyncio.gather(*[bounded(i) for i in items])


# ─────────────────────────────────────────────────────────────
# SECTION 8: async with / async for — context managers & iteration
# ─────────────────────────────────────────────────────────────
class MockStreamingClient:
    """A tiny mock that behaves like httpx.AsyncClient / an LLM streaming call."""

    async def __aenter__(self):
        print("  [async ctx] opened connection")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        print("  [async ctx] closed connection")

    async def stream_tokens(self, prompt: str):
        """Yield tokens one at a time — simulates streaming LLM output."""
        for tok in ["The ", "quick ", "brown ", "fox."]:
            await asyncio.sleep(0.05)
            yield tok


async def streaming_demo():
    print("\n  [streaming] token-by-token output...")
    async with MockStreamingClient() as client:
        chunks = []
        async for tok in client.stream_tokens("Tell me a story"):
            print(f"    tok: {tok!r}")
            chunks.append(tok)
        print(f"    joined: {''.join(chunks)}")


# ─────────────────────────────────────────────────────────────
# MAIN — run everything
# ─────────────────────────────────────────────────────────────
async def main():
    prompt = "Summarize the current state of LLMs in one line."

    print("\n" + "=" * 70)
    print("SECTION 3+4: Serial vs Parallel benchmark")
    print("=" * 70)

    serial_results = await serial_multi_provider(prompt)
    for r in serial_results:
        print(f"    {r['provider']:<20} ({r['latency_ms']} ms)")

    parallel_results = await parallel_multi_provider(prompt)
    for r in parallel_results:
        print(f"    {r['provider']:<20} ({r['latency_ms']} ms)")

    print("\n" + "=" * 70)
    print("SECTION 5: asyncio.as_completed()")
    print("=" * 70)
    await stream_results_as_they_arrive(prompt)

    print("\n" + "=" * 70)
    print("SECTION 6: gather with return_exceptions")
    print("=" * 70)
    await parallel_with_failures()

    print("\n" + "=" * 70)
    print("SECTION 7: Bounded concurrency (semaphore)")
    print("=" * 70)
    items = [f"doc-{i}" for i in range(6)]
    bounded_results = await bounded_batch_processing(items, concurrency=2)
    print(f"    completed {len(bounded_results)} items")

    print("\n" + "=" * 70)
    print("SECTION 8: async with / async for (streaming)")
    print("=" * 70)
    await streaming_demo()


if __name__ == "__main__":
    asyncio.run(main())

    print("\n" + "=" * 70)
    print("✅ Summary")
    print("=" * 70)
    print("""
You now have the async toolkit that separates junior AI code from production:

  • asyncio.gather()          → parallel N-way fan-out (single result set)
  • asyncio.as_completed()    → progressive results (UI streaming, early exits)
  • return_exceptions=True    → don't crash the whole batch on one failure
  • asyncio.Semaphore         → respect provider rate limits (max in-flight)
  • async with / async for    → streaming clients and iterators

Rule of thumb: EVERY external call in an AI system (LLM, embedding, vector DB,
web scrape, tool invocation) should be `async def`. The moment you have >1
such call to make, you save latency for free with `gather`.

Next file: 05_pydantic_and_typing.py — Structured output the sane way.
""")
