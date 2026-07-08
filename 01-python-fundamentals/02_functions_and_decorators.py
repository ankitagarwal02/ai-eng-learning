"""
02_functions_and_decorators.py — Functions, closures, decorators for AI Engineers
=================================================================================

WHAT THIS FILE TEACHES
----------------------
• Function definitions with default args, *args, **kwargs.
• Closures — a function that "remembers" its enclosing scope (rate limiter demo).
• Decorators — the middleware pattern:  @timer, @retry, @cache.
• `functools.wraps` and `functools.lru_cache`.
• Lambda functions and higher-order functions (map/filter/reduce).

HOW TO RUN
----------
    python 02_functions_and_decorators.py

REAL-WORLD SCENARIO
-------------------
Build a @retry decorator that wraps a flaky LLM API call and retries with exponential
backoff. Then combine @timer + @retry + @cache to build a production-grade wrapper
that measures, retries, and remembers.
"""

import os
import time
import random
import functools
from functools import reduce
from typing import Callable, Any

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Functions — defaults, *args, **kwargs
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Functions — defaults, *args, **kwargs")
print("=" * 70)

def chat(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    """A tiny LLM wrapper with defaults."""
    return f"[{model} @ T={temperature}] echo: {prompt}"

print(chat("Summarize this ticket"))
print(chat("Explain RAG", model="gpt-4o"))
print(chat("Be creative", temperature=1.2))

# *args and **kwargs — passing through unknown arguments:
def call_provider(provider: str, /, *args, **kwargs) -> str:
    """/ means everything before it is positional-only.
    *args collects extra positional args; **kwargs collects extra keyword args."""
    return f"provider={provider}, args={args}, kwargs={kwargs}"

print(call_provider("openai", "chat", "completions", model="gpt-4o", stream=True))


# ─────────────────────────────────────────────────────────────
# SECTION 2: Closures — functions that remember scope
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: Closures — a rate limiter that remembers state")
print("=" * 70)

def make_rate_limiter(max_per_second: int):
    """Return a rate-check function that remembers its own call log across invocations.
    This is a CLOSURE — the returned function keeps `calls` and `max_per_second` alive."""
    calls: list[float] = []

    def allow() -> bool:
        now = time.time()
        # Drop calls older than 1s:
        nonlocal calls
        calls = [t for t in calls if now - t < 1.0]
        if len(calls) >= max_per_second:
            return False
        calls.append(now)
        return True

    return allow

allow_3_per_sec = make_rate_limiter(3)
for i in range(5):
    ok = allow_3_per_sec()
    print(f"  Call {i+1}: {'✅ allowed' if ok else '❌ throttled'}")
    time.sleep(0.05)


# ─────────────────────────────────────────────────────────────
# SECTION 3: Decorators — the middleware pattern
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: Decorators — @timer")
print("=" * 70)

def timer(func: Callable) -> Callable:
    """Simplest useful decorator — logs how long a function took."""
    @functools.wraps(func)           # preserves __name__, __doc__, etc.
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  ⏱  {func.__name__} took {elapsed:.1f} ms")
        return result
    return wrapper

@timer
def slow_operation(n: int) -> int:
    """Simulate a slow LLM call."""
    time.sleep(0.1)
    return n * 2

print(f"Result: {slow_operation(21)}")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Parameterized decorators — @retry(max_attempts=3)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: @retry with exponential backoff")
print("=" * 70)

def retry(max_attempts: int = 3, backoff: float = 0.1, exceptions: tuple = (Exception,)):
    """Decorator factory. Retries on failure with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = backoff
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        print(f"  ✗ {func.__name__} exhausted {max_attempts} attempts")
                        raise
                    print(f"  ↻ {func.__name__} attempt {attempt} failed ({e}) — retry in {delay:.2f}s")
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
        return wrapper
    return decorator

# Simulate an LLM call that fails the first 2 attempts:
_call_count = 0

@retry(max_attempts=4, backoff=0.05)
def flaky_llm_call(prompt: str) -> str:
    global _call_count
    _call_count += 1
    if _call_count < 3:
        raise ConnectionError(f"transient network error (call #{_call_count})")
    return f"[Success on attempt {_call_count}] response to: {prompt}"

print(flaky_llm_call("What is RAG?"))


# ─────────────────────────────────────────────────────────────
# SECTION 5: functools.lru_cache — automatic memoization
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: functools.lru_cache — free memoization")
print("=" * 70)

@functools.lru_cache(maxsize=128)
def embed(text: str) -> tuple:
    """Simulate an expensive embedding call. Returns a tuple so the result is hashable."""
    print(f"  [computing embedding for: {text!r}]")
    time.sleep(0.05)
    random.seed(hash(text))
    return tuple(random.random() for _ in range(4))

# First call: computes and prints.
v1 = embed("hello world")
# Second call with same input: cache hit — no print.
v2 = embed("hello world")
# Different input: computes again.
v3 = embed("goodbye world")

print(f"v1 == v2 (cache hit)?  {v1 == v2}")
print(f"v1 == v3?              {v1 == v3}")
print(f"Cache stats:           {embed.cache_info()}")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Stacked decorators — @timer + @retry + @lru_cache
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: Stacked decorators — production wrapper")
print("=" * 70)

@timer
@retry(max_attempts=3, backoff=0.05)
@functools.lru_cache(maxsize=64)
def production_llm_call(prompt: str) -> str:
    """Combines timing, retries, and caching. Order matters — bottom decorator is
    applied first, so lru_cache wraps the raw function, then retry wraps that,
    then timer wraps the whole thing."""
    return f"[MOCK] processed: {prompt}"

print(production_llm_call("What is a decorator?"))
print(production_llm_call("What is a decorator?"))   # cache hit — should be FAST
print(production_llm_call("What is a closure?"))     # miss


# ─────────────────────────────────────────────────────────────
# SECTION 7: Lambdas + higher-order functions
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7: Lambdas, map, filter, reduce")
print("=" * 70)

token_counts = [1200, 3400, 800, 12000, 450, 6700]

# map — apply function to every element:
costs = list(map(lambda t: t * 0.15 / 1000, token_counts))   # $/token approx
print(f"Costs per call: {[round(c, 3) for c in costs]}")

# filter — keep matching elements:
big_calls = list(filter(lambda t: t > 5000, token_counts))
print(f"Calls over 5k tokens: {big_calls}")

# reduce — collapse to a single value:
total_tokens = reduce(lambda a, b: a + b, token_counts)
print(f"Total tokens: {total_tokens}")

# In practice, comprehensions are more readable than map/filter:
costs_readable = [t * 0.15 / 1000 for t in token_counts]
big_readable = [t for t in token_counts if t > 5000]
print(f"Readable version — same result? {costs == costs_readable and big_calls == big_readable}")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
You built the toolbox that will appear in every production AI codebase:

  • Functions with sensible defaults and *args/**kwargs pass-through
  • Closures for stateful helpers (rate limiter)
  • @timer, @retry, @lru_cache — the three decorators you'll actually use
  • Parameterized decorators via decorator factories
  • Stacking decorators — order matters!
  • Lambdas + map/filter/reduce (and why comprehensions usually win)

Every LLM call in production should be wrapped like:
    @timer @retry(max_attempts=3) @lru_cache(maxsize=1024) def call(...): ...

Next file: 03_classes_and_oop.py — Building the AIAssistant base + subclasses.
""")
