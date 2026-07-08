"""
05_memory_strategies_comparison.py — Side-by-side benchmark of 4 memory strategies
====================================================================================

WHAT THIS FILE TEACHES
----------------------
A rigorous head-to-head comparison of four memory strategies on the same
30-turn conversation. Measures:
  • Token count at every 5th turn
  • Estimated cost per turn (gpt-4o-mini pricing)
  • Recall of 6 planted facts at turn 30
  • Decision matrix: scenario → recommended strategy

Concludes with the PRODUCTION HYBRID recommendation.

HOW TO RUN
----------
    python 05_memory_strategies_comparison.py

REAL-WORLD SCENARIO
-------------------
A software engineer at FinBank asks 30 turns about MLOps for a fraud
detection project. We plant 6 facts at specific turns and ask: at turn 30,
can each strategy recall them?
"""

import os
from dataclasses import dataclass
from typing import Callable

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# Conversation with 6 planted facts we'll test recall on
# ─────────────────────────────────────────────────────────────
# Format: (role, content, planted_fact_key)
CONVERSATION = [
    ("system", "You are an AI engineering advisor.", None),
    # Turn 1
    ("user",     "Hi, I'm Alice, software engineer at FinBank.", "name-and-company"),
    ("assistant", "Nice to meet you, Alice. What are you working on?", None),
    # Turn 3
    ("user",     "Fraud detection using gradient boosting.", "model-type"),
    ("assistant", "Great choice. Baseline metrics?", None),
    # Turn 5
    ("user",     "Precision 92%, recall 85%.", "baseline-metrics"),
    ("assistant", "Solid baseline. Any bottlenecks?", None),
    ("user",     "Latency is my concern — production p95 is 900ms.", None),
    ("assistant", "Where's the latency coming from?", None),
    # Turn 9
    ("user",     "Mostly feature computation on transaction history.", None),
    ("assistant", "Consider precomputing features nightly.", None),
    # Turn 11
    ("user",     "Team of 3 engineers reports to me.", "team-size"),
    ("assistant", "Got it. And your MLOps stack?", None),
    ("user",     "MLflow for tracking, AWS SageMaker for serving.", None),
    ("assistant", "Standard stack. Deployment cadence?", None),
    # Turn 15
    ("user",     "Weekly deploys, Fridays.", "deploy-day"),
    ("assistant", "Fridays are risky — consider Mondays.", None),
    ("user",     "Point taken. What about model monitoring?", None),
    ("assistant", "Data drift detection is essential.", None),
    ("user",     "I use Evidently AI for that.", None),
    ("assistant", "Good tool. Alerting setup?", None),
    ("user",     "PagerDuty routed via CloudWatch.", None),
    ("assistant", "Standard. What LLM projects on your roadmap?", None),
    ("user",     "Transaction narrative summarization with gpt-4o-mini.", None),
    ("assistant", "Cost-effective. Volume?", None),
    # Turn 25
    ("user",     "5M transactions per day.", "volume"),
    ("assistant", "~$375/day for -mini. Batching cuts that.", None),
    ("user",     "OK, let's use Batch API.", None),
    ("assistant", "50% discount, 24h latency. Good fit for post-hoc summaries.", None),
    ("user",     "Perfect. Thanks for the guidance.", None),
    ("assistant", "Anytime, Alice. Good luck with the project.", None),
]

# The 6 planted facts we'll test recall on at turn 30:
PLANTED = [
    ("name-and-company", "user's name and company", ["alice", "finbank"]),
    ("model-type",       "the ML model in use",     ["gradient boosting"]),
    ("baseline-metrics", "the baseline metrics",    ["92", "85"]),
    ("team-size",        "the team size",           ["team of 3", "3 engineer"]),
    ("deploy-day",       "the deploy cadence",      ["friday", "weekly"]),
    ("volume",           "transaction volume",      ["5m transactions", "5 million"]),
]


# ─────────────────────────────────────────────────────────────
# Token counter
# ─────────────────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return max(1, int(len(text) / 4))


def context_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m["content"]) + 4 for m in messages)


# ─────────────────────────────────────────────────────────────
# Strategies — each takes full history and returns visible context
# ─────────────────────────────────────────────────────────────
def strategy_none(history: list[dict]) -> list[dict]:
    """No memory: just system + latest user message."""
    sys = [m for m in history if m["role"] == "system"]
    user = [m for m in history if m["role"] == "user"]
    return sys + user[-1:]


def strategy_full(history: list[dict]) -> list[dict]:
    """Full history — grows quadratically."""
    return list(history)


def strategy_sliding(history: list[dict], window: int = 6) -> list[dict]:
    sys = [m for m in history if m["role"] == "system"]
    rest = [m for m in history if m["role"] != "system"]
    return sys + rest[-window:]


def strategy_summary(history: list[dict], keep_recent: int = 6) -> list[dict]:
    sys = [m for m in history if m["role"] == "system"]
    rest = [m for m in history if m["role"] != "system"]
    if len(rest) <= keep_recent:
        return sys + rest
    old = rest[:-keep_recent]
    recent = rest[-keep_recent:]
    # Real: LLM summarizes. Mock: concatenate + truncate.
    combined = " | ".join(f"{m['role']}: {m['content']}" for m in old)
    summary = {"role": "system", "content": "[SUMMARY of prior turns] " + combined[:300] + "..."}
    return sys + [summary] + recent


STRATEGIES: dict[str, Callable] = {
    "no_memory":       strategy_none,
    "full_history":    strategy_full,
    "sliding_w=6":     lambda h: strategy_sliding(h, window=6),
    "summarization":   lambda h: strategy_summary(h, keep_recent=6),
}


# ─────────────────────────────────────────────────────────────
# Benchmark 1: Token growth
# ─────────────────────────────────────────────────────────────
def token_growth_benchmark():
    print("═" * 90)
    print("BENCHMARK 1: token growth at every 5th turn")
    print("═" * 90)

    convos_by_turn: dict[int, list[dict]] = {}
    running = []
    for role, content, _ in CONVERSATION:
        running.append({"role": role, "content": content})
        if role == "user":
            convos_by_turn[len(convos_by_turn) + 1] = list(running)   # after each user

    checkpoints = [1, 5, 10, 15, 20, 25, 30]
    header = f"{'Strategy':<20}" + "".join(f"T{cp:>3}   " for cp in checkpoints)
    print(header)
    print("─" * len(header))

    for name, strategy in STRATEGIES.items():
        row = f"{name:<20}"
        for cp in checkpoints:
            history = convos_by_turn.get(min(cp, max(convos_by_turn)), running)
            visible = strategy(history)
            t = context_tokens(visible)
            row += f"{t:>6}  "
        print(row)


# ─────────────────────────────────────────────────────────────
# Benchmark 2: Cost projection
# ─────────────────────────────────────────────────────────────
def cost_projection():
    print("\n" + "═" * 90)
    print("BENCHMARK 2: Cost per 30-turn session (gpt-4o-mini input: $0.15/1M)")
    print("═" * 90)

    running = []
    per_strategy_cost = {name: 0.0 for name in STRATEGIES}

    for role, content, _ in CONVERSATION:
        running.append({"role": role, "content": content})
        if role == "user":
            for name, strategy in STRATEGIES.items():
                visible = strategy(running)
                per_strategy_cost[name] += context_tokens(visible) * 0.15 / 1_000_000

    print(f"{'Strategy':<20}{'$/session':<15}{'$/day (10k sessions)':<25}{'$/month'}")
    for name, cost in per_strategy_cost.items():
        print(f"{name:<20}${cost:.6f}     ${cost*10000:.2f}                  ${cost*10000*30:.0f}")


# ─────────────────────────────────────────────────────────────
# Benchmark 3: Recall of planted facts at turn 30
# ─────────────────────────────────────────────────────────────
def recall_test():
    print("\n" + "═" * 90)
    print("BENCHMARK 3: Recall of 6 planted facts at turn 30")
    print("═" * 90)

    full_history = [{"role": r, "content": c} for r, c, _ in CONVERSATION]

    print(f"{'Fact':<30}", end="")
    for name in STRATEGIES:
        print(f"{name:<18}", end="")
    print()
    print("─" * 90)

    for key, description, keywords in PLANTED:
        row = f"{description:<30}"
        for name, strategy in STRATEGIES.items():
            visible = strategy(full_history)
            visible_text = " ".join(m["content"] for m in visible).lower()
            found = any(kw in visible_text for kw in keywords)
            summary_present = "[summary of prior turns]" in visible_text
            mark = "✅" if found else ("~" if summary_present else "❌")
            row += f"{mark:<18}"
        print(row)

    print("\nLegend: ✅ verbatim in context   ~ likely summarized   ❌ forgotten")


# ─────────────────────────────────────────────────────────────
# Decision matrix
# ─────────────────────────────────────────────────────────────
def decision_matrix():
    print("\n" + "═" * 90)
    print("DECISION MATRIX")
    print("═" * 90)
    print("""
  Scenario                                       Recommended strategy
  ────────────────────────────────────────────────────────────────────────
  Short chat, <10 turns                          Full history (cheapest, fine)
  Long chat, low-recall need                     Sliding window (w=6-10)
  Long chat, need to remember older facts        Sliding + episodic memory
  Multi-session persistent user                  Sliding + semantic profile
                                                 + episodic memory
  RAG-style: user brings context each time       Sliding + retrieved docs
  Latency-critical                               Sliding (no summary LLM call)
  Cost-critical, high volume                     Sliding (no extra LLM calls)
  Session may pause and resume                   Full state via LangGraph
                                                 checkpoints
""")


# ─────────────────────────────────────────────────────────────
# Production hybrid recommendation
# ─────────────────────────────────────────────────────────────
def production_hybrid():
    print("═" * 90)
    print("PRODUCTION HYBRID RECOMMENDATION")
    print("═" * 90)
    print("""
No single strategy wins for real apps. Layer them:

  Token budget for a production turn (~3,000 tokens overhead):
    ┌────────────────────────────────────────────────────────────┐
    │  ① Sliding window: last 6 turns          ~800 tokens        │
    │  ② Semantic profile: user facts          ~300 tokens        │
    │  ③ Episodic: 2-3 relevant past episodes  ~400 tokens        │
    │  ④ External retrieval: top-3 docs        ~1500 tokens       │
    │  ─────────────────────────────────────────────────         │
    │  TOTAL                                    ~3000 tokens      │
    └────────────────────────────────────────────────────────────┘

Why this scales to millions of users:
  • None of the 4 layers grows with total session count
  • Sliding window flat
  • Semantic profile bounded (a few dozen facts)
  • Episodic memory retrieved top-K, not full history
  • External retrieval already top-K

The 4-layer hybrid is what production systems (ChatGPT with Memory,
Anthropic's Claude with Projects, GPT store persistence) actually run.
""")


# ─────────────────────────────────────────────────────────────
# Run everything
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    token_growth_benchmark()
    cost_projection()
    recall_test()
    decision_matrix()
    production_hybrid()

    print("\n" + "═" * 90)
    print("✅ Phase 13 conclusion")
    print("═" * 90)
    print("""
No single memory strategy is 'correct' — the best system layers all four:

  Working memory      → sliding window (last 6 turns)
  Semantic memory     → user profile facts (versioned)
  Episodic memory     → session-level summaries indexed by topic/entity
  External memory     → vector retrieval on the corpus

Layer them via the pattern in Phase 13, evaluate with Phase 14 RAGAS,
guard the outputs with Phase 15, and observe with Phase 16.

You now have the full memory stack of a modern AI assistant.
""")
