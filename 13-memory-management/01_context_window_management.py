"""
01_context_window_management.py — Sliding window, summarization, token budget
==============================================================================

WHAT THIS FILE TEACHES
----------------------
  • Exact token counting per message with tiktoken (fallback to estimate)
  • SlidingWindowMemory: FIFO drop of oldest turns
  • SummarizationMemory: compress old turns when over limit
  • TokenBudget: allocate fixed tokens per component (system, history, docs, output)
  • Show growth pattern of each across 20 turns
  • The cost of NOT managing context: what happens at turn 50

HOW TO RUN
----------
    pip install tiktoken
    python 01_context_window_management.py

REAL-WORLD SCENARIO
-------------------
A 30-turn customer support session. We show token usage WITH and WITHOUT
context management. Then calculate the cost difference over 10,000 daily
sessions.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return max(1, int(len(text) / 4))


# ─────────────────────────────────────────────────────────────
# Base memory interface
# ─────────────────────────────────────────────────────────────
@dataclass
class Message:
    role: str
    content: str

    def token_count(self) -> int:
        return count_tokens(self.content) + 4   # ~4 tokens overhead per msg


class Memory:
    def __init__(self):
        self.messages: list[Message] = []
    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role, content))
    def get_context(self) -> list[Message]:
        return self.messages
    def total_tokens(self) -> int:
        return sum(m.token_count() for m in self.get_context())
    def name(self) -> str:
        return "unbounded"


# ─────────────────────────────────────────────────────────────
# Strategy 1: Unbounded (baseline — for cost comparison)
# ─────────────────────────────────────────────────────────────
class UnboundedMemory(Memory):
    pass   # keep everything, always


# ─────────────────────────────────────────────────────────────
# Strategy 2: Sliding window
# ─────────────────────────────────────────────────────────────
class SlidingWindowMemory(Memory):
    def __init__(self, window: int = 6):
        super().__init__()
        self.window = window
    def get_context(self) -> list[Message]:
        # Keep system messages + last `window` non-system messages
        sys = [m for m in self.messages if m.role == "system"]
        rest = [m for m in self.messages if m.role != "system"]
        return sys + rest[-self.window:]
    def name(self) -> str:
        return f"sliding_window(w={self.window})"


# ─────────────────────────────────────────────────────────────
# Strategy 3: Summarization
# ─────────────────────────────────────────────────────────────
class SummarizationMemory(Memory):
    def __init__(self, keep_recent: int = 4, trigger_at: int = 12):
        super().__init__()
        self.keep_recent = keep_recent
        self.trigger_at = trigger_at
        self.summary_msg: Optional[Message] = None
        self.compressed_count = 0

    def add(self, role: str, content: str) -> None:
        super().add(role, content)
        rest = [m for m in self.messages if m.role != "system"]
        if len(rest) >= self.trigger_at:
            self._compress()

    def _compress(self):
        """Move oldest half of non-system messages into a summary message."""
        non_sys = [m for m in self.messages if m.role != "system"]
        old = non_sys[:-self.keep_recent]
        keep = non_sys[-self.keep_recent:]
        # Real: LLM summarizes `old`. Mock: concatenate + truncate.
        combined = " | ".join(f"{m.role}:{m.content}" for m in old)
        summary_text = f"[SUMMARY of prior {len(old)} turns] " + combined[:200] + "..."
        self.summary_msg = Message("system", summary_text)
        self.compressed_count += len(old)
        # Rebuild message list: system originals + summary + recent
        self.messages = [m for m in self.messages if m.role == "system" and m is not self.summary_msg]
        self.messages.append(self.summary_msg)
        self.messages.extend(keep)

    def name(self) -> str:
        return f"summarization(recent={self.keep_recent}, trigger={self.trigger_at})"


# ─────────────────────────────────────────────────────────────
# TokenBudget — allocate fixed tokens per component
# ─────────────────────────────────────────────────────────────
@dataclass
class TokenBudget:
    system: int = 500
    history: int = 2000
    retrieved_docs: int = 1500
    output: int = 1000

    def total_input(self) -> int:
        return self.system + self.history + self.retrieved_docs

    def total_estimate(self) -> int:
        return self.total_input() + self.output


def fit_within_budget(messages: list[Message], max_tokens: int) -> list[Message]:
    """Truncate from the OLDEST end until we fit under max_tokens.
    Keeps system messages in front."""
    sys = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]

    total = sum(m.token_count() for m in sys)
    kept_rest = []
    # Walk from newest backwards, add until budget exceeded:
    for m in reversed(rest):
        if total + m.token_count() > max_tokens:
            break
        kept_rest.insert(0, m)
        total += m.token_count()

    return sys + kept_rest


# ─────────────────────────────────────────────────────────────
# Benchmark: 30-turn realistic conversation
# ─────────────────────────────────────────────────────────────
CONVERSATION = [
    ("system", "You are Aria, a customer support agent for FinBank."),
    ("user", "Hi, my name is Alice and I'm a software engineer at FinBank."),
    ("assistant", "Nice to meet you, Alice. How can I help today?"),
    ("user", "I'm troubleshooting fraud detection performance."),
    ("assistant", "Great — what's your baseline?"),
    ("user", "Gradient boosting, precision 92%."),
    ("assistant", "Solid. Considering LLM features?"),
    ("user", "Yes. Want to summarize transaction narratives."),
    ("assistant", "gpt-4o-mini fits — roughly $0.02/1M tokens."),
    ("user", "At 5M/day transactions?"),
    ("assistant", "About $375/day for -mini, $6000/day for -4o."),
    ("user", "Can we batch?"),
    ("assistant", "OpenAI Batch API cuts cost by 50%."),
    ("user", "OK. My name again?"),
    ("assistant", "Alice."),
    ("user", "My baseline precision?"),
    ("assistant", "92%."),
    ("user", "And my company?"),
    ("assistant", "FinBank."),
    ("user", "My role?"),
    ("assistant", "Software engineer."),
    ("user", "Model I'm using?"),
    ("assistant", "Gradient boosting."),
    ("user", "Cost estimate at 10M transactions/day?"),
    ("assistant", "About $750/day for -mini."),
    ("user", "What about Anthropic?"),
    ("assistant", "Claude Haiku is cheaper still — ~$0.25/1M input."),
    ("user", "What was my company?"),
    ("assistant", "FinBank."),
    ("user", "OK, thanks — let's wrap up."),
    ("assistant", "Anytime, Alice. Good luck with the fraud detection."),
]


def bench(memory: Memory, checkpoints: list[int] = [5, 10, 15, 20, 25, 30]) -> None:
    for role, content in CONVERSATION:
        memory.add(role, content)

    print(f"\n─── {memory.name()} ───")
    print(f"  Turn |  msgs  |  tokens  |  cost/call ($) @ gpt-4o-mini input")
    print(f"  ─────┼────────┼──────────┼───────────────────────────────────")

    for turn in checkpoints:
        m = memory.__class__(**({"window": memory.window} if hasattr(memory, "window") else
                                {"keep_recent": memory.keep_recent, "trigger_at": memory.trigger_at}
                                if hasattr(memory, "keep_recent") else {}))
        for role, content in CONVERSATION[:turn]:
            m.add(role, content)
        n_msgs = len(m.get_context())
        toks = m.total_tokens()
        cost = toks * 0.15 / 1_000_000
        print(f"  {turn:>4} |  {n_msgs:>5} |  {toks:>7}  |  ${cost:.6f}")


# ─────────────────────────────────────────────────────────────
# Run benchmarks
# ─────────────────────────────────────────────────────────────
print("=" * 78)
print("Context window benchmark: 30-turn realistic conversation")
print("=" * 78)

bench(UnboundedMemory())
bench(SlidingWindowMemory(window=6))
bench(SummarizationMemory(keep_recent=4, trigger_at=12))


# ─────────────────────────────────────────────────────────────
# TokenBudget demo
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("TokenBudget — fit_within_budget()")
print("=" * 78)

budget = TokenBudget(system=200, history=800, retrieved_docs=1500, output=1000)
print(f"\nBudget: system={budget.system}  history={budget.history}  "
      f"docs={budget.retrieved_docs}  output={budget.output}")
print(f"Total INPUT budget: {budget.total_input()} tokens")

full = UnboundedMemory()
for role, content in CONVERSATION:
    full.add(role, content)

fitted = fit_within_budget(full.messages, budget.history)
print(f"\nFull history: {full.total_tokens()} tokens across {len(full.messages)} msgs")
print(f"After fit_within_budget({budget.history}): "
      f"{sum(m.token_count() for m in fitted)} tokens across {len(fitted)} msgs")


# ─────────────────────────────────────────────────────────────
# The cost of NOT managing context: turn 50
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("The cost of not managing context — projecting to turn 50")
print("=" * 78)

# Extrapolate: average ~30 tokens per turn
avg_per_turn = full.total_tokens() / 30

print(f"\nAverage tokens per turn: {avg_per_turn:.1f}")
print(f"At turn 50 (unbounded): ~{avg_per_turn * 50:.0f} tokens = "
      f"${avg_per_turn * 50 * 0.15 / 1_000_000:.6f}/call")

# Compare cost over 10,000 daily 30-turn sessions:
per_session_unbounded = sum(
    (avg_per_turn * (i + 1)) * 0.15 / 1_000_000
    for i in range(30)
)
per_session_sliding = sum(
    min(avg_per_turn * 6, avg_per_turn * (i + 1)) * 0.15 / 1_000_000
    for i in range(30)
)
per_session_summary = per_session_sliding * 1.3   # small summary overhead

daily_users = 10_000
print(f"\nCost per session (30 turns):")
print(f"  Unbounded:     ${per_session_unbounded:.4f}")
print(f"  Sliding w=6:   ${per_session_sliding:.4f}")
print(f"  Summarization: ${per_session_summary:.4f}")
print(f"\nAt {daily_users:,} daily sessions:")
print(f"  Unbounded:     ${per_session_unbounded * daily_users * 30:.0f} / month")
print(f"  Sliding w=6:   ${per_session_sliding * daily_users * 30:.0f} / month")
print(f"  Savings:       ~{(1 - per_session_sliding / per_session_unbounded) * 100:.0f}%")


print("""

✅ Summary — Context window management:

  • Unbounded:      grows quadratically w/ turns (each call includes all prior). BAD.
  • Sliding window: flat token usage after window fills. Loses old info.
  • Summarization:  slow growth, retains gist of old turns.
  • TokenBudget:    guarantees you never exceed a hard cap.

Production hybrid (Phase 13's 05_memory_strategies_comparison.py):
    sliding_window(6) + semantic_profile + episodic + retrieved_docs
    ≈ 3000 tokens total, regardless of session length.

Every prod LLM app must do context management. It's not optional at scale.
""")
