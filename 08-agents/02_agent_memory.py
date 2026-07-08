"""
02_agent_memory.py — 4 agent memory strategies compared side-by-side
======================================================================

STRATEGIES
----------
1. SHORT-TERM (in-context)   — full history in messages array
2. SLIDING WINDOW            — keep only last N messages
3. SUMMARIZATION             — compress old turns into 1 summary message
4. EXTERNAL (vector)         — store/retrieve from a vector DB (dict-simulated)

For each, we show what happens at turns 5, 10, 15, 20 and the token count.
"""

import os
from dataclasses import dataclass, field

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


def approx_tokens(text: str) -> int:
    return max(1, len(text.split()) // 0.75.__floor__() if hasattr(0.75, "__floor__") else int(len(text.split()) / 0.75))


def token_count(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) // 4 for m in messages)


# ─────────────────────────────────────────────────────────────
# Strategy 1: Full history — no memory management
# ─────────────────────────────────────────────────────────────
class FullHistoryMemory:
    def __init__(self):
        self.messages = []
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    def context(self) -> list[dict]:
        return self.messages
    def name(self) -> str:
        return "full history"


# ─────────────────────────────────────────────────────────────
# Strategy 2: Sliding window
# ─────────────────────────────────────────────────────────────
class SlidingWindowMemory:
    def __init__(self, window: int = 10):
        self.window = window
        self.messages = []
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    def context(self) -> list[dict]:
        # Keep system message + last window messages
        sys = [m for m in self.messages if m["role"] == "system"][:1]
        rest = [m for m in self.messages if m["role"] != "system"]
        return sys + rest[-self.window:]
    def name(self) -> str:
        return f"sliding window ({self.window})"


# ─────────────────────────────────────────────────────────────
# Strategy 3: Summarization
# ─────────────────────────────────────────────────────────────
class SummarizationMemory:
    def __init__(self, keep_recent: int = 6):
        self.keep_recent = keep_recent
        self.messages = []
        self.summary = ""
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.keep_recent * 2:
            self._compress()
    def _compress(self):
        # Real: call LLM to summarize the oldest half
        old = self.messages[:-self.keep_recent]
        keep = self.messages[-self.keep_recent:]
        # Mock summary — just append short marker
        summary_text = f"[Prior {len(old)} messages summarized: user asked about X, Y, Z]"
        self.summary = (self.summary + " " + summary_text).strip()
        self.messages = keep
    def context(self) -> list[dict]:
        msgs = []
        if self.summary:
            msgs.append({"role": "system", "content": "Summary of earlier conversation: " + self.summary})
        msgs.extend(self.messages)
        return msgs
    def name(self) -> str:
        return f"summarization (keep {self.keep_recent})"


# ─────────────────────────────────────────────────────────────
# Strategy 4: External (dict-simulated vector DB)
# ─────────────────────────────────────────────────────────────
class ExternalMemory:
    """Store every turn to an 'external store'. On demand, retrieve top-K
    relevant past turns using naive keyword matching (a real system would
    use embeddings — see Phase 13)."""
    def __init__(self, in_context: int = 4):
        self.in_context = in_context
        self.messages = []
        self.store: list[dict] = []
    def add(self, role: str, content: str):
        m = {"role": role, "content": content}
        self.messages.append(m)
        if role != "system":
            self.store.append(m)
    def context(self, query: str = "") -> list[dict]:
        sys = [m for m in self.messages if m["role"] == "system"][:1]
        recent = [m for m in self.messages if m["role"] != "system"][-self.in_context:]
        retrieved = self._search(query, k=2) if query else []
        return sys + retrieved + recent
    def _search(self, query: str, k: int) -> list[dict]:
        q_words = set(query.lower().split())
        scored = [(sum(1 for w in q_words if w in m["content"].lower()), m)
                  for m in self.store]
        scored.sort(key=lambda t: -t[0])
        return [m for s, m in scored[:k] if s > 0]
    def name(self) -> str:
        return f"external memory (in-context {self.in_context})"


# ─────────────────────────────────────────────────────────────
# Benchmark on a fake 20-turn conversation
# ─────────────────────────────────────────────────────────────
CONVERSATION = [
    ("user", "Hi, I'm Alice, a software engineer at FinBank."),
    ("assistant", "Nice to meet you, Alice. How can I help?"),
    ("user", "I'm looking at building a fraud detection system."),
    ("assistant", "Great — what's your baseline approach?"),
    ("user", "We use gradient boosting today. Precision is 92%."),
    ("assistant", "That's solid. Interested in adding LLM-based features?"),
    ("user", "Yes — I want to summarize transaction narratives."),
    ("assistant", "You can use GPT-4o-mini for that."),
    ("user", "What about cost at 5M transactions per day?"),
    ("assistant", "At 5M/day with ~50 tokens each, ~$375/day for -mini."),
    ("user", "Can we batch requests?"),
    ("assistant", "Yes, OpenAI's Batch API gives 50% discount."),
    ("user", "OK. Also, remember I mentioned FinBank earlier."),
    ("assistant", "Yes, you're at FinBank."),
    ("user", "What was my baseline precision I mentioned?"),
    ("assistant", "You said 92%."),
    ("user", "Perfect memory. What's my name?"),
    ("assistant", "Alice."),
    ("user", "And my role?"),
    ("assistant", "Software engineer at FinBank."),
]


def bench(memory_class, ctor_args=None, retrieve_query=None):
    m = memory_class(**(ctor_args or {}))
    m.add("system", "You are a helpful engineer assistant.")

    checkpoints = [5, 10, 15, 20]
    print(f"\n=== {m.name()} ===")
    print(f"  turn |  msgs held  |  approx tokens  |")
    for i, (role, content) in enumerate(CONVERSATION, 1):
        m.add(role, content)
        if i in checkpoints:
            ctx = m.context(retrieve_query) if isinstance(m, ExternalMemory) else m.context()
            print(f"  {i:>4} |  {len(ctx):>9}  |  {token_count(ctx):>13}  |")


if __name__ == "__main__":
    bench(FullHistoryMemory)
    bench(SlidingWindowMemory, {"window": 6})
    bench(SummarizationMemory, {"keep_recent": 6})
    bench(ExternalMemory, {"in_context": 4}, retrieve_query="baseline precision")

    print("""

✅ Summary — 4 memory strategies compared:

  Strategy         | Token growth | Recall of old facts | Cost
  -----------------+--------------+---------------------+------
  Full history     |   linear     |        perfect      | high
  Sliding window   |   flat       |     forgets fast    | low
  Summarization    |   slow       |    lossy but okay   | med
  External memory  |   flat       |   great if right K  | med

Rule for production: HYBRID (Phase 13):
    • Sliding window: last 6 turns (recent context)
    • Semantic profile: user's stable facts ('at FinBank, engineer')
    • Episodic memory: past sessions summarized
    • External retrieval: search for relevant older turns on demand

Total budget: ~3,000 tokens per turn — scales to millions of users.
""")
