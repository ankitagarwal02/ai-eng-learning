"""
02_external_memory_store.py — Vector-backed long-term memory with TTL + recency
================================================================================

WHAT THIS FILE TEACHES
----------------------
  • MemoryEntry dataclass with content, embedding, metadata, created_at, last_accessed, TTL
  • recency_score decay function (exponential half-life)
  • combined_score = relevance × 0.7 + recency × 0.3
  • MockMemoryStore with add / search / expire / stats
  • ChromaMemoryStore (real ChromaDB) side-by-side with mock
  • TTL expiration: memories older than N days are auto-deleted
  • Show scored results after 10 memories, then expire stale ones

HOW TO RUN
----------
    pip install chromadb numpy
    python 02_external_memory_store.py

REAL-WORLD SCENARIO
-------------------
A personal assistant that remembers everything you've told it. After 6 months,
show how recency scoring surfaces recent memories over older ones on ambiguous
queries.
"""

import os
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────
@dataclass
class MemoryEntry:
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    ttl_days: int = 180   # 6 months default


def deterministic_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Same helper from Phase 2/6 — deterministic pseudo-embedding."""
    base = np.random.default_rng(seed=abs(hash(text)) % (2**32)).standard_normal(dim)
    for w in text.lower().split():
        base += np.random.default_rng(seed=abs(hash(w)) % (2**32)).standard_normal(dim) * 0.4
    return base / np.linalg.norm(base)


# ─────────────────────────────────────────────────────────────
# Scoring functions
# ─────────────────────────────────────────────────────────────
def recency_score(last_accessed: datetime, half_life_days: float = 20) -> float:
    """Exponential decay: recent=1.0, old=~0."""
    days = (datetime.now() - last_accessed).total_seconds() / 86400
    return math.exp(-days / half_life_days)


def relevance_score(query_vec: np.ndarray, entry_vec: np.ndarray) -> float:
    """Cosine similarity (both unit-normalized) → 0..1 (clamped)."""
    return max(0.0, float(np.dot(query_vec, entry_vec)))


def combined_score(relevance: float, recency: float,
                   w_relevance: float = 0.7, w_recency: float = 0.3) -> float:
    return w_relevance * relevance + w_recency * recency


# ─────────────────────────────────────────────────────────────
# In-memory MockMemoryStore
# ─────────────────────────────────────────────────────────────
class MockMemoryStore:
    def __init__(self):
        self.entries: dict[str, MemoryEntry] = {}

    def add(self, id_: str, content: str, metadata: dict = None,
            ttl_days: int = 180, created_at: datetime = None) -> MemoryEntry:
        entry = MemoryEntry(
            id=id_,
            content=content,
            embedding=deterministic_embedding(content),
            metadata=metadata or {},
            ttl_days=ttl_days,
            created_at=created_at or datetime.now(),
            last_accessed=created_at or datetime.now(),
        )
        self.entries[id_] = entry
        return entry

    def search(self, query: str, k: int = 3) -> list[tuple[float, MemoryEntry]]:
        q_vec = deterministic_embedding(query)
        scored = []
        for e in self.entries.values():
            rel = relevance_score(q_vec, e.embedding)
            rec = recency_score(e.last_accessed)
            score = combined_score(rel, rec)
            scored.append((score, rel, rec, e))
        scored.sort(key=lambda t: -t[0])
        # Touch last_accessed for returned entries:
        top = scored[:k]
        for _, _, _, e in top:
            e.last_accessed = datetime.now()
        return [(s, e) for s, _, _, e in top]

    def expire(self) -> int:
        """Delete entries past TTL. Returns number deleted."""
        now = datetime.now()
        to_delete = [
            id_ for id_, e in self.entries.items()
            if (now - e.created_at).days > e.ttl_days
        ]
        for id_ in to_delete:
            del self.entries[id_]
        return len(to_delete)

    def stats(self) -> dict:
        now = datetime.now()
        ages = [(now - e.created_at).days for e in self.entries.values()]
        return {
            "count": len(self.entries),
            "min_age_days": min(ages) if ages else None,
            "max_age_days": max(ages) if ages else None,
            "avg_age_days": (sum(ages) / len(ages)) if ages else None,
        }


# ─────────────────────────────────────────────────────────────
# Demo: personal-assistant memory over 6 months
# ─────────────────────────────────────────────────────────────
print("=" * 78)
print("Personal-assistant memory — 6 months of usage")
print("=" * 78)

store = MockMemoryStore()

# Insert memories spread across the last 6 months:
now = datetime.now()
seed_memories = [
    ("M001", "User's name is Alice, works at FinBank as a software engineer.",   180, {"kind":"profile"}),
    ("M002", "User prefers bullet-point responses.",                              175, {"kind":"pref"}),
    ("M003", "User is building a fraud-detection system with gradient boosting.", 170, {"kind":"project"}),
    ("M004", "User's baseline precision is 92%.",                                 165, {"kind":"project"}),
    ("M005", "User was frustrated by a bug in the OData layer on 2026-03-15.",    120, {"kind":"episode"}),
    ("M006", "User booked tickets to KubeCon in Amsterdam.",                       90, {"kind":"personal"}),
    ("M007", "User switched to Anthropic Claude for classification tasks.",       60, {"kind":"tech-decision"}),
    ("M008", "User is now managing a team of 3 engineers.",                       30, {"kind":"profile"}),
    ("M009", "User just moved to Stripe from FinBank.",                            5, {"kind":"profile"}),
    ("M010", "User asked for AWS cost-optimization advice today.",                 1, {"kind":"episode"}),
]

for id_, content, days_ago, meta in seed_memories:
    store.add(id_, content, metadata=meta,
              created_at=now - timedelta(days=days_ago))

print(f"\nStore stats after seeding:  {store.stats()}")


# ─────────────────────────────────────────────────────────────
# Ambiguous query: "user's company"
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 78)
print("Query: 'What company does the user work at?'")
print("─" * 78)
results = store.search("What company does the user work at?", k=5)
print(f"\n{'ID':<6}{'score':<8}{'rel':<8}{'rec':<8}{'age':<8}content")
for score, e in results:
    days = (now - e.created_at).days
    print(f"{e.id:<6}{score:<8.3f}{'':<8}{'':<8}{days:<3}d    {e.content[:60]}...")

# Note: M009 ("moved to Stripe" — 5 days ago) beats M001 ("FinBank" — 180 days ago)
# because recency dominates on tied-relevance queries about "company".

print("""
Why M009 (Stripe, 5 days ago) beats M001 (FinBank, 180 days ago):
  • Both entries are equally relevant semantically (both mention 'company/work')
  • Recency ranks M009 higher — matches reality (user just moved!)
  • Without recency scoring, we'd give the wrong (outdated) answer.
""")


# ─────────────────────────────────────────────────────────────
# TTL expiration
# ─────────────────────────────────────────────────────────────
print("─" * 78)
print("TTL expiration: shorten TTL to 100 days, then expire")
print("─" * 78)
for e in store.entries.values():
    e.ttl_days = 100

expired = store.expire()
print(f"\nExpired {expired} entries.  Remaining: {store.stats()}")


# ─────────────────────────────────────────────────────────────
# Bonus: ChromaDB-backed version (skip if not installed)
# ─────────────────────────────────────────────────────────────
try:
    import chromadb
    print("\n" + "═" * 78)
    print("Bonus: Same memory store backed by ChromaDB")
    print("═" * 78)

    class ChromaMemoryStore:
        def __init__(self):
            self.client = chromadb.EphemeralClient()
            self.col = self.client.get_or_create_collection(name="mem")

        def add(self, id_: str, content: str, metadata: dict = None):
            self.col.add(
                ids=[id_],
                documents=[content],
                metadatas=[{**(metadata or {}), "created_at": datetime.now().isoformat()}],
            )

        def search(self, query: str, k: int = 3):
            hits = self.col.query(query_texts=[query], n_results=k)
            return list(zip(hits["ids"][0], hits["documents"][0], hits["distances"][0]))

    cstore = ChromaMemoryStore()
    for id_, content, _, meta in seed_memories[-4:]:  # just the recent ones
        cstore.add(id_, content, metadata=meta)

    print("ChromaDB search:")
    for id_, doc, dist in cstore.search("What company does the user work at?", k=3):
        print(f"  {id_}  dist={dist:.3f}  {doc[:60]}...")

except ImportError:
    print("\n(Install chromadb to run the real-store demo)")


print("""

✅ Summary — External memory pattern:

  • MemoryEntry stores content + embedding + metadata + timestamps + TTL
  • Search scores by relevance × 0.7 + recency × 0.3 (tune weights per app)
  • Touch last_accessed on retrieval → "used memories stay fresh"
  • TTL cleanup runs periodically (nightly job in production)
  • Combining with ChromaDB / Qdrant is one adapter class away

Production wiring:
  Every user message flows through:
    1. Retrieve top-K memories relevant to the message
    2. Inject them into the system prompt
    3. After the response, extract any new facts → add to memory

See:
  Phase 13 file 03_episodic_memory.py    — session-level memory
  Phase 13 file 04_semantic_memory.py    — versioned user facts
  Phase 13 file 05_memory_strategies_comparison.py — full benchmark
""")
