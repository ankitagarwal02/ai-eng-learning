# Phase 13 — Memory Management

> **Real-life analogy:** Your brain has 4 memory types — working (thinking now), long-term
> (learned facts), episodic ("my first day at school"), semantic ("Paris is in France").
> Real agents need all four.

---

## The 4 memory types

```
┌────────────────────────────────────────────────────────────────────┐
│  WORKING (in-context)     current conversation, evicted at end     │
│  EPISODIC                  past sessions/events, indexed by time    │
│  SEMANTIC                  stable facts about the user/domain       │
│  EXTERNAL (retrieval)      arbitrary content, retrieved on demand   │
└────────────────────────────────────────────────────────────────────┘
```

## The production hybrid

```
Prompt budget: ~3,000 tokens overhead per turn
    ├── Sliding window (last 6 turns)         ~800 tok
    ├── Semantic profile (user facts)          ~300 tok
    ├── Episodic (2-3 relevant past episodes)  ~400 tok
    └── External retrieval (top-3 docs)       ~1500 tok
```

Scales to millions of users because none of these grow linearly with session length.

## Recency scoring

```
recency_score(now, last_accessed):
    days = (now - last_accessed).days
    return exp(-days / 30)          # half-life ~ 20 days

combined_score = 0.7 * relevance + 0.3 * recency
```

## Folder structure

```
13-memory-management/
├── README.md
├── 01_context_window_management.py   ← Sliding window, summarization, token budget
├── 02_external_memory_store.py       ← MemoryEntry, ChromaMemoryStore, TTL
├── 03_episodic_memory.py             ← Episode dataclass, by-topic/entity/time
├── 04_semantic_memory.py             ← UserProfile, fact extraction, versioning
├── 05_memory_strategies_comparison.py ← Side-by-side 30-turn benchmark
└── requirements.txt
```
