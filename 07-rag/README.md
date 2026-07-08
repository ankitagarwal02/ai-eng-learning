# Phase 7 — Retrieval-Augmented Generation (RAG)

> **Real-life analogy:** RAG is an open-book exam. Without RAG, the LLM only knows what it
> memorized during training. With RAG, it can look up the answer in your documents *before*
> responding — dramatically reducing hallucination.

---

## The RAG Pipeline

```
                                        ┌──────────────────┐
User query ─▶ Embed ─▶ Vector search ──▶│  Top-K chunks     │──▶ Prompt template ──▶ LLM ──▶ Answer
                          │             └──────────────────┘                            │
                          ▼                                                              │
                     Vector DB                                                            │
                     (ChromaDB)                                                           │
                                                                                          ▼
                                                                          Answer + inline citations
```

Every RAG failure has one of 3 root causes:

1. **Wrong retrieval** — the right chunk wasn't in top-K
2. **Lost in the middle** — the right chunk was there but LLM ignored it (usually because it was in the middle of a long context)
3. **Hallucination** — LLM invented facts NOT in the retrieved context

Phases 6, 7, and 14 give you the tools to diagnose and fix each.

---

## Naive RAG vs Advanced RAG

```
Naive:  Query ─▶ Vector search top-3 ─▶ LLM
        Wins on: FAQ-style Q&A, small corpora, prototyping

Advanced pipeline (production):
        Query ─▶ Query expansion (3 variants)
              ─▶ HyDE (generate a fake ideal answer, embed that)
              ─▶ Retrieve top-20 (hybrid vector+BM25)
              ─▶ Cross-encoder re-rank → top-5
              ─▶ Context compression (extract sentences that matter)
              ─▶ LLM answer with inline [source] citations
              ─▶ Faithfulness check (Phase 14)
```

Each step adds latency + cost — layer them only when quality demands it.

---

## Advanced Patterns

### HyDE (Hypothetical Document Embeddings)

The user's query is often short and imprecise. HyDE has the LLM *first* draft a fake ideal answer, then embeds THAT for retrieval — usually finds better chunks.

```
Query:  "PTO carryover"                     (short, ambiguous)
        ▼
LLM drafts fake answer: "Paid time off may be carried over up to
                          5 days into the next calendar year..."
        ▼
Embed the fake answer → search → find the REAL policy paragraph.
```

### Query Expansion

Generate multiple query rewrites, retrieve for each, union & dedupe.

```
Query: "how much vacation"
Expansions:
  1. "PTO policy"
  2. "annual leave days"
  3. "how many days off per year"
Union → dedupe → top-K.
```

### Context Compression

After retrieval, use an LLM to extract only the sentences relevant to the query. Cuts context length 5-10× without losing answer quality.

### Citations & Faithfulness

Always tag chunks and require the LLM to cite. Post-check: does every sentence in the answer trace to a cited chunk? If not, the LLM is hallucinating.

---

## Folder structure

```
07-rag/
├── README.md
├── 01_naive_rag.py            ← 30-line minimum working RAG
├── 02_advanced_rag.py         ← HyDE, query expansion, re-ranking, compression
├── 03_rag_with_citations.py   ← Inline citations + faithfulness scoring
└── requirements.txt
```

**Prereqs:** Phase 5 (embeddings), Phase 6 (ChromaDB, chunking).

**Next:** Phase 14 (Evals — RAGAS metrics measure how good your RAG actually is).
