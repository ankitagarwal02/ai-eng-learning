# Design 3 — Enterprise RAG over 100M Documents

## Requirements
- 100M documents (avg 5 pages, ~2000 tokens each)
- Total tokens indexed: ~200B
- Ingestion: 500k new/updated docs per day
- Query QPS: peak 500
- Query latency: p95 < 1.5s
- Answer must include source citations

## The 100M-doc storage math

```
100M docs × ~10 chunks/doc = 1B chunks
1B chunks × 1536-dim float32 embedding = 6TB just for vectors
Metadata (~500 bytes/chunk) = 500GB

→ single vector DB node doesn't fit. Must shard.
```

## Sharding strategy

**Shard by:** tenant_id (if multi-tenant) or doc_source (department, product line).
Never shard randomly — locality matters for both cost and query time.

```
Query Router ─▶ hits N relevant shards in parallel (based on user's ACL / tenant)
             ─▶ merges top-K from each
             ─▶ re-ranks combined
             ─▶ returns top-K global
```

Typical setup: 50 shards × 20M vectors each on Qdrant / Weaviate cluster.

## Ingestion pipeline

```
Source (S3/SharePoint/Confluence)
       │
       ▼
┌──────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Watcher     │──▶│  Chunker │──▶│ Embedder │──▶│ Vector DB│
│  (change     │   │ (parent- │   │(batched  │   │(sharded) │
│  detection)  │   │  child)  │   │via Batch)│   │          │
└──────────────┘   └──────────┘   └──────────┘   └──────────┘
       │                 │              │             │
       └── event bus ────┴──────────────┴─────────────┘
                                │
                                ▼
                       Dead-letter queue
                       (failed docs)

Throughput target: 500k docs/day = ~6/s sustained.
Use OpenAI Batch API for embeddings → 50% cost cut, 24h latency acceptable.
```

## Query path

```
Query (1) ─▶ Query planner: expand + HyDE (LLM call ~200ms)
        (2) ─▶ Route to relevant shards (based on user ACL, ~5ms)
        (3) ─▶ Parallel search all N shards (~200ms)
        (4) ─▶ Merge top-20 from each → top-100 global
        (5) ─▶ Cross-encoder rerank top-100 → top-10 (~300ms)
        (6) ─▶ Context compression (~200ms)
        (7) ─▶ LLM answer with citations (streaming, TTFT ~400ms)

Total to first token: ~1.3s   ✅ under 1.5s SLO
```

## Cost per query

```
Query expansion + HyDE (mini):        ~$0.0003
Rerank (cross-encoder self-hosted):   ~$0.0001
LLM answer (gpt-4o-mini):             ~$0.0015
Vector DB read (5 shards):            ~$0.0005
───────────────────────────────────────────
Total per query:                       ~$0.0024

500 QPS × 86,400s = 43M queries/day × $0.0024 = $103k/day
                                                = $3.1M/month
```

## Optimizations to cut cost by 60-70%

- Cache top-10% frequent queries (Redis, semantic hash) → -25%
- Route "simple" queries (single-doc lookup) to keyword + no LLM → -15%
- Aggressive prompt caching on the LLM answer prompt → -10%
- Use OpenAI's Batch API for offline enrichment (not user queries)

## Failure modes to discuss

- **Cold cache stampede** — sudden re-index causes cache miss cascade. Mitigation: pre-warm.
- **Shard hotspot** — one shard gets 10× queries. Mitigation: monitor + re-shard.
- **Stale index** — user asks about doc modified today. Mitigation: cache TTL per shard, incremental re-embed on webhook.
