# Phase 6 — Vector Databases

> **Real-life analogy:** A vector DB is a library where books are arranged by *topic similarity*,
> not alphabetically. Instead of "find book titled X", you say "find books similar to this one"
> and it returns the 5 most relevant.

---

## Why a "database" for vectors

Regular databases index by *equality* or *range* on scalar columns. Vector DBs index by
*proximity in N-dimensional space*. Different math, different index structure, different
use case.

```
SQL:  WHERE title LIKE '%bluetooth%'          ← keyword match
Vector: ORDER BY cosine(embedding, query_emb) ← meaning match
```

You need vector DBs whenever "similar meaning" beats "same string": search, memory,
recommendation, RAG.

---

## The HNSW Index (the one you'll encounter everywhere)

HNSW = Hierarchical Navigable Small World. It's the algorithm behind ChromaDB, Qdrant,
Weaviate, pgvector, Pinecone — everything.

```
Layer 3:         ●────────●          (few high-level "hub" nodes)
                 │        │
Layer 2:      ●──●──●──●──●──●
              │  │  │  │  │  │
Layer 1:   ●──●──●──●──●──●──●──●
           │  │  │  │  │  │  │  │
Layer 0:  ●──●──●──●──●──●──●──●──●  (all vectors)
              search path ─▶
```

Search jumps from top layer to bottom, following the neighbor closest to the query. Result:
sub-millisecond search over millions of vectors, with a small (1-5%) accuracy tradeoff vs
exact search. Almost always the right call.

---

## Chunking Strategies (the hidden bottleneck in RAG)

Bad chunking is the #1 reason RAG produces wrong answers. Choose consciously.

```
STRATEGY              LOOKS LIKE                    BEST FOR
─────────────────────────────────────────────────────────────────────
Fixed-size 500ch      [────500────][────500────]    Simple, uniform text
Sentence-aware        [sent1. sent2.][sent3. ...    Prose, articles
Paragraph-aware       [para1\n\n][para2\n\n]        Docs w/ clear structure
Semantic (topic-based) [topic-A block][topic-B blk] Long-form (research paper)
Parent-child          Retrieve child, RETURN parent Legal, code, tables
```

**Overlap:** 10-20% overlap prevents a fact getting cut in half at a boundary.

**Rule of thumb by content type:**

| Content | Chunk size | Overlap |
|---------|-----------|---------|
| FAQ / Q&A pairs | Whole answer per chunk | 0 |
| Prose articles | 400-800 tokens | 50-100 tokens |
| Long PDFs | 1000-1500 tokens | 100-150 tokens |
| Code | Function-level (parse AST) | 0 |
| Tables | Whole table if <8k tokens | 0 |

---

## Metadata Filtering — the "cheat code"

Pure vector search returns *semantically similar* items. What if you also want to filter?

```python
collection.query(
    query_texts=["wireless headphones for gym"],
    n_results=3,
    where={"$and": [
        {"category": "electronics"},
        {"price": {"$lt": 200}}
    ]}
)
```

**Why it matters:** without metadata filters, a query for "wireless headphones" returns
headphones AND semantically related items (phones with wireless charging, earbud
accessories, "wireless earbud reviews" blog posts). With a `where={"type": "product"}`
filter, you narrow to actual products.

**Rule:** ALWAYS store metadata alongside vectors. You'll always need to filter later.

---

## Hybrid Search: BM25 + Vector

Vectors are great for *concepts*, terrible for *exact matches* (product SKUs, names, error
codes). Combine both:

```
                              ┌─▶ vector score  ×  0.6
Query ─▶ Embed ─▶ Retrieve top-50 ─▶│                        └─▶ combined ─▶ top 5
                              └─▶ BM25 score    ×  0.4
```

For queries like "error code E-1024", BM25 dominates. For queries like "how do I set up
onboarding for new users", vector dominates. Weighted mix wins on both.

---

## Re-ranking — the quality boost you should always try

```
Query ─▶ Vector search top-20 ─▶ Cross-encoder re-score all 20 ─▶ top 5
                                    (heavy, but only 20 docs)
```

**Cross-encoder** processes (query, doc) *together* — much more accurate than the "bi-encoder"
that produced the initial embeddings, but too slow to run on the whole corpus. Perfect for
re-ranking the top-K.

Typical uplift: +10-25% in retrieval accuracy.

---

## Parent-Child Chunking

You want to *retrieve* by small chunk (precise match) but *return* the surrounding context.

```
Document ─▶ Split into small chunks (child, 200 tok) — used for RETRIEVAL
         └─▶ Split into big chunks  (parent, 2000 tok) — used for CONTEXT

Search  → find best child → look up its parent → return parent to LLM
```

Best of both worlds: precision at search time, context at answer time.

---

## Concept Table

| Term | Plain-English |
|------|---------------|
| **Embedding** | Fixed-length vector representing text meaning |
| **HNSW** | Fast approximate nearest neighbor index |
| **k-NN** | Return the K closest items |
| **Cosine similarity** | Angle-based similarity, 1..-1 |
| **Collection** | A named group of vectors + metadata |
| **Upsert** | Insert-or-update (idempotent) |
| **Metadata filter** | Restrict search to items matching k=v conditions |
| **BM25** | Classic keyword-search algorithm (TF-IDF++) |
| **Cross-encoder** | Model that scores (query, doc) together; slow but accurate |
| **Chunking** | Splitting long docs into retrievable pieces |
| **Overlap** | Repeat some tokens between adjacent chunks |
| **Parent-child** | Search on small chunks, return their bigger parent |

---

## Vector DB Comparison

| DB | Local? | Hosted? | Sweet spot |
|----|--------|---------|-----------|
| **ChromaDB** | ✅ | (managed via Cloud) | Prototyping, small/mid apps |
| **Qdrant** | ✅ | ✅ | Great filtering, Rust-fast |
| **Weaviate** | ✅ | ✅ | Multi-tenant, GraphQL API |
| **pgvector** | ✅ | via any managed Postgres | Already using Postgres |
| **Pinecone** | ❌ | ✅ | Serverless, no ops |
| **Milvus** | ✅ | ✅ | Billions of vectors |
| **FAISS** | ✅ (library, not DB) | — | Research, custom pipelines |

We use **ChromaDB** in this phase because it's local-only, zero-ops, and the API concepts
transfer directly to all the others.

---

## Folder structure

```
06-vector-databases/
├── README.md
├── 01_embeddings_deep_dive.py    ← MOCK embeddings, cosine, top-K by argsort
├── 02_semantic_search.py         ← Full semantic search over 20 products
├── 03_chromadb_local.py          ← ChromaDB in-memory + persistent
├── 04_vector_db_patterns.py      ← Chunking, hybrid search, re-ranking
└── requirements.txt
```
