"""
04_vector_db_patterns.py — Chunking, hybrid search, re-ranking, parent-child
=============================================================================

WHAT THIS FILE TEACHES
----------------------
• Fixed-size, sentence-aware, paragraph-aware chunking (with overlap).
• Parent-child chunking: retrieve small chunk, return its bigger parent.
• Hybrid search: BM25 keyword + vector cosine, weighted.
• Re-ranking: cross-encoder over top-K (simulated in MOCK_MODE).

HOW TO RUN
----------
    pip install numpy
    python 04_vector_db_patterns.py

REAL-WORLD SCENARIO
-------------------
You have a 3000-word company HR policy document. Split it correctly, index it,
and answer the query "How many personal days do I get?" reliably.
"""

import os
import re
import math
import numpy as np
from typing import Optional

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# Sample corpus (a mini HR policy)
# ─────────────────────────────────────────────────────────────
HR_POLICY = """
SECTION 1 — PAID TIME OFF (PTO)

All full-time employees accrue paid time off at 1.25 days per month, totaling
15 days per year. PTO can be used for vacation, personal reasons, or health
appointments and is credited on the first of each month.

Unused PTO carries over up to 5 days into the next calendar year. Any accrued
PTO above the carryover limit is forfeited on January 1.

SECTION 2 — SICK LEAVE

Full-time employees receive 10 paid sick days per calendar year, credited on
January 1. Sick leave does not accrue and does not roll over. For absences of
more than 3 consecutive days, a physician's note is required.

SECTION 3 — PERSONAL DAYS

In addition to PTO and sick leave, employees receive 3 personal days per year.
These may be used for any personal reason without justification. Personal days
must be used by December 31 and do not roll over.

SECTION 4 — PARENTAL LEAVE

New parents (birth, adoption, foster placement) receive 16 weeks of paid
parental leave, which may be taken any time during the first 12 months.
Additional unpaid leave may be requested through HR.

SECTION 5 — BEREAVEMENT

Employees are entitled to 5 days of paid bereavement leave upon the death of
an immediate family member. Extensions and travel time may be granted
case-by-case.
""".strip()


# ─────────────────────────────────────────────────────────────
# Simple embed function
# ─────────────────────────────────────────────────────────────
def embed(texts: list[str], dim: int = 128) -> np.ndarray:
    vectors = []
    for t in texts:
        base = np.random.default_rng(seed=abs(hash(t)) % (2**32)).standard_normal(dim)
        for w in t.lower().split():
            base += np.random.default_rng(seed=abs(hash(w)) % (2**32)).standard_normal(dim) * 0.4
        base /= np.linalg.norm(base)
        vectors.append(base)
    return np.array(vectors)


# ─────────────────────────────────────────────────────────────
# SECTION 1: Fixed-size chunking with overlap
# ─────────────────────────────────────────────────────────────
def fixed_size_chunks(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Split into `size`-char windows with `overlap` chars repeated between."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ─────────────────────────────────────────────────────────────
# SECTION 2: Sentence-aware chunking (respects sentence boundaries)
# ─────────────────────────────────────────────────────────────
def sentence_chunks(text: str, max_chars: int = 400) -> list[str]:
    """Group whole sentences up to `max_chars`."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current.strip())
    return chunks


# ─────────────────────────────────────────────────────────────
# SECTION 3: Paragraph-aware chunking (uses double-newline)
# ─────────────────────────────────────────────────────────────
def paragraph_chunks(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


print("=" * 70)
print("SECTION 1-3: Chunking strategies compared")
print("=" * 70)
for name, chunks in [
    ("Fixed-size (300ch, 50 overlap)", fixed_size_chunks(HR_POLICY, 300, 50)),
    ("Sentence-aware (400ch max)",      sentence_chunks(HR_POLICY, 400)),
    ("Paragraph-aware",                 paragraph_chunks(HR_POLICY)),
]:
    lengths = [len(c) for c in chunks]
    print(f"\n  {name}")
    print(f"    → {len(chunks)} chunks  min={min(lengths)}  max={max(lengths)}  mean={sum(lengths)//len(lengths)}")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Parent-child chunking
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: Parent-child chunking")
print("=" * 70)

parents = paragraph_chunks(HR_POLICY)         # big context blocks
# For each parent, create smaller children (sentence-level):
child_docs: list[str] = []
child_to_parent: list[int] = []
for pi, parent in enumerate(parents):
    for c in sentence_chunks(parent, max_chars=150):
        child_docs.append(c)
        child_to_parent.append(pi)

print(f"Parents (paragraph-level): {len(parents)}")
print(f"Children (sentence-level): {len(child_docs)}")

# Retrieve using children, but return the parent for context:
def parent_child_search(query: str, k: int = 1) -> list[str]:
    child_matrix = embed(child_docs)
    q = embed([query])[0]
    sims = child_matrix @ q
    top_children = np.argsort(sims)[::-1][:k]
    seen_parents = set()
    results = []
    for ci in top_children:
        pi = child_to_parent[ci]
        if pi in seen_parents:
            continue
        seen_parents.add(pi)
        results.append(parents[pi])
    return results

q = "How many personal days do I get?"
best = parent_child_search(q, k=1)
print(f"\nQuery: '{q}'")
print(f"Retrieved parent chunk:\n---\n{best[0]}\n---")


# ─────────────────────────────────────────────────────────────
# SECTION 5: BM25 keyword scoring (from scratch)
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 5: BM25 keyword scoring")
print("=" * 70)

class BM25:
    """A minimal BM25 implementation (Okapi variant)."""
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [self._tokenize(d) for d in corpus]
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in self.docs) / self.n
        # doc frequency
        self.df: dict[str, int] = {}
        for d in self.docs:
            for term in set(d):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf = {t: math.log((self.n - df + 0.5) / (df + 0.5) + 1) for t, df in self.df.items()}

    @staticmethod
    def _tokenize(s: str) -> list[str]:
        return re.findall(r"[a-z]+", s.lower())

    def score(self, query: str) -> np.ndarray:
        qterms = self._tokenize(query)
        scores = np.zeros(self.n)
        for i, doc in enumerate(self.docs):
            dl = len(doc)
            for t in qterms:
                if t not in self.idf:
                    continue
                f = doc.count(t)
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += self.idf[t] * (num / den)
        return scores


# Try BM25 alone:
bm25 = BM25(parents)
q = "personal days"
kw_scores = bm25.score(q)
print(f"BM25 scores for query '{q}':")
for i, s in enumerate(kw_scores):
    print(f"  parent[{i}]: {s:.3f}   '{parents[i][:50]}...'")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Hybrid search — weighted mix of vector + BM25
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: Hybrid search (0.6 × vector + 0.4 × BM25)")
print("=" * 70)

parent_matrix = embed(parents)

def hybrid_search(query: str, k: int = 2, alpha: float = 0.6) -> list[tuple[float, str]]:
    """alpha × vector-score + (1-alpha) × normalized-BM25."""
    v_scores = parent_matrix @ embed([query])[0]
    k_scores = bm25.score(query)

    # min-max normalize BM25 to 0-1 so it's comparable to cosine 0-1:
    if k_scores.max() > k_scores.min():
        k_norm = (k_scores - k_scores.min()) / (k_scores.max() - k_scores.min())
    else:
        k_norm = np.zeros_like(k_scores)

    hybrid = alpha * v_scores + (1 - alpha) * k_norm
    top = np.argsort(hybrid)[::-1][:k]
    return [(float(hybrid[i]), parents[i]) for i in top]

for query in ["personal days", "16 weeks", "how much PTO", "outage"]:
    print(f"\nQuery: '{query}'")
    for score, para in hybrid_search(query, k=2):
        print(f"  score {score:+.3f}  →  {para[:70]}...")


# ─────────────────────────────────────────────────────────────
# SECTION 7: Re-ranking with a cross-encoder (simulated)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7: Cross-encoder re-ranking of top-K")
print("=" * 70)

def mock_cross_encoder(query: str, doc: str) -> float:
    """A cross-encoder takes (query, doc) together and returns a relevance score.

    Real production: `sentence-transformers/ms-marco-MiniLM-L-6-v2`.
    Mock: reward exact keyword overlap + length bonus for short docs."""
    q_words = set(re.findall(r"[a-z]+", query.lower()))
    d_words = re.findall(r"[a-z]+", doc.lower())
    overlap = sum(1 for w in d_words if w in q_words)
    density = overlap / max(len(d_words), 1)
    return density

def rerank_search(query: str, initial_k: int = 5, final_k: int = 2) -> list[tuple[float, str]]:
    # Stage 1: cheap retrieval (vector) → top initial_k
    v_scores = parent_matrix @ embed([query])[0]
    initial = np.argsort(v_scores)[::-1][:initial_k]

    # Stage 2: expensive re-rank on just those K
    rescored = sorted(
        ((mock_cross_encoder(query, parents[i]), parents[i]) for i in initial),
        reverse=True,
    )
    return rescored[:final_k]

for query in ["personal days per year", "who is eligible for parental leave"]:
    print(f"\nQuery: '{query}'")
    for score, para in rerank_search(query, initial_k=5, final_k=2):
        print(f"  rerank_score {score:+.3f}  →  {para[:70]}...")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
The retrieval quality toolkit:

  1. Chunking: match strategy to content type; overlap 10-20%
  2. Parent-child: precise search, contextual return
  3. BM25: keyword baseline; fast, no embeddings needed
  4. Hybrid (weighted mix): production default, beats either alone
  5. Cross-encoder re-ranking: expensive but +10-25% accuracy on top-K

Recipe for production RAG (Phase 7):
    → Split docs (paragraph-aware, parent-child)
    → Embed once, store in ChromaDB with metadata
    → Query: hybrid vector + BM25 top-20
    → Re-rank with cross-encoder → top-5
    → Feed top-5 as context to LLM

Phase 7 (RAG) puts all of this behind a single .answer() function.
""")
