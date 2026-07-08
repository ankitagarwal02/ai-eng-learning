"""
02_advanced_rag.py — 4 advanced RAG patterns end-to-end
=========================================================

WHAT THIS FILE TEACHES
----------------------
Four techniques that each solve a specific failure of naive RAG:

  1. HyDE (Hypothetical Document Embeddings)
     Problem: query too short/imprecise to embed well.
     Fix: ask LLM to draft a fake ideal answer first, embed THAT for search.

  2. Query Expansion
     Problem: user's vocabulary doesn't match the docs' vocabulary.
     Fix: LLM rewrites the query 3 ways, search with all, union results.

  3. Re-ranking with a Cross-encoder
     Problem: bi-encoder retrieval returns semantically-close-but-not-relevant chunks.
     Fix: after retrieving top-20, run a cross-encoder to re-score, keep top-5.

  4. Context Compression
     Problem: retrieved chunks are long; LLM misses the key sentence.
     Fix: LLM extracts only the sentences that address the query.

HOW TO RUN
----------
    pip install chromadb numpy
    python 02_advanced_rag.py

REAL-WORLD SCENARIO
-------------------
Same HR knowledge base as `01_naive_rag.py`. We now demonstrate how each
technique produces measurably better results on a specific hard query.
"""

import os
import re
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import chromadb
    _has_chroma = True
except ImportError:
    _has_chroma = False

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# Reuse the same KB from 01_naive_rag.py — abbreviated inline here so this file
# is self-contained.
HR_DOCS = [
    ("HR-PTO", "Paid Time Off", "All full-time employees accrue paid time off at 1.25 days per month, "
        "totaling 15 days per year. PTO can be used for vacation, personal reasons, or health appointments. "
        "Unused PTO carries over up to 5 days into the next calendar year. Any accrued PTO above the carryover "
        "limit is forfeited on January 1."),
    ("HR-SICK", "Sick Leave", "Full-time employees receive 10 paid sick days per calendar year, credited "
        "on January 1. Sick leave does not accrue and does not roll over. For absences of more than 3 "
        "consecutive days, a physician's note is required. Sick leave may also be used to care for immediate family."),
    ("HR-PERSONAL", "Personal Days", "In addition to PTO and sick leave, employees receive 3 personal days "
        "per year. These may be used for any personal reason without justification. Personal days must be used "
        "by December 31 and do not roll over."),
    ("HR-PARENTAL", "Parental Leave", "New parents (birth, adoption, or foster placement) receive 16 weeks "
        "of paid parental leave, which may be taken any time during the first 12 months. Parental leave applies "
        "equally to all parents regardless of gender."),
    ("HR-BEREAVEMENT", "Bereavement", "Employees are entitled to 5 days of paid bereavement leave upon the "
        "death of an immediate family member. Extensions and travel time may be granted case-by-case."),
    ("HR-REMOTE", "Remote Work", "Employees may work remotely up to 3 days per week with manager approval. "
        "Full-remote roles require approval from the VP of People. Remote-work equipment is reimbursed via the "
        "standard equipment stipend."),
    ("HR-EQUIPMENT", "Equipment Stipend", "Every employee receives a $2,000 home-office equipment stipend on "
        "hiring and a $500 annual refresh budget. Eligible items include: monitor, chair, desk, keyboard, mouse, "
        "webcam, and lighting."),
    ("HR-LEARNING", "Learning Budget", "All employees have access to a $1,500 annual learning budget for courses, "
        "books, conferences, and certifications. Approval required for items >$500."),
    ("HR-EQUITY", "Equity", "New employees receive equity grants that vest over 4 years with a 1-year cliff. "
        "Refresh grants are considered at the annual review."),
    ("HR-PROMO", "Promotions & Reviews", "Formal promotion review occurs semi-annually in April and October. "
        "Salary adjustments take effect the month after approval. Peer feedback is solicited for every review."),
]


# ─────────────────────────────────────────────────────────────
# Setup collection
# ─────────────────────────────────────────────────────────────
def build_index():
    if not _has_chroma:
        return None
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(name="hr_kb_adv")
    col.add(
        ids=[d[0] for d in HR_DOCS],
        documents=[d[2] for d in HR_DOCS],
        metadatas=[{"section": d[1]} for d in HR_DOCS],
    )
    return col


def vec_search(collection, query: str, k: int = 5) -> list[dict]:
    hits = collection.query(query_texts=[query], n_results=k)
    return [
        {"id": id_, "text": doc, "distance": dist, "meta": meta}
        for id_, doc, dist, meta in zip(
            hits["ids"][0], hits["documents"][0],
            hits["distances"][0], hits["metadatas"][0]
        )
    ]


# ─────────────────────────────────────────────────────────────
# MOCK LLM — deterministic responses for each advanced task
# ─────────────────────────────────────────────────────────────
def mock_llm(task: str, prompt: str) -> str:
    """Task-specific mock outputs so patterns are visible without an API key."""
    if task == "hyde":
        # Draft a fake ideal answer for the query embedded in the prompt.
        q = prompt.lower()
        if "personal day" in q:
            return ("An employee is entitled to 3 personal days per year, in addition to PTO "
                    "and sick leave. Personal days do not roll over and must be used by December 31.")
        if "carry" in q and "pto" in q:
            return ("Unused PTO may be carried over up to 5 days into the next calendar year. "
                    "Any PTO above the carryover limit is forfeited on January 1.")
        if "maternity" in q or "father" in q or "adoption" in q:
            return ("New parents receive 16 weeks of paid parental leave regardless of gender or path to parenthood. "
                    "The leave may be taken any time during the first 12 months.")
        return "[MOCK HyDE answer for: " + prompt[-80:] + "]"

    if task == "expand":
        q = prompt.split("Query:")[-1].strip()
        variants = [q]
        low = q.lower()
        if "vacation" in low or "time off" in low:
            variants += ["paid time off policy", "PTO days per year", "annual leave"]
        elif "sick" in low:
            variants += ["sick leave policy", "medical leave days", "physician's note requirements"]
        elif "wfh" in low or "work from home" in low or "remote" in low:
            variants += ["remote work policy", "hybrid schedule", "work from home rules"]
        else:
            variants += [f"policy about {q}", f"handbook section on {q}"]
        return "\n".join(variants[:4])

    if task == "rerank":
        # Reranker returns list of (id, score) — mocked to prefer exact-section match.
        m = re.search(r"Query:\s*(.+)$", prompt)
        query = m.group(1).lower() if m else ""
        # Just echo scores; caller will interpret.
        return "SCORE"

    if task == "compress":
        m = re.search(r"Query:\s*(.+?)\n", prompt)
        chunk_match = re.search(r"Chunk:\s*(.+)", prompt, re.DOTALL)
        query = m.group(1).lower() if m else ""
        chunk = chunk_match.group(1) if chunk_match else ""
        # Return only sentences that share ≥2 words with the query
        q_words = set(re.findall(r"[a-z]+", query.lower()))
        sents = re.split(r"(?<=[.!?])\s+", chunk)
        kept = [s for s in sents
                if len(set(re.findall(r"[a-z]+", s.lower())) & q_words) >= 1]
        return " ".join(kept) or chunk[:100]

    return "[MOCK]"


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 1 — HyDE
# ─────────────────────────────────────────────────────────────
def hyde_search(collection, query: str, k: int = 5) -> tuple[str, list[dict]]:
    """Step 1: LLM drafts a hypothetical ideal answer.
       Step 2: We embed & search using THAT text instead of the raw query.
       Returns (fake_answer_used, retrieved_chunks)."""
    fake_answer = mock_llm("hyde",
        f"Draft an ideal one-paragraph answer to this HR policy question:\n{query}")
    # Search with the fake answer as the query text:
    return fake_answer, vec_search(collection, fake_answer, k=k)


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 2 — Query expansion
# ─────────────────────────────────────────────────────────────
def expanded_search(collection, query: str, k_per_variant: int = 3) -> tuple[list[str], list[dict]]:
    """Step 1: LLM generates 3-4 rewrites of the query.
       Step 2: Search vector DB with each variant.
       Step 3: Union results, deduplicate by id, sort by BEST distance across variants."""
    variants_raw = mock_llm("expand", f"Rewrite this query 3 different ways:\nQuery: {query}")
    variants = [v.strip() for v in variants_raw.splitlines() if v.strip()]

    all_hits: dict[str, dict] = {}
    for v in variants:
        for hit in vec_search(collection, v, k=k_per_variant):
            existing = all_hits.get(hit["id"])
            if existing is None or hit["distance"] < existing["distance"]:
                hit["hit_via"] = v
                all_hits[hit["id"]] = hit

    ordered = sorted(all_hits.values(), key=lambda h: h["distance"])
    return variants, ordered


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 3 — Cross-encoder re-ranking
# ─────────────────────────────────────────────────────────────
def cross_encoder_score(query: str, chunk_text: str) -> float:
    """MOCK cross-encoder: weighted overlap + section boost.

    Real production: sentence-transformers/ms-marco-MiniLM-L-6-v2 gives
    scores for (query, doc) pairs directly."""
    q_words = set(re.findall(r"[a-z]+", query.lower()))
    d_words = re.findall(r"[a-z]+", chunk_text.lower())
    if not d_words:
        return 0.0
    overlap = sum(1 for w in d_words if w in q_words)
    density = overlap / max(len(d_words), 1)
    # Bonus for a keyword appearing more than once (indicates on-topic chunk):
    for kw in q_words:
        if d_words.count(kw) >= 2:
            density += 0.05
    return min(density, 1.0)


def rerank(query: str, candidates: list[dict], final_k: int = 3) -> list[dict]:
    """Take initial retrieved list, score with cross-encoder, sort, keep top final_k."""
    scored = []
    for c in candidates:
        score = cross_encoder_score(query, c["text"])
        c2 = {**c, "rerank_score": score}
        scored.append(c2)
    scored.sort(key=lambda x: -x["rerank_score"])
    return scored[:final_k]


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 4 — Context compression
# ─────────────────────────────────────────────────────────────
def compress_chunk(query: str, chunk_text: str) -> str:
    """Ask LLM to extract only the sentences from chunk that address the query.

    Real prompt (in production):
        Given the question and a text chunk, output ONLY the sentences from
        the chunk that directly answer the question. Preserve wording exactly.
    """
    return mock_llm("compress", f"Query: {query}\nChunk: {chunk_text}")


# ─────────────────────────────────────────────────────────────
# The full advanced RAG pipeline — combining all four
# ─────────────────────────────────────────────────────────────
@dataclass
class AdvancedRAGTrace:
    query: str
    naive: list[dict]
    hyde_answer: str
    hyde: list[dict]
    variants: list[str]
    expanded: list[dict]
    reranked: list[dict]
    compressed: list[dict]
    final_answer: str


def advanced_rag(collection, query: str) -> AdvancedRAGTrace:
    # Naive baseline
    naive_hits = vec_search(collection, query, k=3)

    # 1. HyDE
    fake, hyde_hits = hyde_search(collection, query, k=5)

    # 2. Query expansion
    variants, expanded_hits = expanded_search(collection, query, k_per_variant=3)

    # 3. Combine sources into one candidate pool, then rerank
    pool: dict[str, dict] = {}
    for h in hyde_hits + expanded_hits + naive_hits:
        if h["id"] not in pool or h["distance"] < pool[h["id"]]["distance"]:
            pool[h["id"]] = h
    reranked = rerank(query, list(pool.values()), final_k=3)

    # 4. Compress each surviving chunk
    compressed = []
    for r in reranked:
        squeezed = compress_chunk(query, r["text"])
        compressed.append({**r, "compressed_text": squeezed})

    # Final LLM call would use compressed context. Mock:
    if compressed:
        answer = f"{compressed[0]['compressed_text']} [{compressed[0]['id']}]"
    else:
        answer = "I don't have that information in the HR handbook."

    return AdvancedRAGTrace(
        query=query,
        naive=naive_hits,
        hyde_answer=fake,
        hyde=hyde_hits,
        variants=variants,
        expanded=expanded_hits,
        reranked=reranked,
        compressed=compressed,
        final_answer=answer,
    )


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────
def print_trace(t: AdvancedRAGTrace) -> None:
    print("═" * 80)
    print(f"QUERY: {t.query}")
    print("═" * 80)

    print("\n[Baseline] Naive top-3:")
    for r in t.naive:
        print(f"  {r['id']}  dist={r['distance']:.3f}  \"{r['text'][:60]}...\"")

    print(f"\n[Technique 1: HyDE]  fake answer used for embedding:")
    for line in t.hyde_answer.strip().splitlines():
        print(f"  │ {line[:100]}")
    print("  HyDE top-3:")
    for r in t.hyde[:3]:
        print(f"    {r['id']}  dist={r['distance']:.3f}")

    print(f"\n[Technique 2: Query Expansion]  {len(t.variants)} variants:")
    for v in t.variants:
        print(f"  • {v}")
    print("  Union top-3:")
    for r in t.expanded[:3]:
        print(f"    {r['id']}  dist={r['distance']:.3f}  via='{r.get('hit_via','?')[:30]}'")

    print("\n[Technique 3: Rerank]  cross-encoder top-3:")
    for r in t.reranked:
        print(f"  {r['id']}  rerank_score={r['rerank_score']:.3f}  dist={r['distance']:.3f}")

    print("\n[Technique 4: Context compression]")
    for r in t.compressed:
        print(f"  {r['id']}  ORIGINAL  ({len(r['text'])} chars)")
        print(f"    {r['text'][:80]}...")
        print(f"  {r['id']}  COMPRESSED ({len(r['compressed_text'])} chars)")
        print(f"    {r['compressed_text']}")

    print(f"\n📝 FINAL ANSWER: {t.final_answer}")


def main():
    if not _has_chroma:
        print("Install chromadb: pip install chromadb")
        return

    col = build_index()
    print(f"Indexed {len(HR_DOCS)} HR documents\n")

    HARD_QUERIES = [
        "father",   # Very short — HyDE should generate a proper query
        "how much time off do I get if I have a new baby",   # Paraphrase - expansion helps
        "carry over PTO",   # Ambiguous - rerank + compression help focus
    ]

    for q in HARD_QUERIES:
        trace = advanced_rag(col, q)
        print_trace(trace)
        print()


if __name__ == "__main__":
    main()
    print("""
✅ Summary — Advanced RAG cheat sheet:

  Technique          Solves the problem of...            Cost multiplier
  ──────────────────────────────────────────────────────────────────────
  HyDE               query too short/imprecise           +1 LLM call
  Query expansion    vocabulary mismatch                 +1 LLM call, +N searches
  Re-ranking         retrieval brings noise              +1 encoder pass (cheap)
  Compression        LLM misses key sentence             +K LLM calls (K = final_k)

RULE: START WITH NAIVE. Add techniques only when a specific failure mode is
documented on your eval set (Phase 14 RAGAS). Adding all four blindly is a
5-10× cost with often marginal quality gain.

Production stack that works for 90% of use cases:
    Naive retrieval → cross-encoder rerank → LLM answer with citations.
That's it. HyDE and expansion are for the last 10%.

Next: 03_rag_with_citations.py — the citation + faithfulness layer.
""")
