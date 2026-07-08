"""
01_naive_rag.py — Minimum-viable RAG, end-to-end, production-shape
====================================================================

WHAT THIS FILE TEACHES
----------------------
The complete naive RAG pipeline — every part of it — so you can spot bugs at
each stage in production. When your production RAG breaks, one of these seven
steps is failing:

  1. Document loading                (missing docs, wrong parser)
  2. Chunking                        (chunks too big / small / wrong boundary)
  3. Embedding                       (batch size, retries, cost)
  4. Vector DB storage               (ID collisions, missing metadata)
  5. Query embedding                 (must use SAME model as docs!)
  6. Retrieval                       (top-K, distance threshold, filter)
  7. Prompt assembly & LLM call      (context ordering, "lost in the middle")

HOW TO RUN
----------
    pip install chromadb numpy
    python 01_naive_rag.py

REAL-WORLD SCENARIO
-------------------
Company HR knowledge base: 10 policy paragraphs. Employees ask questions.
We show every intermediate value so you can see EXACTLY what the LLM receives
and why. If retrieval is bad → the answer is bad, no matter how smart the LLM.
"""

import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import chromadb
    _has_chroma = True
except ImportError:
    _has_chroma = False

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# SECTION 1: The Knowledge Base
# ─────────────────────────────────────────────────────────────
# In production this comes from S3, SharePoint, Confluence, etc.
# Here it's a hand-curated list — each doc has: id, source, section, text.

@dataclass
class Document:
    id: str
    source: str        # e.g. "handbook-v3.pdf"
    section: str       # e.g. "PTO Policy"
    text: str
    meta: dict = field(default_factory=dict)


KB: list[Document] = [
    Document("HR-PTO", "handbook-v3.pdf", "Paid Time Off",
             "All full-time employees accrue paid time off at 1.25 days per month, "
             "totaling 15 days per year. PTO can be used for vacation, personal reasons, "
             "or health appointments. Unused PTO carries over up to 5 days into the next "
             "calendar year. Any accrued PTO above the carryover limit is forfeited "
             "on January 1."),
    Document("HR-SICK", "handbook-v3.pdf", "Sick Leave",
             "Full-time employees receive 10 paid sick days per calendar year, credited "
             "on January 1. Sick leave does not accrue and does not roll over. For absences "
             "of more than 3 consecutive days, a physician's note is required. Sick leave "
             "may also be used to care for immediate family members."),
    Document("HR-PERSONAL", "handbook-v3.pdf", "Personal Days",
             "In addition to PTO and sick leave, employees receive 3 personal days per "
             "year. These may be used for any personal reason without justification. "
             "Personal days must be used by December 31 and do not roll over."),
    Document("HR-PARENTAL", "handbook-v3.pdf", "Parental Leave",
             "New parents (birth, adoption, or foster placement) receive 16 weeks of paid "
             "parental leave, which may be taken any time during the first 12 months. "
             "Additional unpaid leave may be requested through HR. Parental leave applies "
             "equally to all parents regardless of gender."),
    Document("HR-BEREAVEMENT", "handbook-v3.pdf", "Bereavement",
             "Employees are entitled to 5 days of paid bereavement leave upon the death "
             "of an immediate family member. Extensions and travel time may be granted "
             "case-by-case."),
    Document("HR-REMOTE", "handbook-v3.pdf", "Remote Work",
             "Employees may work remotely up to 3 days per week with manager approval. "
             "Full-remote roles require approval from the VP of People. Remote-work "
             "equipment is reimbursed via the standard equipment stipend."),
    Document("HR-EQUIPMENT", "handbook-v3.pdf", "Equipment Stipend",
             "Every employee receives a $2,000 home-office equipment stipend on hiring "
             "and a $500 annual refresh budget. Eligible items include: monitor, chair, "
             "desk, keyboard, mouse, webcam, and lighting."),
    Document("HR-LEARNING", "handbook-v3.pdf", "Learning Budget",
             "All employees have access to a $1,500 annual learning budget for courses, "
             "books, conferences, and certifications. Approval required for items >$500."),
    Document("HR-EQUITY", "handbook-v3.pdf", "Equity",
             "New employees receive equity grants that vest over 4 years with a 1-year "
             "cliff. Refresh grants are considered at the annual review."),
    Document("HR-PROMO", "handbook-v3.pdf", "Promotions & Reviews",
             "Formal promotion review occurs semi-annually in April and October. Salary "
             "adjustments take effect the month after approval. Peer feedback is solicited "
             "for every review."),
]


# ─────────────────────────────────────────────────────────────
# SECTION 2: Chunking (with visible strategy)
# ─────────────────────────────────────────────────────────────
# Each doc is 1 paragraph so it fits in a chunk. For real docs, use the
# strategies from `06-vector-databases/04_vector_db_patterns.py`.

@dataclass
class Chunk:
    id: str             # "HR-PTO#0" — doc id + local chunk index
    doc_id: str
    text: str
    source: str
    section: str


def paragraph_chunker(doc: Document, max_chars: int = 800) -> list[Chunk]:
    """One chunk per paragraph, split further if any paragraph is too big."""
    paras = re.split(r"\n\s*\n", doc.text.strip())
    chunks: list[Chunk] = []
    for i, p in enumerate(paras):
        if len(p) <= max_chars:
            chunks.append(Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=p.strip(),
                                source=doc.source, section=doc.section))
        else:
            # If still too big, sentence-split it. See Phase 6 §chunking.
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) > max_chars and buf:
                    chunks.append(Chunk(id=f"{doc.id}#{i}.{len(chunks)}", doc_id=doc.id,
                                        text=buf.strip(), source=doc.source, section=doc.section))
                    buf = s
                else:
                    buf = (buf + " " + s).strip()
            if buf:
                chunks.append(Chunk(id=f"{doc.id}#{i}.{len(chunks)}", doc_id=doc.id,
                                    text=buf.strip(), source=doc.source, section=doc.section))
    return chunks


# ─────────────────────────────────────────────────────────────
# SECTION 3: Build the vector index
# ─────────────────────────────────────────────────────────────
def build_index(kb: list[Document]):
    """Chunk → embed → store in ChromaDB."""
    if not _has_chroma:
        return None, []

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="hr_kb")

    all_chunks: list[Chunk] = []
    for doc in kb:
        all_chunks.extend(paragraph_chunker(doc))

    # Chroma auto-embeds via its default sentence-transformer model. In prod
    # you'd pass an EmbeddingFunction that calls OpenAI or your own provider.
    collection.add(
        ids=[c.id for c in all_chunks],
        documents=[c.text for c in all_chunks],
        metadatas=[{"doc_id": c.doc_id, "source": c.source, "section": c.section}
                   for c in all_chunks],
    )
    return collection, all_chunks


# ─────────────────────────────────────────────────────────────
# SECTION 4: The RAG function (show every intermediate value!)
# ─────────────────────────────────────────────────────────────
@dataclass
class RAGTrace:
    """Every step of a RAG run — invaluable for debugging in production."""
    query: str
    n_retrieved: int
    retrieved: list[dict]
    context_block: str
    system_prompt: str
    user_prompt: str
    prompt_token_estimate: int
    answer: str
    total_latency_ms: float
    retrieval_latency_ms: float
    llm_latency_ms: float


SYSTEM_PROMPT = """\
You are Aria, the HR assistant. Answer using ONLY the context provided below.

Rules:
  1. If the answer is not in the context, respond exactly:
     "I don't have that information in the HR handbook. Please contact HR directly."
  2. For every fact you state, include the [SOURCE-ID] in brackets.
  3. Keep responses concise (2-4 sentences).
  4. Never invent policies, numbers, or dates.
"""


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def rag_answer(collection, query: str, k: int = 3,
               distance_threshold: Optional[float] = None) -> RAGTrace:
    """The naive-RAG function, with every intermediate value captured."""
    total_start = time.perf_counter()

    # ---- Step 1: retrieve
    retr_start = time.perf_counter()
    hits = collection.query(query_texts=[query], n_results=k)
    retrieval_ms = (time.perf_counter() - retr_start) * 1000

    retrieved = []
    for id_, doc, dist, meta in zip(hits["ids"][0], hits["documents"][0],
                                     hits["distances"][0], hits["metadatas"][0]):
        if distance_threshold is not None and dist > distance_threshold:
            continue   # too dissimilar — drop
        retrieved.append({"id": id_, "distance": dist, "text": doc, "meta": meta})

    # ---- Step 2: assemble prompt
    context_block = "\n\n".join(
        f"[{r['id']}] (section: {r['meta']['section']})\n{r['text']}"
        for r in retrieved
    ) or "(no context retrieved)"

    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}"
    prompt_tokens = approx_tokens(SYSTEM_PROMPT + user_prompt)

    # ---- Step 3: LLM call (mock or real)
    llm_start = time.perf_counter()
    answer = _call_llm(query, retrieved)
    llm_ms = (time.perf_counter() - llm_start) * 1000

    return RAGTrace(
        query=query,
        n_retrieved=len(retrieved),
        retrieved=retrieved,
        context_block=context_block,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        prompt_token_estimate=prompt_tokens,
        answer=answer,
        total_latency_ms=(time.perf_counter() - total_start) * 1000,
        retrieval_latency_ms=retrieval_ms,
        llm_latency_ms=llm_ms,
    )


def _call_llm(query: str, retrieved: list[dict]) -> str:
    """MOCK_MODE: produce a deterministic answer that mimics grounded generation."""
    if not retrieved:
        return ("I don't have that information in the HR handbook. "
                "Please contact HR directly.")

    if MOCK_MODE:
        # Compose an answer from the top chunk + relevant sentence extraction.
        top = retrieved[0]
        text = top["text"]
        # Find the most relevant sentence (keyword overlap):
        q_words = set(re.findall(r"[a-z]+", query.lower()))
        best_sent, best_score = "", 0
        for sent in re.split(r"(?<=[.!?])\s+", text):
            s_words = set(re.findall(r"[a-z]+", sent.lower()))
            score = len(q_words & s_words)
            if score > best_score:
                best_sent, best_score = sent, score
        if not best_sent:
            best_sent = text[:120] + "..."
        return f"{best_sent} [{top['id']}]"

    # Real OpenAI path:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n" + "\n\n".join(
                f"[{r['id']}] {r['text']}" for r in retrieved
            ) + f"\n\nQuestion: {query}"},
        ],
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# SECTION 5: Full run + trace inspection
# ─────────────────────────────────────────────────────────────
def print_trace(t: RAGTrace) -> None:
    print("═" * 80)
    print(f"QUERY: {t.query}")
    print("═" * 80)

    print("\n── STEP 1: Retrieval ─────────────────────────────────")
    print(f"  Chunks retrieved: {t.n_retrieved}  (latency: {t.retrieval_latency_ms:.1f}ms)")
    for i, r in enumerate(t.retrieved, 1):
        print(f"  #{i}  id={r['id']}  distance={r['distance']:.3f}  section={r['meta']['section']}")
        print(f"       {r['text'][:100]}...")

    print("\n── STEP 2: Prompt assembly ───────────────────────────")
    print(f"  Estimated tokens: {t.prompt_token_estimate}")
    print(f"  Context length:   {len(t.context_block)} chars")
    print("  ─── System prompt ───")
    for line in t.system_prompt.strip().splitlines():
        print(f"  │ {line}")
    print("  ─── User prompt (truncated) ───")
    for line in t.user_prompt[:400].splitlines():
        print(f"  │ {line}")
    print("  │ ...")

    print("\n── STEP 3: LLM answer ────────────────────────────────")
    print(f"  Latency: {t.llm_latency_ms:.1f}ms")
    print(f"  Answer:  {t.answer}")

    print(f"\nTOTAL LATENCY: {t.total_latency_ms:.1f}ms")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Failure modes gallery
# ─────────────────────────────────────────────────────────────
FAILURE_MODE_QUERIES = [
    ("How many sick days am I entitled to?",
     "Happy path — clean retrieval of HR-SICK, clean answer."),
    ("Do you offer paid maternity leave?",
     "Semantic paraphrase — no 'maternity' in KB but 'parental leave' should be found."),
    ("What is the office holiday schedule?",
     "Out-of-scope — nothing in KB. Should refuse gracefully."),
    ("How much PTO can I roll over into next year?",
     "Multi-sentence answer needed — the '5 days carryover' is a specific sub-fact."),
    ("What's the meaning of life?",
     "Completely off-topic — should refuse."),
]


def main():
    if not _has_chroma:
        print("Install chromadb: pip install chromadb")
        return

    print("Building HR knowledge base index...")
    collection, chunks = build_index(KB)
    print(f"  {len(chunks)} chunks indexed from {len(KB)} documents\n")

    for query, why in FAILURE_MODE_QUERIES:
        trace = rag_answer(collection, query, k=3)
        print_trace(trace)
        print(f"\n💡 Test rationale: {why}\n")


if __name__ == "__main__":
    main()
    print("""
✅ Summary — Naive RAG anatomy:

Every RAG system, no matter how fancy, has this exact skeleton:
    query → embed → search → assemble prompt → LLM → answer

When a production RAG fails, it fails at ONE of these steps. The trace object
lets you diagnose immediately:
    • Empty retrieved list  → embedding model mismatch OR bad chunking
    • High distances        → query wording differs too much from docs
    • Right chunks, wrong answer → system prompt too weak; LLM ignoring context
    • Chunks contain answer but LLM says "I don't know" → "lost in the middle"

Advanced patterns (next file 02_advanced_rag.py) each address a specific
failure mode:
    • HyDE          → query too short/imprecise
    • Query expansion → too much vocabulary mismatch
    • Re-ranking    → retrieval bringing noise
    • Compression   → context too long, LLM missing key sentence

Next: 02_advanced_rag.py — all four techniques implemented end-to-end.
""")
