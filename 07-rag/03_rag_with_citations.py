"""
03_rag_with_citations.py — Citation tracking + faithfulness scoring
====================================================================

WHAT THIS FILE TEACHES
----------------------
Production RAG must be answerable to this question:
    "Where in my docs is that fact from?"

We build:
  • A prompt template that REQUIRES [SOURCE-ID] citations on every claim
  • A faithfulness scorer that flags un-cited sentences
  • An unsupported-claim detector that catches hallucinations
  • A quality gate: if faithfulness < 0.9, block the response

HOW TO RUN
----------
    python 03_rag_with_citations.py

REAL-WORLD SCENARIO
-------------------
The HR bot is deployed to 5000 employees. A bad answer about parental leave
could get you sued. Every claim MUST cite the handbook section it came from,
AND that section must actually support the claim.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# The citation-required prompt template
# ─────────────────────────────────────────────────────────────
CITATION_SYSTEM_PROMPT = """\
You are Aria, the HR assistant. You must follow these rules WITHOUT EXCEPTION:

  1. Answer using ONLY facts from the CONTEXT below.
  2. Every sentence must end with a citation in the form [SOURCE-ID].
  3. If multiple sources support a sentence, cite all: [SOURCE-1][SOURCE-2].
  4. If the answer is not in the context, respond exactly:
     "I don't have that information in the HR handbook. Please contact HR directly."
  5. Never state a fact without a citation. Never invent citations.

Example format:
    "Employees receive 15 days of PTO per year [HR-PTO]. Personal days are separate — 3 per year [HR-PERSONAL]."
"""


CITATION_RE = re.compile(r"\[([A-Z0-9\-#\.]+)\]")


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────
@dataclass
class ChunkCitation:
    id: str
    text: str
    used: bool = False   # whether it was cited in the answer


@dataclass
class Sentence:
    text: str
    citations: list[str]           # IDs found in [brackets]
    supported: bool = False        # is any cited chunk relevant to this sentence?
    confidence: float = 0.0        # 0..1 how strongly supported


@dataclass
class GroundingReport:
    query: str
    answer: str
    sentences: list[Sentence]
    unique_citations: list[str]
    unused_chunks: list[str]     # chunks retrieved but never cited
    unknown_citations: list[str] # cited IDs that weren't in the retrieved set
    faithfulness: float          # % of sentences supported
    citation_coverage: float     # % of sentences with any citation
    quality_pass: bool


# ─────────────────────────────────────────────────────────────
# The core: score every sentence
# ─────────────────────────────────────────────────────────────
def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _support_score(sentence: str, chunk_text: str) -> float:
    """Simple lexical overlap score. Real prod: use an NLI model or LLM judge.

    Returns 0..1 — higher means chunk supports the sentence.
    """
    s_words = _tokenize(sentence) - {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are"}
    c_words = _tokenize(chunk_text)
    if not s_words:
        return 0.0
    overlap = len(s_words & c_words) / len(s_words)
    return overlap


def score_grounding(query: str, answer: str, chunks: list[dict],
                    support_threshold: float = 0.4,
                    faithfulness_gate: float = 0.9) -> GroundingReport:
    """Analyze the answer against the chunks that were used to produce it.

    chunks: [{'id': str, 'text': str}, ...]  the retrieved chunks
    """
    chunk_map = {c["id"]: c["text"] for c in chunks}
    known_ids = set(chunk_map)

    # Split answer into sentences (naive but sufficient for demo):
    raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    sentences: list[Sentence] = []
    all_cited: list[str] = []
    unknown: list[str] = []

    for raw in raw_sents:
        cites = CITATION_RE.findall(raw)
        # Strip citations before assessing content:
        content = CITATION_RE.sub("", raw).strip()

        # Check each citation exists in the retrieved set:
        for c in cites:
            all_cited.append(c)
            if c not in known_ids:
                unknown.append(c)

        # Compute support: max overlap of content with any CITED chunk:
        best = 0.0
        for cid in cites:
            if cid in chunk_map:
                score = _support_score(content, chunk_map[cid])
                if score > best:
                    best = score

        sentences.append(Sentence(
            text=raw,
            citations=cites,
            supported=(best >= support_threshold),
            confidence=best,
        ))

    if sentences:
        with_citations = sum(1 for s in sentences if s.citations)
        supported = sum(1 for s in sentences if s.supported)
        citation_coverage = with_citations / len(sentences)
        faithfulness = supported / len(sentences)
    else:
        citation_coverage = 0.0
        faithfulness = 0.0

    unused = [cid for cid in known_ids if cid not in all_cited]

    return GroundingReport(
        query=query,
        answer=answer,
        sentences=sentences,
        unique_citations=sorted(set(all_cited)),
        unused_chunks=unused,
        unknown_citations=sorted(set(unknown)),
        faithfulness=faithfulness,
        citation_coverage=citation_coverage,
        quality_pass=(faithfulness >= faithfulness_gate and not unknown),
    )


# ─────────────────────────────────────────────────────────────
# Print a human-readable report
# ─────────────────────────────────────────────────────────────
def print_report(r: GroundingReport) -> None:
    print("═" * 80)
    print(f"QUERY:  {r.query}")
    print(f"ANSWER: {r.answer}")
    print("═" * 80)

    print("\n── Sentence-by-sentence grounding ──")
    for i, s in enumerate(r.sentences, 1):
        mark = "✅" if s.supported else ("❓" if s.citations else "❌")
        cite_str = "[" + ",".join(s.citations) + "]" if s.citations else "(no citation!)"
        print(f"  {i}. {mark}  conf={s.confidence:.2f}  {cite_str}")
        print(f"      {s.text}")

    print("\n── Citations used ──")
    print(f"  Unique cited IDs:      {r.unique_citations}")
    if r.unknown_citations:
        print(f"  ⚠️  UNKNOWN citations:  {r.unknown_citations}  ← hallucinated IDs!")
    if r.unused_chunks:
        print(f"  Unused retrieved:      {r.unused_chunks}  (retrieved but not cited)")

    print("\n── Scores ──")
    print(f"  Citation coverage:  {r.citation_coverage:.0%}  (sentences with ANY citation)")
    print(f"  Faithfulness:       {r.faithfulness:.0%}  (sentences supported by cited chunk)")
    print(f"  Quality gate:       {'✅ PASS' if r.quality_pass else '❌ FAIL'}")
    if not r.quality_pass:
        reasons = []
        if r.faithfulness < 0.9:
            reasons.append(f"faithfulness {r.faithfulness:.0%} < 90%")
        if r.unknown_citations:
            reasons.append(f"{len(r.unknown_citations)} unknown citation(s)")
        print(f"                      Reasons: {'; '.join(reasons)}")


# ─────────────────────────────────────────────────────────────
# Test cases: good answers, bad answers, obvious hallucinations
# ─────────────────────────────────────────────────────────────
TEST_CHUNKS = [
    {"id": "HR-PTO",
     "text": "All full-time employees accrue paid time off at 1.25 days per month, totaling 15 days per year. "
             "Unused PTO carries over up to 5 days into the next calendar year."},
    {"id": "HR-SICK",
     "text": "Full-time employees receive 10 paid sick days per calendar year, credited on January 1."},
    {"id": "HR-PERSONAL",
     "text": "Employees receive 3 personal days per year. Personal days must be used by December 31 "
             "and do not roll over."},
]


TEST_CASES = [
    ("How many days off do I get per year?",
     "Employees get 15 days of PTO per year [HR-PTO]. They also receive 10 sick days [HR-SICK] "
     "and 3 personal days [HR-PERSONAL]."),

    ("How many days off do I get per year?",   # answer with missing citation
     "You get 15 days of PTO per year [HR-PTO]. You also get sick days and personal days."),

    ("How much PTO do I get?",   # hallucination — invented "unlimited PTO"
     "You get unlimited PTO — take as much as you want! [HR-PTO]"),

    ("What's the parental leave policy?",   # fake citation
     "You get 16 weeks paid parental leave [HR-PARENTAL]."),

    ("How many personal days do I get?",   # perfect
     "Employees receive 3 personal days per year, and they must be used by December 31 [HR-PERSONAL]."),
]


def main():
    for query, answer in TEST_CASES:
        report = score_grounding(query, answer, TEST_CHUNKS)
        print_report(report)
        print()


if __name__ == "__main__":
    main()
    print("""
✅ Summary — Citations & Faithfulness:

For every production RAG answer, run score_grounding() and use it as a gate:

  quality_pass = faithfulness >= 0.9 AND no unknown citations

Common failure signatures the report catches:
  • Citation coverage < 100%   → LLM didn't cite every sentence
                                 Fix: strengthen system prompt with example.
  • Faithfulness < 90%         → LLM cited a chunk but stated something not in it
                                 Fix: strengthen 'answer ONLY from context' rule
                                       OR the chunk was missing the info
                                       (bad retrieval — go fix your chunker).
  • Unknown citations         → LLM invented an ID
                                 Fix: refuse the response, retry. If persistent,
                                       reduce temperature or switch model.

Real production adds:
  • An LLM-judge to verify support (better than lexical overlap)
  • A retry loop that feeds the failure back to the LLM as feedback
  • Live monitoring: track faithfulness over time; alert on drops

Full RAG stack:
  Phase 6 (retrieval)  + Phase 7 (RAG + citations)
+ Phase 14 (evals: RAGAS metrics on golden set)
+ Phase 15 (guardrails: PII, output moderation)
+ Phase 16 (observability: log every retrieved chunk with the answer)
""")
