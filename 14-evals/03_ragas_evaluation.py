"""
03_ragas_evaluation.py — RAGAS metrics for RAG evaluation, from scratch
========================================================================

The 4 RAGAS metrics (implemented independently so it works in MOCK_MODE):

  1. Faithfulness         % of answer claims supported by retrieved context
  2. Answer Relevancy     Does the answer address the question?
  3. Context Precision    Are retrieved chunks relevant? (retrieval precision)
  4. Context Recall       Does retrieved context contain the answer? (retrieval recall)

Each metric is a signal to fix a specific stage:
  - Low faithfulness    → LLM hallucinating; strengthen system prompt / add citation
  - Low answer_relev.   → LLM wandering off-topic; add "answer ONLY this question"
  - Low context_prec.   → retrieval bringing noise; improve chunking / re-ranking
  - Low context_recall  → retrieval missing key info; expand top-K / chunk smaller
"""

import re
from dataclasses import dataclass


@dataclass
class RagSample:
    question: str
    answer: str
    contexts: list[str]     # retrieved chunks
    ground_truth: str       # for recall calc


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def _overlap_ratio(a: str, b: str) -> float:
    aw = set(re.findall(r"[a-z]+", a.lower()))
    bw = set(re.findall(r"[a-z]+", b.lower()))
    if not aw:
        return 0.0
    return len(aw & bw) / len(aw)


def faithfulness(sample: RagSample) -> float:
    """% of answer sentences whose content overlaps some retrieved chunk."""
    sents = _split_sentences(sample.answer)
    if not sents:
        return 0.0
    joined_ctx = " ".join(sample.contexts)
    supported = sum(1 for s in sents if _overlap_ratio(s, joined_ctx) > 0.3)
    return supported / len(sents)

def answer_relevancy(sample: RagSample) -> float:
    """How much of the question's key terms appear in the answer."""
    return _overlap_ratio(sample.question, sample.answer)

def context_precision(sample: RagSample) -> float:
    """Fraction of retrieved chunks that overlap the ground truth."""
    if not sample.contexts:
        return 0.0
    relevant = sum(1 for c in sample.contexts if _overlap_ratio(c, sample.ground_truth) > 0.2)
    return relevant / len(sample.contexts)

def context_recall(sample: RagSample) -> float:
    """Fraction of ground-truth key terms present in the retrieved context."""
    joined_ctx = " ".join(sample.contexts)
    return _overlap_ratio(sample.ground_truth, joined_ctx)


SAMPLES = [
    RagSample(
        question="How many personal days do I get?",
        answer="Employees receive 3 personal days per year, and they must be used by Dec 31.",
        contexts=[
            "In addition to PTO and sick leave, employees receive 3 personal days per year.",
            "Personal days must be used by December 31 and do not roll over.",
        ],
        ground_truth="3 personal days per year, must be used by December 31.",
    ),
    RagSample(
        question="What's the parental leave policy?",
        answer="Parental leave is 16 weeks paid.",
        contexts=[
            "New parents receive 16 weeks of paid parental leave.",
            "Full-time employees accrue PTO at 1.25 days per month.",   # noise chunk
        ],
        ground_truth="16 weeks paid parental leave, may be taken within 12 months.",
    ),
    RagSample(
        question="How much PTO do I get?",
        answer="You get unlimited PTO — take as much as you want!",   # HALLUCINATION
        contexts=[
            "Employees accrue PTO at 1.25 days per month for a total of 15 days per year.",
        ],
        ground_truth="15 days of PTO per year, accruing 1.25 days per month.",
    ),
]

if __name__ == "__main__":
    print(f"{'Sample':<8}{'Faith':<8}{'AnsRel':<8}{'CtxP':<8}{'CtxR':<8}  Notes")
    print("-" * 78)
    for i, s in enumerate(SAMPLES, 1):
        f  = faithfulness(s)
        ar = answer_relevancy(s)
        cp = context_precision(s)
        cr = context_recall(s)
        note = ""
        if f < 0.5: note = "HALLUCINATION"
        elif cp < 0.7: note = "noisy retrieval"
        elif cr < 0.5: note = "missing context"
        print(f"S{i:<7}{f:<8.2f}{ar:<8.2f}{cp:<8.2f}{cr:<8.2f}  {note}")

    print("""

✅ RAGAS interpretation:

  S1 — all high → RAG is working correctly.
  S2 — context_precision drops (noise chunk) but recall is fine.
       Fix: better re-ranking or metadata filter.
  S3 — faithfulness LOW → the LLM invented 'unlimited PTO'.
       Fix: prompt the model to answer ONLY from context, add citation
            requirement, block deploy if faithfulness < 0.9.

RAGAS gives you actionable diagnosis, not just a single score.
""")
