"""
02_llm_as_judge.py — LLM-as-Judge with rubrics + pairwise + calibration
=========================================================================

WHAT THIS FILE TEACHES
----------------------
  • Why exact-match fails for open-ended outputs
  • Rubric-based scoring: 5 dimensions, score 1-5 each
  • Multi-dimensional scoring: helpfulness, accuracy, safety, conciseness, tone
  • Pairwise evaluation: which of A/B is better
  • Judge calibration: agreement with human ratings
  • Quality gate: block deploy if avg score < 3.5

HOW TO RUN
----------
    python 02_llm_as_judge.py

REAL-WORLD SCENARIO
-------------------
Compare gpt-4o vs gpt-4o-mini on 5 customer support responses. Show dimension-
by-dimension breakdown and pick a winner. Bonus: show judge calibration.
"""

import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# The rubric
# ─────────────────────────────────────────────────────────────
RUBRIC = {
    "helpfulness": {
        1: "Doesn't address the question at all",
        2: "Barely relevant / mostly misses the point",
        3: "Partially addresses; some gaps",
        4: "Fully addresses the question",
        5: "Fully addresses AND anticipates follow-up needs",
    },
    "accuracy": {
        1: "Factually wrong",
        2: "Some inaccuracies",
        3: "Mostly accurate, minor issues",
        4: "Fully accurate",
        5: "Fully accurate with citations",
    },
    "safety": {
        1: "Contains harmful / toxic content or PII leak",
        2: "Risky tone or borderline content",
        3: "Neutral / no safety issues",
        4: "Safe and considerate",
        5: "Safe, considerate, escalates appropriately",
    },
    "conciseness": {
        1: "Excessively long, meanders",
        2: "Overly verbose",
        3: "Reasonable length",
        4: "Concise",
        5: "Perfect length — nothing wasted",
    },
    "tone": {
        1: "Rude / dismissive",
        2: "Cold / robotic",
        3: "Professional",
        4: "Warm and professional",
        5: "Warm, empathetic, appropriate to context",
    },
}


# ─────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────
@dataclass
class Response:
    model: str
    text: str

@dataclass
class Score:
    dimension: str
    score: int          # 1-5
    reason: str

@dataclass
class Evaluation:
    query: str
    response: Response
    scores: list[Score]
    total: float

    @property
    def by_dimension(self) -> dict[str, int]:
        return {s.dimension: s.score for s in self.scores}


# ─────────────────────────────────────────────────────────────
# The judge (mock — deterministic heuristics)
# ─────────────────────────────────────────────────────────────
def judge_response(query: str, response: Response) -> Evaluation:
    """MOCK judge — inspects text and applies heuristics for each dimension."""
    text = response.text
    low = text.lower()
    scores: list[Score] = []

    # helpfulness: does it address the query keywords?
    q_words = set(re.findall(r"[a-z]{3,}", query.lower()))
    d_words = set(re.findall(r"[a-z]{3,}", low))
    overlap = len(q_words & d_words) / max(len(q_words), 1)
    if overlap > 0.5:
        scores.append(Score("helpfulness", 5, "addresses all key terms in query"))
    elif overlap > 0.3:
        scores.append(Score("helpfulness", 4, "addresses most of the query"))
    elif overlap > 0.15:
        scores.append(Score("helpfulness", 3, "partial coverage"))
    else:
        scores.append(Score("helpfulness", 2, "off-topic or generic"))

    # accuracy: presence of citations & specific numbers/names
    has_cite = bool(re.search(r"\[[A-Z0-9\-]+\]", text))
    has_specifics = bool(re.search(r"\b\d+\b|\bpolicy\b|\bhandbook\b", text))
    if has_cite and has_specifics:
        scores.append(Score("accuracy", 5, "specific & cited"))
    elif has_specifics:
        scores.append(Score("accuracy", 4, "specific claims"))
    elif len(text) > 40:
        scores.append(Score("accuracy", 3, "generic but plausible"))
    else:
        scores.append(Score("accuracy", 2, "vague"))

    # safety
    bad = any(w in low for w in ("stupid", "you're wrong", "shut up"))
    good = any(w in low for w in ("please contact", "happy to help", "let me"))
    scores.append(Score("safety",
                        1 if bad else (5 if good else 4),
                        "toxic phrasing" if bad else "safe"))

    # conciseness
    n_chars = len(text)
    if n_chars < 30:
        scores.append(Score("conciseness", 2, "too short — likely uninformative"))
    elif n_chars < 200:
        scores.append(Score("conciseness", 5, "concise and tight"))
    elif n_chars < 400:
        scores.append(Score("conciseness", 4, "reasonable length"))
    elif n_chars < 700:
        scores.append(Score("conciseness", 3, "a bit long"))
    else:
        scores.append(Score("conciseness", 2, "verbose"))

    # tone
    warm = any(w in low for w in ("thanks", "happy to", "understand", "sorry to hear", "let me help"))
    cold = any(w in low for w in ("processed", "input received", "acknowledged"))
    if warm:
        scores.append(Score("tone", 5, "warm and empathetic"))
    elif cold:
        scores.append(Score("tone", 2, "robotic"))
    else:
        scores.append(Score("tone", 3, "neutral professional"))

    total = statistics.mean(s.score for s in scores)
    return Evaluation(query=query, response=response, scores=scores, total=total)


# ─────────────────────────────────────────────────────────────
# Pairwise judge — A vs B, pick winner
# ─────────────────────────────────────────────────────────────
def pairwise_judge(query: str, a: Response, b: Response) -> dict:
    """Ask judge to score both; winner = higher total. Ties broken by helpfulness."""
    eval_a = judge_response(query, a)
    eval_b = judge_response(query, b)

    if eval_a.total > eval_b.total:
        winner = a.model
    elif eval_b.total > eval_a.total:
        winner = b.model
    else:
        # tie — helpfulness wins:
        winner = a.model if eval_a.by_dimension["helpfulness"] > eval_b.by_dimension["helpfulness"] else b.model

    return {
        "winner": winner,
        "a_total": eval_a.total,
        "b_total": eval_b.total,
        "detail": {"a": eval_a.by_dimension, "b": eval_b.by_dimension},
    }


# ─────────────────────────────────────────────────────────────
# Test cases: 5 real support queries + 2 responses each
# ─────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "query": "How many sick days do I get per year?",
        "gpt-4o":       "You receive 10 paid sick days per calendar year, credited on January 1. For absences over 3 days, a physician's note is required. [HR-SICK]",
        "gpt-4o-mini":  "10 sick days per year.",
    },
    {
        "query": "My laptop won't turn on and I have a demo in 30 minutes.",
        "gpt-4o":       "I understand this is urgent. Let me help you troubleshoot right away — first, please try holding the power button for 10 seconds. If that doesn't work, plug into charger for 5 minutes and try again. I've also opened a P1 ticket for IT.",
        "gpt-4o-mini":  "Input received. Try turning it off and on.",
    },
    {
        "query": "I want to cancel my subscription.",
        "gpt-4o":       "I'm sorry to hear that. You can cancel from Billing → Subscription → Cancel. Your access continues until the end of the billing period. Anything I can help address first?",
        "gpt-4o-mini":  "Go to billing settings to cancel.",
    },
    {
        "query": "What's your parental leave policy?",
        "gpt-4o":       "New parents receive 16 weeks of paid parental leave, regardless of gender or path to parenthood, taken any time in the first 12 months. [HR-PARENTAL]",
        "gpt-4o-mini":  "16 weeks paid parental leave for new parents.",
    },
    {
        "query": "You're being useless — just tell me the answer.",
        "gpt-4o":       "I understand the frustration and want to help. Please share more about what you're trying to accomplish and I'll get you a specific answer.",
        "gpt-4o-mini":  "Please rephrase your question.",
    },
]


# ─────────────────────────────────────────────────────────────
# Run pairwise evaluation
# ─────────────────────────────────────────────────────────────
def main():
    print("═" * 90)
    print("PAIRWISE EVAL — gpt-4o vs gpt-4o-mini on 5 support responses")
    print("═" * 90)

    winners = []
    for i, case in enumerate(TEST_CASES, 1):
        a = Response("gpt-4o", case["gpt-4o"])
        b = Response("gpt-4o-mini", case["gpt-4o-mini"])
        result = pairwise_judge(case["query"], a, b)

        print(f"\nCase {i}: {case['query']}")
        print(f"  A (gpt-4o):      total={result['a_total']:.2f}  {result['detail']['a']}")
        print(f"  B (gpt-4o-mini): total={result['b_total']:.2f}  {result['detail']['b']}")
        print(f"  Winner: {result['winner']}")
        winners.append(result["winner"])

    # Aggregate
    tally = Counter(winners)
    print("\n" + "═" * 90)
    print("OVERALL")
    print("═" * 90)
    print(f"  Wins: {dict(tally)}")

    # Multi-dimensional per-model
    print("\n" + "═" * 90)
    print("PER-DIMENSION averages")
    print("═" * 90)
    per_model = {"gpt-4o": [], "gpt-4o-mini": []}
    for case in TEST_CASES:
        for model_key in per_model:
            per_model[model_key].append(judge_response(case["query"], Response(model_key, case[model_key])))

    dims = list(RUBRIC.keys())
    print(f"{'Model':<15}" + "".join(f"{d:<14}" for d in dims) + "AVG")
    for model_key, evals in per_model.items():
        row = f"{model_key:<15}"
        for d in dims:
            avg = statistics.mean(e.by_dimension[d] for e in evals)
            row += f"{avg:<14.2f}"
        overall = statistics.mean(e.total for e in evals)
        row += f"{overall:.2f}"
        print(row)

    # Quality gate
    print("\n" + "═" * 90)
    print("QUALITY GATE")
    print("═" * 90)
    for model_key, evals in per_model.items():
        avg = statistics.mean(e.total for e in evals)
        gate = "✅ PASS" if avg >= 3.5 else "❌ FAIL"
        print(f"  {model_key}: avg={avg:.2f}  →  {gate}  (threshold 3.5)")


# ─────────────────────────────────────────────────────────────
# Bonus: judge calibration
# ─────────────────────────────────────────────────────────────
def calibration_check():
    print("\n" + "═" * 90)
    print("JUDGE CALIBRATION vs human ratings (mocked)")
    print("═" * 90)
    print("""
Every 100 productions runs, sample 10 outputs and have humans rate them on
the same rubric. Compute agreement:

  Cohen's kappa (0..1):
    < 0.4  → judge is unreliable; use human review only
    0.4-0.6 → moderate agreement; use judge as SIGNAL, not gate
    0.6-0.8 → substantial agreement; judge as gate for medium-stakes
    > 0.8  → near-human; safe as automated gate

If kappa drops over time → judge drift; recalibrate the rubric prompt.
""")


if __name__ == "__main__":
    main()
    calibration_check()
    print("""
✅ Summary — LLM-as-Judge:

  • Rubric with 5 dimensions gives DIAGNOSABLE scores
    (not just "3.7/5", but "helpful=5, accuracy=3, safety=5")
  • Pairwise A/B is more reliable than absolute scoring
  • Cost: 1 extra LLM call per response evaluated
  • Calibrate quarterly against human ratings to detect judge drift

Combined stack:
  Phase 14 §01 (exact match)  — for classification/extraction
  Phase 14 §02 (LLM judge)    — for open-ended generation
  Phase 14 §03 (RAGAS)         — for RAG-specific metrics

All three run in CI. Block deploy if any regresses.
""")
