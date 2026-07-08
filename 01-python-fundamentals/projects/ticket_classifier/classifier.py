"""
Ticket Classifier — Phase 1 Mini-Project
=========================================

WHAT THIS BUILDS
----------------
A complete ticket-classification pipeline that showcases every Phase 1 concept:

  • Load 20 realistic customer support tickets
  • Fast path: rule-based classification (keyword matching)
  • Smart path: LLM-based classification (MOCK_MODE safe)
  • Save results to results.json
  • Print summary: accuracy, category distribution, processing time

REAL-WORLD PRODUCTION PATTERN
-----------------------------
Big companies use exactly this hybrid pattern:
  1. Cheap rule-based path handles the 80% of easy cases (~$0)
  2. LLM handles the 20% of ambiguous cases ($$)
  3. Human review the 1% where LLM is uncertain
Cost savings vs "LLM for everything": ~90%.

HOW TO RUN
----------
    python classifier.py
"""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime

MOCK_MODE = not os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────────────────────
# 20 realistic tickets with ground-truth labels
# ─────────────────────────────────────────────────────────────
TICKETS = [
    # (ticket_id, text, ground_truth_category)
    (1001, "My credit card was charged twice for last month's subscription.", "BILLING"),
    (1002, "The dashboard is completely down since 3pm today.",              "OUTAGE"),
    (1003, "How do I export my project data as CSV?",                         "HOWTO"),
    (1004, "It would be great if you added dark mode to the UI.",             "FEATURE"),
    (1005, "You guys are amazing — this tool changed my workflow!",           "PRAISE"),
    (1006, "Refund request: cancelled service still being billed.",           "BILLING"),
    (1007, "500 errors when I try to load the reports page.",                 "OUTAGE"),
    (1008, "Where can I find the API documentation?",                         "HOWTO"),
    (1009, "Please consider adding SSO with Okta.",                            "FEATURE"),
    (1010, "Fantastic support, resolved my issue in 5 minutes.",              "PRAISE"),
    (1011, "Invoice for July shows the wrong amount.",                         "BILLING"),
    (1012, "Login is failing intermittently for our whole team.",              "OUTAGE"),
    (1013, "How to invite additional users to my workspace?",                  "HOWTO"),
    (1014, "Feature idea: bulk-edit for reports.",                             "FEATURE"),
    (1015, "Just want to say thanks for a great product.",                     "PRAISE"),
    (1016, "Please downgrade my plan — I'm being overcharged.",                "BILLING"),
    (1017, "Site returning 502 gateway errors.",                                "OUTAGE"),
    (1018, "What is the difference between viewer and editor roles?",          "HOWTO"),
    (1019, "Can you add webhook support for report generation?",               "FEATURE"),
    (1020, "Absolute game changer. Recommending to my whole team.",           "PRAISE"),
]

CATEGORIES = ["BILLING", "OUTAGE", "HOWTO", "FEATURE", "PRAISE"]


# ─────────────────────────────────────────────────────────────
# Data models (dataclasses from Phase 1)
# ─────────────────────────────────────────────────────────────
@dataclass
class Classification:
    ticket_id: int
    text: str
    ground_truth: str
    rule_pred: str
    llm_pred: str
    latency_ms_rule: float
    latency_ms_llm: float

    @property
    def rule_correct(self) -> bool:
        return self.rule_pred == self.ground_truth

    @property
    def llm_correct(self) -> bool:
        return self.llm_pred == self.ground_truth


# ─────────────────────────────────────────────────────────────
# Fast path — rule-based (uses list/dict, comprehensions, control flow)
# ─────────────────────────────────────────────────────────────
def classify_rules(text: str) -> str:
    """Cheap first-pass classifier using keyword rules."""
    t = text.lower()
    rules = {
        "BILLING": ["charge", "bill", "refund", "invoice", "overcharge", "downgrade"],
        "OUTAGE":  ["down", "outage", "not working", "500", "502", "error", "failing"],
        "HOWTO":   ["how do i", "how to", "where can i find", "what is", "documentation"],
        "FEATURE": ["feature", "would be great", "please add", "please consider", "idea", "add "],
        "PRAISE":  ["amazing", "great product", "thanks", "fantastic", "love", "recommend", "game changer"],
    }

    scores = {cat: sum(1 for kw in kws if kw in t) for cat, kws in rules.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "UNKNOWN"


# ─────────────────────────────────────────────────────────────
# Smart path — LLM-based (mock in MOCK_MODE, else real call)
# ─────────────────────────────────────────────────────────────
def classify_llm(text: str) -> str:
    """Simulate a more nuanced LLM classification.

    In MOCK_MODE we approximate the LLM with slightly smarter keyword logic
    that would catch the cases the rule-based path misses.
    """
    if MOCK_MODE:
        t = text.lower()
        # LLM catches paraphrases the rules miss:
        if any(w in t for w in ("charged", "billing", "credit card", "subscription", "plan")):
            return "BILLING"
        if any(w in t for w in ("down", "outage", "error", "502", "500", "login is failing")):
            return "OUTAGE"
        if any(w in t for w in ("how do i", "how to", "where", "what is", "documentation", "invite")):
            return "HOWTO"
        if any(w in t for w in ("feature", "add", "consider", "webhook", "sso")):
            return "FEATURE"
        if any(w in t for w in ("thanks", "great", "love", "amazing", "fantastic", "recommend", "game changer")):
            return "PRAISE"
        return "UNKNOWN"

    # Real LLM path (unreachable in MOCK_MODE):
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Classify into one of: {', '.join(CATEGORIES)}. Answer with the label only."},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"  LLM error, falling back to rules: {e}")
        return classify_rules(text)


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def run() -> None:
    print("=" * 70)
    print("TICKET CLASSIFIER — Phase 1 Mini-Project")
    print("=" * 70)
    print(f"MOCK_MODE: {MOCK_MODE}")
    print(f"Tickets:   {len(TICKETS)}")
    print()

    results: list[Classification] = []
    total_start = time.perf_counter()

    for tid, text, gt in TICKETS:
        t0 = time.perf_counter()
        rp = classify_rules(text)
        t1 = time.perf_counter()
        lp = classify_llm(text)
        t2 = time.perf_counter()

        c = Classification(
            ticket_id=tid,
            text=text,
            ground_truth=gt,
            rule_pred=rp,
            llm_pred=lp,
            latency_ms_rule=round((t1 - t0) * 1000, 3),
            latency_ms_llm=round((t2 - t1) * 1000, 3),
        )
        results.append(c)

        rule_mark = "✅" if c.rule_correct else "❌"
        llm_mark = "✅" if c.llm_correct else "❌"
        print(f"  #{tid}  gt={gt:<8}  rule={rp:<8} {rule_mark}   llm={lp:<8} {llm_mark}   \"{text[:45]}...\"")

    total_elapsed = time.perf_counter() - total_start

    # Metrics:
    n = len(results)
    rule_acc = sum(1 for r in results if r.rule_correct) / n
    llm_acc = sum(1 for r in results if r.llm_correct) / n

    print("\n" + "=" * 70)
    print("Accuracy")
    print("=" * 70)
    print(f"  Rule-based accuracy: {rule_acc:.0%}   ({sum(1 for r in results if r.rule_correct)}/{n})")
    print(f"  LLM       accuracy:  {llm_acc:.0%}   ({sum(1 for r in results if r.llm_correct)}/{n})")

    # Per-category breakdown:
    print("\nPer-category accuracy (LLM):")
    for cat in CATEGORIES:
        cat_items = [r for r in results if r.ground_truth == cat]
        if cat_items:
            acc = sum(1 for r in cat_items if r.llm_correct) / len(cat_items)
            print(f"  {cat:<10} {acc:.0%}  ({len(cat_items)} tickets)")

    # Save results:
    out_path = Path(__file__).parent / "results.json"
    payload = {
        "run_id": datetime.now().strftime("run_%Y%m%d_%H%M%S"),
        "mock_mode": MOCK_MODE,
        "n_tickets": n,
        "rule_accuracy": rule_acc,
        "llm_accuracy": llm_acc,
        "total_elapsed_s": round(total_elapsed, 3),
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n📄 Results saved to: {out_path}")

    # Cost calc — production-style output:
    llm_calls = n
    est_tokens_per_call = 40   # small prompt + short answer
    price_per_1M_input = 0.15
    est_cost = llm_calls * est_tokens_per_call * price_per_1M_input / 1_000_000
    print(f"\n💰 Cost estimate (gpt-4o-mini prices): ${est_cost:.6f} for {llm_calls} calls")

    print("\n" + "=" * 70)
    print("✅ Summary — What this project demonstrated")
    print("=" * 70)
    print("""
  • Dataclasses for the Classification record (Phase 1 §3)
  • List/dict comprehensions for scoring rules (§1-2)
  • Timing with time.perf_counter (§4 async parallel — same primitive)
  • Exception handling for real LLM fallback (§6)
  • f-string formatting for aligned tabular output (§7)
  • Path + JSON serialization for results (§8)
  • Real-world hybrid pattern: cheap rules + expensive LLM

You are ready for Phase 2 — AI/ML Basics.
""")


if __name__ == "__main__":
    run()
