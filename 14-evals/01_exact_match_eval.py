"""
01_exact_match_eval.py — Structured eval harness for classification
====================================================================

EvalCase / EvalReport dataclasses. Runs 15 labeled test cases through
a classifier. Produces a report with per-category breakdown and failures.

SCENARIO
--------
Ticket classifier from Phase 1. Eval on 15 held-out labeled tickets.
Show pass rate by category, which category has lowest accuracy, and
recommendations.
"""

from dataclasses import dataclass, field, asdict
from typing import Callable, Any
import json


@dataclass
class EvalCase:
    id: str
    input: Any
    expected: Any
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    case: EvalCase
    actual: Any
    passed: bool


@dataclass
class EvalReport:
    total: int
    passed: int
    failed: int
    results: list[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0

    def by_metadata(self, key: str) -> dict[str, dict]:
        buckets: dict[str, dict] = {}
        for r in self.results:
            bucket = r.case.metadata.get(key, "unknown")
            b = buckets.setdefault(bucket, {"passed": 0, "total": 0})
            b["total"] += 1
            b["passed"] += int(r.passed)
        return buckets

    def print_summary(self):
        print(f"\nOverall: {self.passed}/{self.total} = {self.pass_rate:.0%}")
        print("\nBy category:")
        for cat, stats in self.by_metadata("category").items():
            print(f"  {cat:<12} {stats['passed']}/{stats['total']}")
        failures = [r for r in self.results if not r.passed]
        if failures:
            print(f"\nFailures ({len(failures)}):")
            for r in failures:
                print(f"  {r.case.id}: expected {r.case.expected!r}  got {r.actual!r}")


def exact_match(a, b) -> bool:
    return a == b


def contains(a: str, b: str) -> bool:
    return b.lower() in a.lower()


def run_eval(cases: list[EvalCase], predict: Callable[[Any], Any],
             match_fn: Callable = exact_match) -> EvalReport:
    results = []
    for c in cases:
        actual = predict(c.input)
        passed = match_fn(actual, c.expected)
        results.append(EvalResult(c, actual, passed))
    return EvalReport(
        total=len(cases),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        results=results,
    )


# ─────────────────────────────────────────────────────────────
# The system under test — a tiny classifier
# ─────────────────────────────────────────────────────────────
def classify(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("charge", "refund", "bill")):    return "BILLING"
    if any(w in t for w in ("down", "outage", "error")):     return "OUTAGE"
    if any(w in t for w in ("how do", "how to")):            return "HOWTO"
    if any(w in t for w in ("please add", "feature")):       return "FEATURE"
    if any(w in t for w in ("great", "love", "thanks")):     return "PRAISE"
    return "OTHER"


CASES = [
    EvalCase("T01", "Please refund my last payment.",           "BILLING",  {"category": "BILLING"}),
    EvalCase("T02", "My card was charged twice.",                "BILLING",  {"category": "BILLING"}),
    EvalCase("T03", "The bill for June looks wrong.",            "BILLING",  {"category": "BILLING"}),
    EvalCase("T04", "The site is down since 3pm.",               "OUTAGE",   {"category": "OUTAGE"}),
    EvalCase("T05", "Getting 500 errors on API.",                "OUTAGE",   {"category": "OUTAGE"}),
    EvalCase("T06", "Dashboard outage since morning.",           "OUTAGE",   {"category": "OUTAGE"}),
    EvalCase("T07", "How do I export my data?",                  "HOWTO",    {"category": "HOWTO"}),
    EvalCase("T08", "How to invite users?",                      "HOWTO",    {"category": "HOWTO"}),
    EvalCase("T09", "Where can I find docs?",                    "HOWTO",    {"category": "HOWTO"}),  # will miss
    EvalCase("T10", "Please add dark mode.",                     "FEATURE",  {"category": "FEATURE"}),
    EvalCase("T11", "Feature request: SSO.",                     "FEATURE",  {"category": "FEATURE"}),
    EvalCase("T12", "Would love a bulk import.",                 "FEATURE",  {"category": "FEATURE"}),  # will miss
    EvalCase("T13", "Great product, thanks!",                    "PRAISE",   {"category": "PRAISE"}),
    EvalCase("T14", "You guys are amazing.",                     "PRAISE",   {"category": "PRAISE"}),   # will miss
    EvalCase("T15", "Love the new update.",                      "PRAISE",   {"category": "PRAISE"}),
]

if __name__ == "__main__":
    report = run_eval(CASES, classify)
    report.print_summary()

    print("""

✅ Eval framework — the shape of every regression suite:

  EvalCase + EvalReport dataclasses (portable, testable)
  Metadata for slicing metrics by category, difficulty, etc.
  Multiple match functions: exact_match, contains, regex, precision/recall

Recommendations from this run:
  - HOWTO recall is weak — 'where can I find docs?' misses (no 'how do/to' pattern)
  - PRAISE recall — 'amazing' isn't in the pattern list
  - Fix: add these keywords, re-run, ensure pass_rate stays ≥ old
""")
