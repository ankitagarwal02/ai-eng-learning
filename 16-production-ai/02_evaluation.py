"""
02_evaluation.py — Production eval framework: golden set + regression harness
================================================================================

WHAT THIS FILE TEACHES
----------------------
  • Golden dataset definition and structure
  • Exact-match and contains evaluators
  • LLM-as-judge integration (mock)
  • Regression testing: new vs baseline
  • Report generation with pass rate + failures
  • A/B testing framework

HOW TO RUN
----------
    python 02_evaluation.py

REAL-WORLD SCENARIO
-------------------
Every day at 3am, run this eval on the latest prompt against a curated golden
set. If pass rate drops or any critical test regresses, PagerDuty the on-call.
"""

import os
import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import Callable, Any, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────
class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EvalCase:
    id: str
    input: Any
    expected: Any
    metadata: dict = field(default_factory=dict)
    severity: Severity = Severity.MEDIUM


@dataclass
class EvalResult:
    case: EvalCase
    actual: Any
    passed: bool
    reason: str = ""
    latency_ms: float = 0.0


@dataclass
class EvalReport:
    total: int
    passed: int
    pass_rate: float
    results: list[EvalResult]
    generated_at: str

    def failures(self) -> list[EvalResult]:
        return [r for r in self.results if not r.passed]

    def by_severity(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.results:
            sev = r.case.severity.value
            bucket = out.setdefault(sev, {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += int(r.passed)
        return out


# ─────────────────────────────────────────────────────────────
# Evaluators (composable)
# ─────────────────────────────────────────────────────────────
def evaluator_exact_match(actual: Any, expected: Any) -> tuple[bool, str]:
    return (actual == expected), f"expected={expected!r} got={actual!r}"


def evaluator_contains(actual: str, expected: str) -> tuple[bool, str]:
    return (expected.lower() in str(actual).lower()), f"expected substring {expected!r}"


def evaluator_regex(actual: str, expected_pattern: str) -> tuple[bool, str]:
    import re
    return (re.search(expected_pattern, str(actual)) is not None), f"pattern {expected_pattern}"


# LLM judge (mock — returns 0-5 score for a response)
def evaluator_llm_judge(actual: str, expected: dict) -> tuple[bool, str]:
    """expected: {"min_score": 3.5, "required_keywords": [...]}"""
    low = actual.lower()
    keywords_ok = all(k in low for k in expected.get("required_keywords", []))
    # Mock 0-5 score using keyword density
    score = 4.5 if keywords_ok else 2.0
    return (score >= expected["min_score"]), f"score={score:.1f} keywords_ok={keywords_ok}"


# ─────────────────────────────────────────────────────────────
# The system under test — swap this for your real pipeline
# ─────────────────────────────────────────────────────────────
def classifier(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("charge", "refund", "bill")):    return "BILLING"
    if any(w in t for w in ("down", "outage", "error")):     return "OUTAGE"
    if any(w in t for w in ("how do", "how to")):            return "HOWTO"
    if any(w in t for w in ("please add", "feature")):       return "FEATURE"
    if any(w in t for w in ("great", "love", "thanks")):     return "PRAISE"
    return "OTHER"


def rag_answerer(query: str) -> str:
    """Mock RAG — returns answer + citation."""
    if "sick" in query.lower():
        return "You get 10 paid sick days per year [HR-SICK]."
    if "pto" in query.lower() or "vacation" in query.lower():
        return "You get 15 PTO days per year, 5 can carry over [HR-PTO]."
    return "I don't have that information."


# ─────────────────────────────────────────────────────────────
# The runner
# ─────────────────────────────────────────────────────────────
def run_eval(
    cases: list[EvalCase],
    predict: Callable[[Any], Any],
    evaluator: Callable[[Any, Any], tuple[bool, str]] = evaluator_exact_match,
) -> EvalReport:
    import time
    results: list[EvalResult] = []
    for c in cases:
        t0 = time.perf_counter()
        actual = predict(c.input)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        passed, reason = evaluator(actual, c.expected)
        results.append(EvalResult(c, actual, passed, reason, elapsed_ms))
    passed = sum(1 for r in results if r.passed)
    return EvalReport(
        total=len(results),
        passed=passed,
        pass_rate=passed / len(results) if results else 0.0,
        results=results,
        generated_at=datetime.now().isoformat(),
    )


# ─────────────────────────────────────────────────────────────
# Regression check
# ─────────────────────────────────────────────────────────────
def regression_check(current: EvalReport, baseline_path: Path) -> dict:
    """A test that passed in baseline but FAILS now is a regression."""
    if not baseline_path.exists():
        baseline_path.write_text(json.dumps({
            "generated_at": current.generated_at,
            "pass_rate": current.pass_rate,
            "failures": [f.case.id for f in current.failures()],
        }))
        return {"new_baseline": True, "regressions": []}

    baseline = json.loads(baseline_path.read_text())
    baseline_failures = set(baseline.get("failures", []))
    current_failures = {f.case.id for f in current.failures()}
    new_failures = current_failures - baseline_failures
    fixed = baseline_failures - current_failures
    return {
        "baseline_pass_rate": baseline["pass_rate"],
        "current_pass_rate":  current.pass_rate,
        "regressions": sorted(new_failures),
        "fixed":       sorted(fixed),
    }


# ─────────────────────────────────────────────────────────────
# Quality gate
# ─────────────────────────────────────────────────────────────
def quality_gate(report: EvalReport, min_pass_rate: float = 0.95,
                 block_on_critical: bool = True) -> tuple[bool, list[str]]:
    reasons = []
    if report.pass_rate < min_pass_rate:
        reasons.append(f"pass rate {report.pass_rate:.0%} < {min_pass_rate:.0%}")
    if block_on_critical:
        crit_fail = [r for r in report.failures() if r.case.severity == Severity.CRITICAL]
        if crit_fail:
            reasons.append(f"{len(crit_fail)} CRITICAL regression(s)")
    return (len(reasons) == 0), reasons


# ─────────────────────────────────────────────────────────────
# Golden sets
# ─────────────────────────────────────────────────────────────
CLASSIFIER_CASES = [
    EvalCase("C01", "Please refund my last payment.",           "BILLING", severity=Severity.HIGH),
    EvalCase("C02", "The site is down since 3pm.",              "OUTAGE",  severity=Severity.CRITICAL),
    EvalCase("C03", "How do I export my data?",                 "HOWTO",   severity=Severity.MEDIUM),
    EvalCase("C04", "Please add dark mode.",                    "FEATURE", severity=Severity.LOW),
    EvalCase("C05", "Love this product!",                       "PRAISE",  severity=Severity.LOW),
    EvalCase("C06", "Cancel my subscription now.",              "BILLING", severity=Severity.HIGH),
    EvalCase("C07", "500 error on the API.",                    "OUTAGE",  severity=Severity.CRITICAL),
    EvalCase("C08", "Where can I find the documentation?",       "HOWTO",  severity=Severity.MEDIUM),  # may miss
]


RAG_CASES = [
    EvalCase(
        "R01",
        "How many sick days do I get?",
        {"min_score": 3.5, "required_keywords": ["10", "sick"]},
        severity=Severity.HIGH,
    ),
    EvalCase(
        "R02",
        "What's the PTO policy?",
        {"min_score": 3.5, "required_keywords": ["15", "pto"]},
        severity=Severity.HIGH,
    ),
]


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 90)
    print("EVAL — classifier (exact match)")
    print("=" * 90)
    clf_report = run_eval(CLASSIFIER_CASES, classifier, evaluator_exact_match)

    print(f"Pass rate: {clf_report.pass_rate:.0%} ({clf_report.passed}/{clf_report.total})")
    for sev, stats in clf_report.by_severity().items():
        print(f"  {sev:<10} {stats['passed']}/{stats['total']}")
    if clf_report.failures():
        print(f"\nFailures:")
        for f in clf_report.failures():
            print(f"  {f.case.id} ({f.case.severity.value}): {f.reason}")

    ok, reasons = quality_gate(clf_report, min_pass_rate=0.90)
    print(f"\nQuality gate: {'✅ PASS' if ok else '❌ FAIL — ' + '; '.join(reasons)}")

    print("\n" + "=" * 90)
    print("EVAL — RAG (LLM judge)")
    print("=" * 90)
    rag_report = run_eval(RAG_CASES, rag_answerer, evaluator_llm_judge)
    print(f"Pass rate: {rag_report.pass_rate:.0%} ({rag_report.passed}/{rag_report.total})")
    for r in rag_report.results:
        mark = "✅" if r.passed else "❌"
        print(f"  {r.case.id} {mark} — {r.actual[:60]}...   ({r.reason})")

    print("\n" + "=" * 90)
    print("REGRESSION vs baseline.json")
    print("=" * 90)
    baseline = Path("eval_baseline.json")
    reg = regression_check(clf_report, baseline)
    if reg.get("new_baseline"):
        print("  (created new baseline)")
    else:
        print(f"  baseline pass_rate: {reg['baseline_pass_rate']:.0%}")
        print(f"  current  pass_rate: {reg['current_pass_rate']:.0%}")
        print(f"  regressions:        {reg['regressions']}")
        print(f"  fixed:              {reg['fixed']}")


if __name__ == "__main__":
    main()
    print("""

✅ Summary — Production eval framework:

  • Golden set with SEVERITY labels — CRITICAL failures block deploy
  • Multiple evaluators plug into the same runner
  • Regression check: diff current failures vs stored baseline
  • Quality gate: min pass rate + no critical regressions

CI integration:
    if not quality_gate(report)[0]:
        sys.exit(1)   # blocks the merge

Run this daily against production traffic samples too (shadow eval) — that's
the earliest signal of model drift, prompt regression, or data changes.
""")
