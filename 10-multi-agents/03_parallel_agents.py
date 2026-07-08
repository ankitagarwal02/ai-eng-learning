"""
03_parallel_agents.py — Fan-out / Fan-in with async.gather + conflict resolution
================================================================================

WHAT THIS FILE TEACHES
----------------------
  • True parallel agent execution via asyncio.gather
  • return_exceptions=True for partial failure handling
  • Merger agent aggregating N results into one report
  • Conflict resolution (majority vote, weighted, supervisor tiebreak)
  • Timing comparison: serial vs parallel

HOW TO RUN
----------
    python 03_parallel_agents.py

REAL-WORLD SCENARIO
-------------------
"Analyze this contract." Three specialists run in PARALLEL:
   LegalRiskAgent (flags legal issues)
   FinancialAgent (extracts money terms)
   ComplianceAgent (checks regulatory requirements)
Results merged into one report by an Aggregator.

Bonus: what to do when specialists disagree on the risk level.
"""

import os
import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# Specialist agents (all async)
# ─────────────────────────────────────────────────────────────
@dataclass
class AgentResult:
    agent: str
    risk_level: str          # LOW / MEDIUM / HIGH / CRITICAL
    findings: list[str]
    latency_ms: float
    error: Optional[str] = None


async def legal_risk_agent(contract: str) -> AgentResult:
    start = time.perf_counter()
    await asyncio.sleep(random.uniform(0.4, 0.7))    # simulate LLM call

    findings = []
    low = contract.lower()
    if "auto-renew" in low or "automatic renewal" in low:
        findings.append("Auto-renewal clause without required notice")
    if "any jurisdiction" in low or "ambiguous jurisdiction" in low:
        findings.append("Jurisdiction clause is ambiguous")
    if "indemnif" in low and "unlimited" in low:
        findings.append("Unlimited indemnification")

    risk = "HIGH" if len(findings) >= 2 else "MEDIUM" if findings else "LOW"
    return AgentResult(
        agent="legal",
        risk_level=risk,
        findings=findings,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


async def financial_agent(contract: str) -> AgentResult:
    start = time.perf_counter()
    await asyncio.sleep(random.uniform(0.3, 0.6))

    findings = []
    low = contract.lower()
    if "$50" in contract or "50k" in low or "50,000" in contract:
        findings.append("Annual value: $50k")
    if "penalt" in low:
        findings.append("Penalty clause identified")
    if "late fee" in low:
        findings.append("Late-fee clause: standard")

    risk = "MEDIUM" if "penalt" in low else "LOW"
    return AgentResult(
        agent="financial",
        risk_level=risk,
        findings=findings,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


async def compliance_agent(contract: str) -> AgentResult:
    start = time.perf_counter()
    await asyncio.sleep(random.uniform(0.5, 0.8))

    findings = []
    low = contract.lower()
    gdpr_ok = "gdpr" in low or "data protection" in low
    if not gdpr_ok:
        findings.append("Missing GDPR / data protection clause")
    if "share" in low and "third part" in low and "consent" not in low:
        findings.append("Third-party sharing without explicit consent term")

    risk = "CRITICAL" if len(findings) >= 2 else ("MEDIUM" if findings else "LOW")
    return AgentResult(
        agent="compliance",
        risk_level=risk,
        findings=findings,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


# One that sometimes fails, to demonstrate partial-failure handling:
async def flaky_agent(contract: str) -> AgentResult:
    await asyncio.sleep(0.1)
    if random.random() < 0.5:
        raise RuntimeError("Flaky provider timeout")
    return AgentResult(agent="flaky", risk_level="LOW", findings=[], latency_ms=100)


# ─────────────────────────────────────────────────────────────
# Merger / aggregator with conflict resolution
# ─────────────────────────────────────────────────────────────
RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def merge_results(results: list[AgentResult]) -> dict:
    """Combine specialists' findings.  Conflict resolution: take the STRICTEST
    risk level (max), collect all findings, note if there was disagreement."""
    all_findings = []
    risks = []
    for r in results:
        if r.error:
            continue
        for f in r.findings:
            all_findings.append({"agent": r.agent, "finding": f})
        risks.append(r.risk_level)

    if not risks:
        return {"risk_level": "UNKNOWN", "error": "all specialists failed"}

    # Strictest wins:
    max_risk = max(risks, key=lambda r: RISK_ORDER.index(r))
    disagreement = len(set(risks)) > 1

    return {
        "risk_level": max_risk,
        "specialists_ran": [r.agent for r in results if not r.error],
        "specialists_failed": [r.agent for r in results if r.error],
        "individual_risks": {r.agent: r.risk_level for r in results if not r.error},
        "all_findings": all_findings,
        "disagreement": disagreement,
        "recommendation": _recommendation(max_risk, disagreement),
    }


def _recommendation(risk: str, disagreement: bool) -> str:
    if risk == "CRITICAL":
        return "REJECT — critical compliance issue; requires legal review."
    if risk == "HIGH":
        return "NEGOTIATE — significant issues found; renegotiate flagged clauses."
    if disagreement:
        return f"REVIEW — specialists disagree; risk={risk} but human tiebreak advised."
    if risk == "MEDIUM":
        return "PROCEED WITH CAUTION — minor issues, note and monitor."
    return "APPROVE — no material issues found."


# ─────────────────────────────────────────────────────────────
# Serial vs parallel demonstration
# ─────────────────────────────────────────────────────────────
async def analyze_serial(contract: str) -> dict:
    t0 = time.perf_counter()
    results = []
    for agent in [legal_risk_agent, financial_agent, compliance_agent]:
        results.append(await agent(contract))
    elapsed = (time.perf_counter() - t0) * 1000
    return {"results": results, "wall_ms": elapsed}


async def analyze_parallel(contract: str) -> dict:
    t0 = time.perf_counter()
    results = await asyncio.gather(
        legal_risk_agent(contract),
        financial_agent(contract),
        compliance_agent(contract),
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return {"results": results, "wall_ms": elapsed}


async def analyze_with_failures(contract: str) -> dict:
    """Show partial-failure handling."""
    t0 = time.perf_counter()
    raw = await asyncio.gather(
        legal_risk_agent(contract),
        flaky_agent(contract),
        compliance_agent(contract),
        return_exceptions=True,
    )
    # Convert exceptions to failed results:
    results = []
    for r in raw:
        if isinstance(r, Exception):
            results.append(AgentResult(
                agent="unknown", risk_level="LOW", findings=[],
                latency_ms=0, error=f"{type(r).__name__}: {r}"))
        else:
            results.append(r)
    elapsed = (time.perf_counter() - t0) * 1000
    return {"results": results, "wall_ms": elapsed}


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────
SAMPLE_CONTRACT = """
Service Agreement between Widget Co and Acme Corp.

Payment terms: $50,000 annually, auto-renewal after 12 months unless notified
30 days prior. Late fees apply at 5% per month overdue.

Jurisdiction: any jurisdiction as chosen by Widget Co. Compliance: parties
agree to share necessary data with third parties for service delivery.
Indemnification: Widget Co indemnifies Acme against unlimited claims.

Term: 3 years, automatic renewal, no penalty for early termination beyond
outstanding invoices.
"""


async def main():
    print("═" * 78)
    print("SERIAL execution")
    print("═" * 78)
    serial = await analyze_serial(SAMPLE_CONTRACT)
    for r in serial["results"]:
        print(f"  {r.agent:<12} risk={r.risk_level:<10} findings={len(r.findings)}  "
              f"latency={r.latency_ms:.0f}ms")
    print(f"  ─── wall time: {serial['wall_ms']:.0f}ms ───")

    print("\n" + "═" * 78)
    print("PARALLEL execution (asyncio.gather)")
    print("═" * 78)
    parallel = await analyze_parallel(SAMPLE_CONTRACT)
    for r in parallel["results"]:
        print(f"  {r.agent:<12} risk={r.risk_level:<10} findings={len(r.findings)}  "
              f"latency={r.latency_ms:.0f}ms")
    print(f"  ─── wall time: {parallel['wall_ms']:.0f}ms ───")
    speedup = serial["wall_ms"] / parallel["wall_ms"]
    print(f"  ─── speedup: {speedup:.1f}× ───")

    print("\n" + "═" * 78)
    print("MERGED REPORT")
    print("═" * 78)
    report = merge_results(parallel["results"])
    print(f"  Risk level:      {report['risk_level']}")
    print(f"  Ran:             {report['specialists_ran']}")
    print(f"  Failed:          {report['specialists_failed']}")
    print(f"  Individual risks: {report['individual_risks']}")
    print(f"  Disagreement:    {report['disagreement']}")
    print(f"  Recommendation:  {report['recommendation']}")
    print(f"\n  Findings:")
    for f in report["all_findings"]:
        print(f"    [{f['agent']}] {f['finding']}")

    print("\n" + "═" * 78)
    print("PARTIAL FAILURE (one agent flakes)")
    print("═" * 78)
    for _ in range(3):
        fail_run = await analyze_with_failures(SAMPLE_CONTRACT)
        errors = [r for r in fail_run["results"] if r.error]
        oks = [r for r in fail_run["results"] if not r.error]
        print(f"  ok={len(oks)}  errors={len(errors)}   "
              f"error msgs: {[r.error for r in errors]}")


if __name__ == "__main__":
    asyncio.run(main())
    print("""
✅ Summary — Parallel pattern:

  • asyncio.gather() launches N agents concurrently
  • return_exceptions=True prevents ONE failure from killing the whole batch
  • Merger handles conflict resolution:
      – Strictest wins (max risk)
      – Majority vote
      – Weighted by agent confidence
      – Supervisor LLM tiebreak
  • Wall-clock speedup = N (for I/O-bound), but token cost is the same

Rule of thumb: any time 2+ agents can run independently, run them in parallel.
Latency drops from sum-of-agents to max-of-agents. Zero token cost.
""")
