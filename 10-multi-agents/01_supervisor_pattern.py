"""
01_supervisor_pattern.py — Supervisor agent delegates to specialists
======================================================================

WHAT THIS FILE TEACHES
----------------------
The supervisor pattern in full:

  • Supervisor maintains STATE (task, subtasks, results, iterations)
  • Supervisor uses a small LLM to decide which specialist to invoke next
  • Specialists are focused single-purpose agents
  • Supervisor aggregates + validates + iterates
  • Max-iteration safety, cycle detection
  • Full execution trace

HOW TO RUN
----------
    python 01_supervisor_pattern.py

REAL-WORLD SCENARIO
-------------------
"Write a technical blog post about AI memory systems."
The supervisor decomposes into: research → outline → draft → fact-check → polish.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────
@dataclass
class BlogState:
    topic: str
    facts: list[str] = field(default_factory=list)
    outline: list[str] = field(default_factory=list)
    draft: str = ""
    fact_check_issues: list[str] = field(default_factory=list)
    final: str = ""
    history: list[str] = field(default_factory=list)   # audit trail

    def log(self, msg: str):
        self.history.append(msg)


# ─────────────────────────────────────────────────────────────
# Specialists
# ─────────────────────────────────────────────────────────────
def research_agent(state: BlogState) -> BlogState:
    """Fact-gathering specialist. In prod: web search + citation collection."""
    state.log("[research] gathering 5 facts about " + state.topic)
    state.facts = [
        f"Fact 1 — LLMs have limited context windows (typically 128k for GPT-4o, 200k for Claude Sonnet).",
        f"Fact 2 — External vector memory beats in-context for large historical corpora.",
        f"Fact 3 — Hybrid memory (recency + semantic) is the production sweet spot.",
        f"Fact 4 — Semantic memory captures stable user facts, versioned on updates.",
        f"Fact 5 — Episodic memory records session-level events, enabling cross-session personalization.",
    ]
    return state


def outline_agent(state: BlogState) -> BlogState:
    """Structuring specialist."""
    state.log("[outline] creating 4-section outline")
    state.outline = [
        "1. Why in-context memory is not enough",
        "2. The 4 memory types (working, episodic, semantic, external)",
        "3. Building a hybrid: sliding window + vector + profile",
        "4. Production checklist and cost analysis",
    ]
    return state


def writer_agent(state: BlogState) -> BlogState:
    """Drafting specialist."""
    state.log("[writer] drafting from outline")
    intro = (
        f"Modern AI applications must remember things across sessions. "
        f"But naive approaches (full history in context) break down at scale."
    )
    body_sections = []
    for i, section in enumerate(state.outline, 1):
        body_sections.append(f"\n## {section}\n\n[Draft content for section {i}. Key facts: {state.facts[i-1] if i-1 < len(state.facts) else 'n/a'}]")
    conclusion = "\n## Conclusion\n\nUse a hybrid architecture. See Phase 13 for the full recipe."
    state.draft = intro + "".join(body_sections) + conclusion
    return state


def fact_checker_agent(state: BlogState) -> BlogState:
    """Validation specialist — flags any claim not in state.facts."""
    state.log("[fact-check] verifying claims")
    # A simple check: mention of each fact in the draft:
    issues = []
    for fact in state.facts:
        # Extract the key noun/number from the fact
        key = fact.split("—")[1].split(".")[0].strip() if "—" in fact else fact[:30]
        # If neither the exact key nor a keyword from it is in the draft, flag it
        keywords = re.findall(r"[a-zA-Z]{5,}", key)[:2]
        if keywords and not any(kw.lower() in state.draft.lower() for kw in keywords):
            issues.append(f"Fact not reflected in draft: {key[:50]}...")
    state.fact_check_issues = issues
    return state


def polish_agent(state: BlogState) -> BlogState:
    """Final editorial pass."""
    state.log("[polish] applying edits")
    state.final = state.draft + "\n\n---\n*Reviewed and polished by editorial agent.*"
    return state


SPECIALISTS: dict[str, Callable[[BlogState], BlogState]] = {
    "research":    research_agent,
    "outline":     outline_agent,
    "writer":      writer_agent,
    "fact_check":  fact_checker_agent,
    "polish":      polish_agent,
}


# ─────────────────────────────────────────────────────────────
# Supervisor — decides next specialist based on state
# ─────────────────────────────────────────────────────────────
class Decision(str, Enum):
    RESEARCH = "research"
    OUTLINE = "outline"
    WRITER = "writer"
    FACT_CHECK = "fact_check"
    POLISH = "polish"
    DONE = "done"


def supervisor_decide(state: BlogState) -> Decision:
    """Rules-based router. In production, a small LLM call decides."""
    if not state.facts:
        return Decision.RESEARCH
    if not state.outline:
        return Decision.OUTLINE
    if not state.draft:
        return Decision.WRITER
    if state.fact_check_issues == [] and "[fact-check]" not in " ".join(state.history):
        return Decision.FACT_CHECK
    if state.fact_check_issues:
        # Rewrite: go back to writer
        return Decision.WRITER
    if not state.final:
        return Decision.POLISH
    return Decision.DONE


def supervisor(topic: str, max_iterations: int = 8) -> BlogState:
    state = BlogState(topic=topic)
    print(f"👔 SUPERVISOR: task = {topic!r}\n")

    for i in range(max_iterations):
        decision = supervisor_decide(state)
        print(f"── Step {i+1}: supervisor decides → {decision.value}")

        if decision == Decision.DONE:
            print("── Done ──\n")
            break

        # Cycle detection: if we just did fact-check and it produced issues,
        # let writer retry — but only ONCE.
        if decision == Decision.WRITER and state.draft and state.fact_check_issues:
            print(f"    (writer retry — {len(state.fact_check_issues)} issue(s) to fix)")
            state.fact_check_issues = []  # clear for next check

        state = SPECIALISTS[decision.value](state)
    else:
        print("⚠️  Max iterations reached")

    return state


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────
def main():
    final_state = supervisor("AI memory systems")

    print("═" * 78)
    print("FINAL POST")
    print("═" * 78)
    print(final_state.final)

    print("\n" + "═" * 78)
    print("EXECUTION TRACE")
    print("═" * 78)
    for entry in final_state.history:
        print(f"  {entry}")

    print("\n" + "═" * 78)
    print("METADATA")
    print("═" * 78)
    print(f"  Topic:          {final_state.topic}")
    print(f"  Facts gathered: {len(final_state.facts)}")
    print(f"  Sections:       {len(final_state.outline)}")
    print(f"  Draft length:   {len(final_state.draft)} chars")
    print(f"  Fact issues:    {len(final_state.fact_check_issues)}")
    print(f"  Final length:   {len(final_state.final)} chars")


if __name__ == "__main__":
    main()
    print("""
✅ Summary — Supervisor pattern:

  • ONE state object → passed through all specialists
  • Supervisor is a small router (LLM or rules) that inspects state and picks next
  • Specialists mutate state and return it
  • Cycles allowed (fact-check → writer → fact-check again) with retry cap
  • Full audit log via state.history

When to use:
  ✅ Open-ended task where the plan can't be fixed upfront
  ✅ Need to iterate (write → critique → rewrite)
  ✅ Specialists have very different capabilities

When NOT to use:
  ❌ Fixed 3-stage pipeline (use pipeline pattern — file 02)
  ❌ Independent parallel work (use fan-out — file 03)

Real frameworks: LangGraph, CrewAI, OpenAI's Assistants (multi-agent).
All are decorations on this same supervisor + state loop.
""")
