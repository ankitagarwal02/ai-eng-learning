"""
01_data_types_and_control.py — Python data structures + control flow for AI Engineers
======================================================================================

WHAT THIS FILE TEACHES
----------------------
• Lists, dicts, sets, tuples with AI-relevant real-world uses (support tickets, configs).
• List / dict comprehensions — the compact idiom you'll see in every AI codebase.
• Control flow: if/elif/else, for, while, break, continue.
• Exception handling with try/except/finally — gracefully handling an API failure.
• f-strings and string formatting for prompt templates.
• File I/O: JSON config + CSV + text.

HOW TO RUN
----------
    python 01_data_types_and_control.py

No API key required — MOCK_MODE is on.

REAL-WORLD SCENARIO
-------------------
You are a support engineer at a SaaS company. Ten customer tickets just arrived
in your queue. You will:
    1. Store and index them.
    2. Filter by priority.
    3. Count by category.
    4. Compute average resolution time.
    5. Save a JSON report to disk.
    6. Simulate one API call failing and recover gracefully.
"""

import os
import json
from pathlib import Path
from datetime import datetime

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Lists — the workhorse
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Lists — the workhorse")
print("=" * 70)

# 10 realistic support tickets — a list of dicts, the most common shape in AI apps.
tickets = [
    {"id": 1001, "priority": "high",   "category": "billing",   "hours_open": 2.5},
    {"id": 1002, "priority": "low",    "category": "feature",   "hours_open": 48.0},
    {"id": 1003, "priority": "high",   "category": "outage",    "hours_open": 0.5},
    {"id": 1004, "priority": "medium", "category": "billing",   "hours_open": 8.0},
    {"id": 1005, "priority": "low",    "category": "docs",      "hours_open": 72.0},
    {"id": 1006, "priority": "high",   "category": "outage",    "hours_open": 1.2},
    {"id": 1007, "priority": "medium", "category": "account",   "hours_open": 14.0},
    {"id": 1008, "priority": "high",   "category": "billing",   "hours_open": 3.0},
    {"id": 1009, "priority": "low",    "category": "feature",   "hours_open": 90.0},
    {"id": 1010, "priority": "medium", "category": "docs",      "hours_open": 24.0},
]

print(f"Total tickets ingested: {len(tickets)}")
print(f"First ticket: {tickets[0]}")
print(f"Last ticket:  {tickets[-1]}")  # Negative indexing


# ─────────────────────────────────────────────────────────────
# SECTION 2: List comprehensions — the idiom
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: List comprehensions")
print("=" * 70)

# Bad — imperative:
high_pri_slow_way = []
for t in tickets:
    if t["priority"] == "high":
        high_pri_slow_way.append(t)

# Good — Pythonic:
high_pri = [t for t in tickets if t["priority"] == "high"]
print(f"High-priority tickets ({len(high_pri)}):")
for t in high_pri:
    print(f"  #{t['id']:>5}  {t['category']:<10}  open {t['hours_open']:>5.1f}h")

# Extract just the IDs — this pattern is called "projection":
open_ticket_ids = [t["id"] for t in tickets if t["hours_open"] > 24]
print(f"\nTickets open >24h: {open_ticket_ids}")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Dicts — the config / lookup workhorse
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: Dicts — config + lookups")
print("=" * 70)

# LLM configuration — every AI app has this pattern:
llm_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "max_tokens": 1024,
    "system_prompt": "You are a triage assistant.",
    "retry": {"max_attempts": 3, "backoff": 2.0},
}
print(f"Model: {llm_config['model']}")
print(f"Retry backoff: {llm_config['retry']['backoff']}s")

# .get() with default — safer than [key] because it can't raise KeyError:
top_p = llm_config.get("top_p", 1.0)  # not defined -> uses default
print(f"top_p (default): {top_p}")

# Dict comprehension — count tickets per category:
category_counts = {}
for t in tickets:
    category_counts[t["category"]] = category_counts.get(t["category"], 0) + 1

# The comprehension-based version:
categories = {t["category"] for t in tickets}          # set comprehension → unique categories
by_cat = {c: sum(1 for t in tickets if t["category"] == c) for c in categories}
print(f"\nTickets per category: {by_cat}")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Sets & tuples
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: Sets & tuples")
print("=" * 70)

# Sets — fast membership + dedup:
allowed_models = {"gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "gemini-1.5-pro"}
requested = "gpt-4o-mini"
print(f"Is '{requested}' allowed? {requested in allowed_models}")

# Set operations — great for eval / analysis:
supported_a = {"streaming", "tools", "vision"}
supported_b = {"tools", "vision", "json_mode"}
print(f"Features both providers support: {supported_a & supported_b}")     # intersection
print(f"Features either supports:       {supported_a | supported_b}")     # union
print(f"Only in provider A:             {supported_a - supported_b}")     # difference

# Tuples — immutable records. Perfect for coordinates, versions, hashable keys.
model_version = ("gpt-4o", 2026, "07")
name, year, month = model_version                      # tuple unpacking
print(f"\nModel {name} released {year}-{month}")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Control flow
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: Control flow")
print("=" * 70)

def sla_status(hours_open: float, priority: str) -> str:
    """Classify a ticket against SLA. Real triage logic."""
    if priority == "high" and hours_open > 4:
        return "SLA BREACH"
    elif priority == "medium" and hours_open > 24:
        return "SLA BREACH"
    elif priority == "low" and hours_open > 72:
        return "SLA BREACH"
    else:
        return "OK"

# Show SLA status for every ticket:
for t in tickets:
    status = sla_status(t["hours_open"], t["priority"])
    marker = "❌" if status == "SLA BREACH" else "✅"
    print(f"  {marker} #{t['id']}  ({t['priority']:<6}, {t['hours_open']:>5.1f}h)  → {status}")

# `while` with break — retry loop skeleton:
attempts, max_attempts = 0, 3
while attempts < max_attempts:
    attempts += 1
    # Simulate: attempt 1 fails, attempt 2 succeeds
    succeeded = (attempts >= 2)
    if succeeded:
        print(f"\nRetry loop: attempt {attempts} SUCCESS")
        break
    else:
        print(f"Retry loop: attempt {attempts} failed, retrying...")
else:
    # `else` on a while runs only if loop finished WITHOUT break — Pythonism.
    print("All retries exhausted!")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Exception handling — the API failure pattern
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: Exception handling")
print("=" * 70)

def mock_llm_call(prompt: str, fail: bool = False) -> str:
    """Simulates an LLM API. If fail=True, raises."""
    if fail:
        raise ConnectionError("LLM provider unreachable (mock)")
    return f"[MOCK reply to: {prompt[:30]}...]"

# The universal "call an API, don't let it kill your program" pattern:
def safe_llm_call(prompt: str, fail_first: bool = True) -> str:
    try:
        response = mock_llm_call(prompt, fail=fail_first)
        return response
    except ConnectionError as e:
        # Log, fall back:
        print(f"  ⚠️  LLM call failed: {e} — using fallback")
        return "[fallback: unable to reach model, please retry]"
    except Exception as e:
        # Never let unknown exceptions escape silently:
        print(f"  ❗ Unexpected error: {type(e).__name__}: {e}")
        raise
    finally:
        # Always runs — good spot for metrics / cleanup:
        pass

print(f"Result: {safe_llm_call('Summarize ticket #1001')}")


# ─────────────────────────────────────────────────────────────
# SECTION 7: f-strings + prompt templates
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7: f-strings + prompt templates")
print("=" * 70)

# The most common LLM idiom — a prompt template with f-string interpolation:
def build_triage_prompt(ticket: dict) -> str:
    return (
        f"You are a support triage agent.\n"
        f"Ticket #{ticket['id']}\n"
        f"Priority: {ticket['priority']}\n"
        f"Category: {ticket['category']}\n"
        f"Hours open: {ticket['hours_open']:.1f}\n"
        f"\n"
        f"Return one of: ESCALATE / SELF-SERVICE / SCHEDULE."
    )

sample_prompt = build_triage_prompt(tickets[0])
print("Sample prompt:\n" + "-" * 40)
print(sample_prompt)
print("-" * 40)

# f-string formatting shortcuts:
n = 3.14159
print(f"\nPi to 2 decimals:  {n:.2f}")
print(f"As percentage:     {n/10:.1%}")
print(f"Padded number:     {42:05d}")
print(f"Debug shortcut:    {n=}")   # Prints "n=3.14159"


# ─────────────────────────────────────────────────────────────
# SECTION 8: File I/O — save the report
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 8: File I/O — save the report")
print("=" * 70)

report = {
    "generated_at": datetime.now().isoformat(),
    "total_tickets": len(tickets),
    "by_priority": {
        p: sum(1 for t in tickets if t["priority"] == p)
        for p in {"high", "medium", "low"}
    },
    "by_category": by_cat,
    "sla_breaches": [
        t["id"] for t in tickets if sla_status(t["hours_open"], t["priority"]) == "SLA BREACH"
    ],
    "avg_hours_open": round(sum(t["hours_open"] for t in tickets) / len(tickets), 2),
}

output_path = Path("ticket_report.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"Report saved to: {output_path.resolve()}")

# Round-trip: read it back to prove it worked:
with open(output_path, "r", encoding="utf-8") as f:
    loaded = json.load(f)
print(f"Round-trip check — total from file: {loaded['total_tickets']}")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
You just used every core Python data structure and control-flow tool on a
realistic AI-adjacent task (support-ticket triage):

  • Lists for ordered records + slicing + comprehensions
  • Dicts for LLM configuration + counting + .get() defaults
  • Sets for feature intersection / dedup
  • Tuples for immutable records + unpacking
  • if/elif/else + while/break/else for SLA logic and retry loops
  • try/except/finally to survive a failed LLM call
  • f-strings to build a prompt template
  • JSON file I/O to persist the analysis report

Next file: 02_functions_and_decorators.py — @retry, @cache, closures.
""")
