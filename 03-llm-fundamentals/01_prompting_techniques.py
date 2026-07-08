"""
01_prompting_techniques.py — 8 prompting techniques with real examples
=======================================================================

WHAT THIS FILE TEACHES
----------------------
All 8 core prompting techniques with working MOCK_MODE examples:
  1. Zero-shot
  2. One-shot
  3. Few-shot
  4. Chain-of-Thought (CoT)
  5. System prompt engineering (persona + constraints + format)
  6. Role prompting
  7. Self-consistency
  8. ReAct (Thought → Action → Observation)

For each: (a) the prompt template, (b) example input/output,
(c) when to use it, (d) when NOT to use it.

HOW TO RUN
----------
    python 01_prompting_techniques.py

REAL-WORLD SCENARIOS
--------------------
Techniques are demonstrated on realistic tasks:
  - Support ticket classification
  - Invoice extraction
  - Math word problem
  - Customer-tone rewriting
"""

import os
import random
from collections import Counter
from typing import Callable

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# Mock LLM — replace with `openai.chat.completions.create(...)`
# ─────────────────────────────────────────────────────────────
def mock_llm(messages: list[dict], temperature: float = 0.0) -> str:
    """Deterministic mock that responds sensibly based on the last user message.

    Keeps outputs illustrative — enough to see each technique in action.
    """
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    combined = (system + " " + user).lower()

    if "step by step" in combined or "think through" in combined:
        # Chain-of-thought response
        return ("Step 1: Extract the numbers.\n"
                "Step 2: Identify the operation.\n"
                "Step 3: Compute.\n"
                "Answer: 42")

    if "self-consistency" in combined:
        return random.choice(["A", "A", "B", "A", "C"])

    if "classify" in combined and "ticket" in combined:
        if "refund" in user.lower() or "charged" in user.lower():
            return "BILLING"
        if "down" in user.lower() or "outage" in user.lower():
            return "OUTAGE"
        return "OTHER"

    if "extract" in combined and "invoice" in combined:
        return '{"vendor":"Acme Corp","total":2200.00,"date":"2026-07-01"}'

    if "rewrite" in combined and "polite" in combined:
        return "Thank you for reaching out. I completely understand your concern..."

    if "react" in combined or "thought:" in combined:
        return "Thought: I need to look up the weather.\nAction: get_weather(city='Paris')\nObservation: 22°C, sunny.\nFinal Answer: Paris is 22°C and sunny today."

    return "[MOCK] I have processed your request."


def call_llm(messages: list[dict], temperature: float = 0.0) -> str:
    if MOCK_MODE:
        return mock_llm(messages, temperature)
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def show(title: str, prompt: list[dict], response: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for m in prompt:
        content = m["content"] if len(m["content"]) < 300 else m["content"][:300] + "..."
        print(f"  [{m['role']:>9}] {content}")
    print(f"  [response ] {response}")


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 1 — Zero-shot
# ─────────────────────────────────────────────────────────────
prompt = [
    {"role": "user", "content": "Classify this support ticket as BILLING, OUTAGE, or OTHER: 'You double-charged me for July.'"},
]
show("1. Zero-shot  —  no examples, just ask", prompt, call_llm(prompt))
print("""
  ✅ Use when: task is common (classification, summary, translation) and the
     labels are obvious from words alone.
  ❌ Avoid when: labels are ambiguous or format matters.
""")


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 2 — One-shot
# ─────────────────────────────────────────────────────────────
prompt = [
    {"role": "user", "content": (
        "Classify this support ticket as BILLING, OUTAGE, or OTHER.\n\n"
        "Example:\n"
        "  Ticket: 'My credit card was charged twice.'\n"
        "  Category: BILLING\n\n"
        "Now classify:\n"
        "  Ticket: 'The dashboard is down since 2pm.'\n"
        "  Category:"
    )},
]
show("2. One-shot  —  show one worked example", prompt, call_llm(prompt))
print("""
  ✅ Use when: you want to teach a specific FORMAT with minimum tokens.
  ❌ Avoid when: task has many edge cases — one example won't cover them.
""")


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 3 — Few-shot
# ─────────────────────────────────────────────────────────────
prompt = [
    {"role": "user", "content": (
        "Classify the ticket. Respond with only the category label.\n\n"
        "Examples:\n"
        "  'My card was charged twice.' → BILLING\n"
        "  'Site is down since 3pm.'    → OUTAGE\n"
        "  'How to export data?'        → HOWTO\n"
        "  'Please add dark mode.'      → FEATURE\n"
        "  'Great product, thanks!'     → PRAISE\n\n"
        "Ticket: 'Refund my last payment please'\n"
        "Category:"
    )},
]
show("3. Few-shot  —  3-5 examples, teaches nuance", prompt, call_llm(prompt))
print("""
  ✅ Use when: you have 3-5 canonical examples that show the class boundaries.
  ❌ Avoid when: your examples are wildly different lengths — biases the model.
""")


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 4 — Chain-of-Thought (CoT)
# ─────────────────────────────────────────────────────────────
prompt = [
    {"role": "user", "content": (
        "A store sells apples at $2 each and oranges at $3 each. "
        "A customer buys 5 apples and 3 oranges. What is the total?\n\n"
        "Think step by step, then give your final answer."
    )},
]
show("4. Chain-of-Thought  —  'Let's think step by step'", prompt, call_llm(prompt))
print("""
  ✅ Use when: multi-step reasoning, math, code debugging, planning.
  ❌ Avoid when: task is a simple lookup — CoT just wastes tokens.

  Insight: giving the model 'thinking room' before the answer boosts accuracy
  on reasoning tasks by 10-40 percentage points on some benchmarks.
""")


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 5 — System prompt engineering
# ─────────────────────────────────────────────────────────────
prompt = [
    {"role": "system", "content": (
        "You are a customer support agent for FinBank.\n"
        "Rules:\n"
        "  1. Never quote interest rates — direct users to the website.\n"
        "  2. Always ask for the last 4 digits of their card before account questions.\n"
        "  3. If frustrated, apologize once, then focus on resolution.\n"
        "Response format: 2-3 sentences, warm and professional tone."
    )},
    {"role": "user", "content": "What's the current interest rate on your gold card?"},
]
show("5. System prompt engineering  —  persona + rules + format", prompt, call_llm(prompt))
print("""
  ✅ Always use a system prompt in production. It's the cheapest quality lever.
  ❌ Don't stuff too much — 200-500 tokens is usually the sweet spot.

  System prompts persist across turns; user messages don't. Load-bearing content
  belongs in system, not user.
""")


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 6 — Role prompting
# ─────────────────────────────────────────────────────────────
prompt = [
    {"role": "system", "content": (
        "You are a senior financial analyst with 20 years of experience "
        "analyzing tech company earnings. Be precise, quantitative, and skeptical."
    )},
    {"role": "user", "content": "Rewrite this to sound polite:\n'Fix your broken product now'"},
]
show("6. Role prompting  —  'You are an expert X'", prompt, call_llm(prompt))
print("""
  ✅ Use when: task benefits from a specific persona (analyst, teacher, editor).
  ❌ Avoid when: role has ethical implications ('You are a hacker...' can jailbreak).

  Under the hood: role prompting biases the model toward vocabulary/style used
  by that role in training data — it doesn't grant new knowledge.
""")


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 7 — Self-consistency
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. Self-consistency  —  sample 3 times at temp>0, majority vote")
print("=" * 70)

question = "Which is the odd one out: apple, orange, carrot, banana? (Answer with just the word.)"

# Sample 5 times at temperature 0.8:
random.seed(1)
samples = []
for _ in range(5):
    # We mock varied outputs; real code passes temperature=0.8 to the API.
    ans = random.choices(["carrot", "carrot", "carrot", "orange", "banana"], k=1)[0]
    samples.append(ans)

print(f"  Samples: {samples}")
winner = Counter(samples).most_common(1)[0][0]
print(f"  Majority vote: {winner}")
print("""
  ✅ Use when: task has a definite right answer but the model wobbles.
  ❌ Avoid when: open-ended tasks (no 'majority') or cost-sensitive (3× the tokens).

  Cost: multiply your bill by N. Only pay this for evals or high-stakes decisions.
""")


# ─────────────────────────────────────────────────────────────
# TECHNIQUE 8 — ReAct (Thought/Action/Observation)
# ─────────────────────────────────────────────────────────────
prompt = [
    {"role": "system", "content": (
        "You are an assistant that solves tasks by ReActing: for each step, output\n"
        "  Thought: <what to do next>\n"
        "  Action: <tool_name(args)>\n"
        "  Observation: <result>\n"
        "Available tools:\n"
        "  get_weather(city: str) — returns current weather\n"
        "  send_email(to: str, body: str) — sends an email\n"
        "Finish with:\n"
        "  Final Answer: <the answer>"
    )},
    {"role": "user", "content": "Tell me the current weather in Paris and email it to me at me@x.com."},
]
show("8. ReAct  —  Thought → Action → Observation loop", prompt, call_llm(prompt))
print("""
  ✅ Foundation of agentic behavior (Phase 8). Every tool-using agent uses this.
  ❌ Overkill for one-shot Q&A — you're paying for a loop you don't need.

  In practice, you rarely write ReAct by hand — you use structured tool calling
  (Phase 8) which is a cleaner version of the same idea.
""")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
The 8 techniques in one decision tree:

  Is the answer format obvious and simple?           → Zero-shot
  Need to teach a specific format?                    → One-shot / Few-shot
  Multi-step reasoning?                               → Chain-of-Thought
  Need a persona or hard rules?                       → System + Role prompting
  High-stakes but model wobbles?                      → Self-consistency (3-5×)
  Need to use tools / do research?                    → ReAct (→ Phase 8)

Cost hierarchy (cheapest first):
  Zero-shot  <  One/Few-shot  <  CoT  <  Self-consistency (N×)  <  ReAct (loops)

Golden rule: start SIMPLE, escalate only when quality demands it. Every prompt
technique above zero-shot costs more tokens — often 2-5× more.

Next: 02_structured_output.py — parsing LLM output SAFELY.
""")
