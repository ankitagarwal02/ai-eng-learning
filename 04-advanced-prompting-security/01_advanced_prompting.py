"""
01_advanced_prompting.py — Tree-of-Thought, Self-Refine, Meta-prompt, PAL
=========================================================================

WHAT THIS FILE TEACHES
----------------------
• Tree-of-Thought (ToT) — explore multiple reasoning paths, score, pick best.
• Self-Refine / Reflection — critique own output and rewrite.
• Meta-prompting — model writes its own sub-prompt.
• Program-Aided LM (PAL) — model writes Python to solve, then executes.
• When each is worth the cost.

HOW TO RUN
----------
    python 01_advanced_prompting.py

REAL-WORLD SCENARIOS
--------------------
• ToT: choose the best migration strategy for a legacy database.
• Self-Refine: draft, critique, and rewrite a customer apology email.
• Meta-prompt: generate an optimal system prompt for a domain-specific task.
• PAL: compute a complex financial calculation the LLM would flub.
"""

import os
import random
import ast
import operator
from typing import Callable

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# Mock LLM
# ─────────────────────────────────────────────────────────────
def call_llm(messages: list[dict], temperature: float = 0.0) -> str:
    """MOCK_MODE returns deterministic-but-varying responses based on prompt content."""
    prompt = " ".join(m["content"] for m in messages).lower()

    if "tree of thought" in prompt or "reasoning path" in prompt:
        # Different paths return different plans:
        seed = int(temperature * 100) + len(prompt) % 5
        random.seed(seed)
        paths = [
            "Path A: Blue-green deploy → cut over → decommission old. Score: 0.85",
            "Path B: Strangler-fig pattern, migrate module by module. Score: 0.92",
            "Path C: Big-bang rewrite over a weekend. Score: 0.35",
        ]
        return random.choice(paths)

    if "critique" in prompt:
        return ("Critique:\n"
                "  1. Opening feels generic — mention their specific issue.\n"
                "  2. No commitment to resolution timeline.\n"
                "  3. Passive voice weakens the apology.")

    if "rewrite based on the critique" in prompt or "improved version" in prompt:
        return ("Hi Alex,\n\n"
                "I'm sorry that your invoice showed the wrong amount — that's on us. "
                "I've credited the difference back to your card today and you'll see it "
                "within 3 business days. We've also updated our billing check to prevent "
                "this from recurring.\n\nThank you for your patience,\nSam")

    if "write the optimal system prompt" in prompt:
        return ("You are an expert legal contract reviewer. For each clause the user "
                "shares:\n  1. Classify risk: LOW / MEDIUM / HIGH\n  2. Cite the specific "
                "wording that raises risk\n  3. Suggest a rewording that reduces it\n"
                "Respond in Markdown with a 3-column table.")

    if "write python code" in prompt or "compute using code" in prompt:
        return ("```python\n"
                "principal = 250000\n"
                "annual_rate = 0.065\n"
                "months = 360\n"
                "monthly_rate = annual_rate / 12\n"
                "payment = principal * (monthly_rate * (1+monthly_rate)**months) "
                "/ ((1+monthly_rate)**months - 1)\n"
                "print(round(payment, 2))\n"
                "```")

    return "[MOCK] response."


# ─────────────────────────────────────────────────────────────
# SECTION 1: Tree-of-Thought
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Tree-of-Thought — explore multiple reasoning paths")
print("=" * 70)

def tree_of_thought(question: str, n_paths: int = 3) -> str:
    """Ask the LLM to generate N reasoning paths, then pick the highest-scoring one."""
    print(f"\nQuestion: {question}")

    paths = []
    for i in range(n_paths):
        resp = call_llm(
            [
                {"role": "system", "content":
                    "You are exploring multiple reasoning paths. Each path should propose "
                    "a distinct strategy and end with 'Score: <0.00-1.00>'."},
                {"role": "user", "content": f"Path {i+1}: {question}"},
            ],
            temperature=0.4 + i * 0.1,   # different temp to get different paths
        )
        paths.append(resp)
        print(f"  {resp}")

    # Extract scores and pick winner
    best_path = ""
    best_score = -1.0
    for p in paths:
        try:
            score = float(p.split("Score:")[-1].strip().split()[0])
            if score > best_score:
                best_score = score
                best_path = p
        except (ValueError, IndexError):
            continue

    print(f"\n✅ Winner (score {best_score}):\n   {best_path}")
    return best_path

tree_of_thought("What's the safest way to migrate our legacy Postgres to a new region?")

print("""
  ✅ Use when: high-stakes decision, budget for 3-5× the cost, one right answer.
  ❌ Avoid when: simple task, streaming UI (ToT can't stream cleanly).
""")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Self-Refine / Reflection
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 2: Self-Refine — draft, critique, rewrite")
print("=" * 70)

def self_refine(task: str, max_iterations: int = 2) -> str:
    """1. Draft.  2. Critique the draft.  3. Rewrite based on critique.  Repeat."""

    # Step 1: initial draft
    draft = call_llm([
        {"role": "system", "content": "You are a customer service manager writing a professional apology."},
        {"role": "user", "content": task},
    ])
    print(f"\n[Draft 1]\n{draft}")

    for iteration in range(max_iterations):
        # Step 2: critique
        critique = call_llm([
            {"role": "system", "content": "You are a critical editor. Find 3 concrete issues with the draft below."},
            {"role": "user", "content": f"Draft:\n{draft}\n\nProvide a critique."},
        ])
        print(f"\n[Critique {iteration+1}]\n{critique}")

        # Step 3: rewrite using the critique
        draft = call_llm([
            {"role": "system", "content": "Rewrite the draft addressing every point in the critique."},
            {"role": "user", "content":
                f"Original:\n{draft}\n\nCritique:\n{critique}\n\nProvide the improved version."},
        ])
        print(f"\n[Draft {iteration+2} — improved version]\n{draft}")

    return draft

self_refine("Draft an apology email to Alex whose latest invoice showed the wrong amount.", max_iterations=1)

print("""
  ✅ Use when: quality > cost (writing, code review, plan generation).
  ❌ Avoid when: latency-sensitive user path (each iteration is another call).

  Diminishing returns after 2-3 iterations. Add a stop condition like
  'critique returns empty' to save cost.
""")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Meta-prompting — model writes its own prompt
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 3: Meta-prompting — the model designs its own system prompt")
print("=" * 70)

def meta_prompt(task_description: str) -> str:
    """Have the model generate a system prompt tailored to the task."""
    generated = call_llm([
        {"role": "system", "content":
            "You are a prompt engineer. Given a task description, write the optimal "
            "system prompt that GPT-4o could use to perform it. Include role, rules, "
            "output format."},
        {"role": "user", "content": f"Task: {task_description}\n\nWrite the optimal system prompt for this."},
    ])
    print(f"\n[Meta-generated system prompt]\n{generated}")
    return generated

sys_prompt = meta_prompt("Review contract clauses for legal risk and suggest fixes.")

print("""
  ✅ Use when: creating tools/agents where the domain is user-specified.
  ❌ Avoid when: task is stable — hand-craft once, reuse forever.
""")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Program-Aided LM (PAL) — LLM writes code we execute
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: PAL — Program-Aided Language Model")
print("=" * 70)

def safe_eval_expr(expr: str) -> float:
    """Whitelist arithmetic evaluator — NEVER use eval() on LLM output directly.

    Allows +, -, *, /, **, parens, numbers, and named vars via ast walk.
    Any other syntax → raises."""
    tree = ast.parse(expr, mode="exec")
    env: dict[str, float] = {}
    binops = {ast.Add: operator.add, ast.Sub: operator.sub,
              ast.Mult: operator.mul, ast.Div: operator.truediv,
              ast.Pow: operator.pow, ast.USub: operator.neg}

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.BinOp):
            return binops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return binops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "round":
            args = [_eval(a) for a in node.args]
            return round(*args)
        raise ValueError(f"Disallowed syntax: {ast.dump(node)}")

    result = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            val = _eval(stmt.value)
            for tgt in stmt.targets:
                env[tgt.id] = val
        elif isinstance(stmt, ast.Expr):
            result = _eval(stmt.value)
    return result if result is not None else env.get(list(env)[-1] if env else "", 0)


def program_aided(question: str) -> str:
    code_response = call_llm([
        {"role": "system", "content":
            "For any calculation, write Python code. Output ONLY code inside "
            "```python fences. Assign intermediate values to variables. "
            "Compute the final answer with `payment = ...`."},
        {"role": "user", "content": f"Write Python code to answer: {question}"},
    ])
    print(f"\n[Model wrote code]\n{code_response}")

    # Extract code and safely evaluate (whitelist evaluator only).
    code = code_response.split("```python")[1].split("```")[0] if "```python" in code_response else ""
    # Strip the print call and evaluate the assignment portion:
    lines = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("print")]
    body = "\n".join(lines)

    try:
        result = safe_eval_expr(body)
        print(f"[Executed safely] Answer: {result}")
        return str(result)
    except Exception as e:
        print(f"[Execution refused] {e}")
        return "[unsafe code refused]"

program_aided("What's the monthly payment on a $250,000 30-year mortgage at 6.5% annual rate?")

print("""
  ✅ Use when: math, finance, physics — any task where arithmetic accuracy matters.
  ❌ Avoid when: task is language-only (summary, translation, chat).

  SECURITY NOTE: never `exec()` or `eval()` LLM-generated code. Use a whitelist
  evaluator (as above), a sandbox (Docker, Firecracker), or a language subset.
""")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Cost/quality comparison
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 5: When to use which advanced technique")
print("=" * 70)
print("""
Technique         | Extra LLM calls | Best for                     | Cost mult.
------------------+-----------------+------------------------------+-----------
Zero/few-shot     | 1               | Simple classification / QA   |    1×
CoT               | 1               | Reasoning problems           |    1×
Self-Refine       | 3-5             | Writing, code review         |   3-5×
Tree-of-Thought   | 3-10            | Complex planning             |   3-10×
Meta-prompting    | 2               | Tool/agent creation          |    2×
PAL               | 1 + eval        | Numerical accuracy           |    1×
Self-Consistency  | N (usually 5)   | Reasoning w/ verifiable ans  |    N×
""")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
Advanced prompting is not "more prompting" — it's structured multi-step reasoning:

  • Tree-of-Thought: parallel paths → best wins
  • Self-Refine: critique → improve (until diminishing returns)
  • Meta-prompt: model designs its own instructions
  • PAL: outsource math to Python (with a SAFE evaluator)

Always ask: is the quality lift worth the 3-10× cost? Sometimes yes (high-stakes
legal work, complex code generation). Often no (chat, classification).

Next: 02_injection_defense.py — 20+ real attack patterns and layered defense.
""")
