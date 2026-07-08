"""
02_lcel_chains.py — LCEL composition patterns
==============================================

WHAT THIS FILE TEACHES
----------------------
The five LCEL combinators + itemgetter — everything you need to compose any
production chain WITHOUT writing loops or if-statements around LLM calls.

  1. RunnableParallel        — fan-out, results merged into a dict
  2. RunnablePassthrough     — pass input through unchanged (RAG's best friend)
  3. RunnableBranch          — conditional routing based on input
  4. RunnableWithFallbacks   — primary + fallback if primary fails
  5. RunnableLambda          — any Python function becomes a chainable step
  + itemgetter               — pluck sub-fields from a state dict

HOW TO RUN
----------
    pip install langchain langchain-core
    python 02_lcel_chains.py

MOCK_MODE-safe (does not call OpenAI).
"""

import os
from operator import itemgetter

try:
    from langchain_core.runnables import (
        RunnableLambda, RunnableParallel, RunnablePassthrough,
        RunnableBranch, RunnableWithFallbacks,
    )
    _has_lc = True
except ImportError:
    _has_lc = False

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


if not _has_lc:
    print("Install: pip install langchain-core")
    print("\nThis file demonstrates all five LCEL combinators. See docstring for pattern.")
    exit()


# ─────────────────────────────────────────────────────────────
# Fake "models" as pure functions — replace with real ChatOpenAI in prod
# ─────────────────────────────────────────────────────────────
def fake_summary(article: str) -> str:
    return f"[summary] {article[:60]}..."

def fake_keywords(article: str) -> list[str]:
    return sorted({w.lower() for w in article.split() if len(w) > 6})[:5]

def fake_classify(article: str) -> str:
    low = article.lower()
    if any(k in low for k in ("earnings", "revenue", "profit")):  return "finance"
    if any(k in low for k in ("layoff", "hiring", "employee")):   return "hr"
    return "general"

summarize    = RunnableLambda(fake_summary)
keywords     = RunnableLambda(fake_keywords)
classify_it  = RunnableLambda(fake_classify)


# ─────────────────────────────────────────────────────────────
# PATTERN 1 — RunnableParallel: fan-out
# ─────────────────────────────────────────────────────────────
print("=" * 78)
print("PATTERN 1: RunnableParallel — one input, two independent outputs")
print("=" * 78)

# Given an article, produce BOTH a summary and a keyword list in parallel:
extract = RunnableParallel({
    "summary":  summarize,
    "keywords": keywords,
})

article = "The Acme Corp Q3 earnings beat forecasts with 22% revenue growth."
result = extract.invoke(article)
print(f"Input: {article!r}")
print(f"Output: {result}")


# ─────────────────────────────────────────────────────────────
# PATTERN 2 — RunnablePassthrough: keep the input alongside computed values
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("PATTERN 2: RunnablePassthrough — preserve original + add computed")
print("=" * 78)

# We want: {"article": <original>, "summary": <computed>, "category": <computed>}
enrich = RunnableParallel({
    "article":  RunnablePassthrough(),
    "summary":  summarize,
    "category": classify_it,
})

result = enrich.invoke("Acme announced 300 layoffs in the enterprise division today.")
for k, v in result.items():
    print(f"  {k:<10} = {v}")


# ─────────────────────────────────────────────────────────────
# PATTERN 3 — RunnableBranch: conditional routing
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("PATTERN 3: RunnableBranch — route based on classification")
print("=" * 78)

def format_finance(article: str) -> str:
    return f"📊 FINANCE\n  Summary: {fake_summary(article)}\n  → route to CFO briefing"

def format_hr(article: str) -> str:
    return f"👥 HR\n  Summary: {fake_summary(article)}\n  → route to People Ops"

def format_general(article: str) -> str:
    return f"📝 GENERAL\n  Summary: {fake_summary(article)}\n  → route to newsroom"

# Branch takes (condition_fn, chain) pairs, plus a default:
router = RunnableBranch(
    (lambda x: fake_classify(x) == "finance", RunnableLambda(format_finance)),
    (lambda x: fake_classify(x) == "hr",      RunnableLambda(format_hr)),
    RunnableLambda(format_general),  # default (last positional arg)
)

for a in [
    "Q3 revenue was up 22% year over year.",
    "We are announcing 300 layoffs today.",
    "The new office lobby renovation begins Monday.",
]:
    print(f"\nInput: {a}")
    print(router.invoke(a))


# ─────────────────────────────────────────────────────────────
# PATTERN 4 — RunnableWithFallbacks: resilient chain
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("PATTERN 4: RunnableWithFallbacks — resilient chain")
print("=" * 78)

def unreliable_primary(text: str) -> str:
    # Simulate a flaky primary model
    if "fail" in text.lower():
        raise RuntimeError("Primary model timed out")
    return f"[PRIMARY reply] {text[:40]}..."

def cheap_fallback(text: str) -> str:
    return f"[FALLBACK reply] {text[:40]}..."

primary = RunnableLambda(unreliable_primary)
fallback = RunnableLambda(cheap_fallback)

resilient = primary.with_fallbacks([fallback])

for prompt in ["This should succeed", "Please fail here"]:
    print(f"\nInput: {prompt!r}")
    print(f"Output: {resilient.invoke(prompt)}")


# ─────────────────────────────────────────────────────────────
# PATTERN 5 — Threading state through a chain (itemgetter)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("PATTERN 5: itemgetter — thread state through parallel branches")
print("=" * 78)

# Simulate a RAG chain where we need to pass BOTH the retrieved docs AND
# the original query to the answer step:
def retrieve(query: str) -> list[str]:
    return [f"[doc {i}] relevant to '{query}'" for i in range(1, 4)]

def answer_with_docs(inputs: dict) -> str:
    q = inputs["question"]
    docs = "\n  ".join(inputs["docs"])
    return f"Q: {q}\nContext:\n  {docs}\n\nAnswer: [based on context]"

# The chain:
rag_chain = (
    {
        "docs":     itemgetter("question") | RunnableLambda(retrieve),
        "question": itemgetter("question"),
    }
    | RunnableLambda(answer_with_docs)
)

print(rag_chain.invoke({"question": "What is LCEL?"}))


# ─────────────────────────────────────────────────────────────
# PATTERN 6 — Full 4-stage pipeline built from combinators only
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("PATTERN 6: 4-stage pipeline via composition only")
print("=" * 78)

pipeline = (
    # Stage 1 — enrich with summary + keywords + category IN PARALLEL
    RunnableParallel({
        "article":  RunnablePassthrough(),
        "summary":  summarize,
        "keywords": keywords,
        "category": classify_it,
    })
    # Stage 2 — build a report dict
    | RunnableLambda(lambda x: {
        **x,
        "report": f"[{x['category'].upper()}] {x['summary']} | Keywords: {x['keywords']}",
    })
    # Stage 3 — route to team based on category (branching!)
    | RunnableBranch(
        (lambda x: x["category"] == "finance", RunnableLambda(lambda x: {**x, "team": "CFO briefing"})),
        (lambda x: x["category"] == "hr",      RunnableLambda(lambda x: {**x, "team": "People Ops"})),
        RunnableLambda(lambda x: {**x, "team": "Newsroom"}),
    )
    # Stage 4 — produce a final string
    | RunnableLambda(lambda x: f"Deliver to {x['team']}:\n  {x['report']}")
)

for input_article in [
    "Revenue grew 22% and margins improved.",
    "Layoffs of 300 employees announced.",
    "The cafeteria is closed on Friday.",
]:
    print(f"\nInput:  {input_article}")
    print(f"Output: {pipeline.invoke(input_article)}")


# ─────────────────────────────────────────────────────────────
# PATTERN 7 — .stream() and .batch() are FREE with any chain
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("PATTERN 7: .batch() — apply a chain to many inputs in parallel")
print("=" * 78)

articles = [
    "Q4 revenue up 15%.",
    "Employee onboarding revamped.",
    "New coffee machine installed.",
]
batch_results = pipeline.batch(articles)
for a, r in zip(articles, batch_results):
    print(f"  '{a[:35]}' →\n    {r}")


print("""

✅ Summary — LCEL combinator toolkit:

  RunnableParallel       fan-out
  RunnablePassthrough    preserve original alongside computed
  RunnableBranch         conditional routing
  RunnableWithFallbacks  resilience
  RunnableLambda         any function → chainable
  itemgetter             thread state through branches

  .invoke(x)  → single call
  .batch(xs)  → parallel
  .stream(x)  → token stream (works up through the LLM step)

Every LangChain agent, RAG, or chatbot in production is a composition of
these seven pieces. Master them once, use them everywhere.

Next: 03_langchain_memory.py.
""")
