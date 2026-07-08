"""
04_langchain_rag.py — LangChain-native RAG chain
=================================================

WHAT THIS FILE TEACHES
----------------------
Build a full RAG chain using LangChain's LCEL:

  retriever         → docs
  ↓
  parallel(context, question)
  ↓
  prompt
  ↓
  model
  ↓
  output parser

… all composed with `|`. Compare vs Phase 7 (from-scratch RAG) — LangChain
handles the plumbing so you focus on retrieval quality and prompt design.

HOW TO RUN
----------
    pip install langchain langchain-core
    python 04_langchain_rag.py

MOCK_MODE-safe.
"""

import os
from operator import itemgetter

try:
    from langchain_core.runnables import (
        RunnableLambda, RunnableParallel, RunnablePassthrough,
    )
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    _has_lc = True
except ImportError:
    _has_lc = False

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


if not _has_lc:
    print("Install: pip install langchain langchain-core")
    exit()


# ─────────────────────────────────────────────────────────────
# 1. Retriever (mock — replace with Chroma/Qdrant retriever)
# ─────────────────────────────────────────────────────────────
KB = [
    {"id": "HR-PTO", "text": "Employees accrue PTO at 1.25 days per month, 15 days per year. Carryover: 5 days."},
    {"id": "HR-SICK", "text": "10 paid sick days per calendar year, credited January 1."},
    {"id": "HR-PERSONAL", "text": "3 personal days per year, must be used by December 31."},
    {"id": "HR-PARENTAL", "text": "16 weeks paid parental leave regardless of gender or path to parenthood."},
    {"id": "HR-REMOTE", "text": "Up to 3 days remote per week with manager approval."},
]

def retrieve(query: str, k: int = 3) -> list[dict]:
    """Toy retriever: keyword scoring. Real code: `chroma_collection.as_retriever()`."""
    q_words = set(query.lower().split())
    scored = [
        (sum(1 for w in q_words if w in d["text"].lower()), d)
        for d in KB
    ]
    scored.sort(key=lambda t: -t[0])
    return [d for score, d in scored[:k] if score > 0]

retriever = RunnableLambda(lambda q: retrieve(q, k=3))


# ─────────────────────────────────────────────────────────────
# 2. Format docs into a context string
# ─────────────────────────────────────────────────────────────
def format_docs(docs: list[dict]) -> str:
    if not docs:
        return "(no relevant documents found)"
    return "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)


# ─────────────────────────────────────────────────────────────
# 3. The prompt template
# ─────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an HR assistant. Answer using ONLY the CONTEXT. Cite [SOURCE-ID] "
               "for every claim. If not in context, say you don't know."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


# ─────────────────────────────────────────────────────────────
# 4. Mock model (replace with ChatOpenAI in prod)
# ─────────────────────────────────────────────────────────────
def mock_model(messages) -> str:
    # messages is a list of BaseMessage in real LangChain; here we approximate.
    human = str(messages)
    # Extract Question part
    import re
    q_match = re.search(r"Question:\s*(.+?)(?:\Z|\n)", human)
    q = q_match.group(1) if q_match else ""

    ctx_match = re.search(r"Context:\s*\[([A-Z0-9\-#]+)\]\s*(.+?)(?=\[|Question:|\Z)", human, re.DOTALL)
    if ctx_match:
        cid, ctext = ctx_match.group(1), ctx_match.group(2).strip()
        # Pull a relevant sentence from the context
        for sent in re.split(r"(?<=[.!?])\s+", ctext):
            if any(w.lower() in sent.lower() for w in q.split() if len(w) > 3):
                return f"{sent} [{cid}]"
        return f"{ctext[:100]}... [{cid}]"

    return "I don't have that information."

model = RunnableLambda(mock_model)


# ─────────────────────────────────────────────────────────────
# 5. Compose the RAG chain via LCEL
# ─────────────────────────────────────────────────────────────
rag_chain = (
    {
        "context":  itemgetter("question") | retriever | RunnableLambda(format_docs),
        "question": itemgetter("question"),
    }
    | RAG_PROMPT
    | model
    | StrOutputParser()
)


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────
QUESTIONS = [
    "How many sick days do I get?",
    "Can I take my parental leave over 12 months?",
    "How many personal days per year?",
    "What is the office holiday schedule?",   # not in KB
]

for q in QUESTIONS:
    print("═" * 78)
    print(f"Q: {q}")
    print("─" * 78)
    answer = rag_chain.invoke({"question": q})
    print(f"A: {answer}")

# ─────────────────────────────────────────────────────────────
# RAG + Memory (bonus)
# ─────────────────────────────────────────────────────────────
print("\n\n" + "═" * 78)
print("BONUS: RAG + conversation history in one chain")
print("═" * 78)

CONVO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer using CONTEXT below. Consider CHAT HISTORY. Cite [SOURCE-ID]."),
    ("human", "CONTEXT:\n{context}\n\nHISTORY:\n{history}\n\nQUESTION: {question}"),
])

# Same shape as rag_chain but with a `history` string threaded through:
convo_rag = (
    {
        "context":  itemgetter("question") | retriever | RunnableLambda(format_docs),
        "question": itemgetter("question"),
        "history":  itemgetter("history"),
    }
    | CONVO_PROMPT
    | model
    | StrOutputParser()
)

print(convo_rag.invoke({
    "question": "And what about personal days?",
    "history": "User previously asked about PTO. Answer: 15 days/year [HR-PTO].",
}))


print("""

✅ Summary — LangChain-native RAG:

  Retrieve → format → prompt → model → parse
  All composed with `|`. That's it.

Advantages over from-scratch (Phase 7):
  • Free tracing (LangSmith)
  • .batch() = concurrent RAG calls
  • .stream() = token streaming works through the whole chain
  • Model swap = one line change
  • Easy to add memory (see 03_langchain_memory.py)

Rule of thumb: use raw OpenAI for single-call scripts and latency-critical
paths. Use LangChain for composed pipelines where the observability + swap
benefits are worth the ~50ms overhead.

Next: 05_langchain_agents.py.
""")
