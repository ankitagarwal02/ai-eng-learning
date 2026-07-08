# Phase 11 — LangChain

> **Real-life analogy:** LangChain is LEGO for AI apps. Snap together pre-built pieces
> (models, prompts, memory, tools) using the `|` pipe operator (LCEL).

---

## LCEL — the pipe operator

```python
chain = prompt | model | StrOutputParser()
result = chain.invoke({"question": "hi"})
```

Every LangChain object that implements `Runnable` can be composed with `|`. This is the
core idiom you'll see everywhere.

## The 4 Runnable combinators

```
RunnableParallel:  fan-out into multiple chains, gather all results into a dict
RunnablePassthrough: pass input through unchanged (great for RAG — pass docs AND query)
RunnableBranch:      conditional routing based on input
RunnableWithFallbacks: primary + fallback (retry with different model)
```

## When to use LangChain vs raw OpenAI

**Use LangChain when:**
- You want tracing (LangSmith) with zero setup
- Chaining multiple LLM calls with memory
- Provider-agnostic (swap OpenAI for Claude with one line)

**Use raw OpenAI when:**
- Single LLM call in an app
- Latency-critical (LangChain adds ~50ms overhead)
- Custom control flow LangChain doesn't support cleanly

## Folder structure

```
11-langchain/
├── README.md
├── 01_langchain_basics.py   ← ChatOpenAI, ChatPromptTemplate, LCEL chain
├── 02_lcel_chains.py        ← Parallel, Passthrough, Branch, Fallbacks
├── 03_langchain_memory.py   ← RunnableWithMessageHistory, session-based
├── 04_langchain_rag.py      ← LangChain-native RAG chain
└── 05_langchain_agents.py   ← @tool, create_openai_tools_agent, AgentExecutor
```

## Quickstart

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("Classify as SPAM or HAM: {msg}")
chain = prompt | model | StrOutputParser()

chain.invoke({"msg": "Free iPhone click here!"})   # → "SPAM"
```

Prereqs: Phase 3 (prompts), Phase 8 (agents).
Next: Phase 12 (LangGraph — for cyclical / stateful workflows).
