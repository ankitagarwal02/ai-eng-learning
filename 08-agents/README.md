# Phase 8 — Agents from Scratch

> **Real-life analogy:** An AI agent is an employee with a job description, tools on their desk,
> and the autonomy to decide what to do next. You give them a *goal*, not a script.

---

## What separates an agent from a chain?

```
Chain:                     Agent:
  A → B → C → D                     ┌──────┐
  (fixed sequence)                  │ LLM  │
                                    └──┬───┘
                                       │  what next?
                                       ▼
                            ┌──── tool 1 ────┐
                            │                │
                            │   tool 2       │
                            │                │
                            │   tool 3       │
                            └──── ..N ───────┘
                                       │
                                       ▼
                            (repeat until DONE)
```

Agents **choose** — chains **execute**.

## The ReAct Agent Loop

```python
while iter < MAX and not done:
    response = llm(messages, tools=TOOLS)
    if response.tool_calls:
        for tc in response.tool_calls:
            result = execute(tc)
            messages.append({"role": "tool", "content": result})
    else:
        done = True
        return response.content
```

That is the entire agent. Everything else — memory, multi-agent, LangGraph —
is a decoration on this loop.

## Common Failure Modes

- **Infinite loops** — agent keeps calling the same tool. Fix: max_iterations.
- **Tool abuse** — calls tools for things it should answer directly. Fix: better system prompt.
- **Missed parallelism** — one tool call at a time when many could run together. Fix: parallel tool calls.
- **Hallucinated tool args** — invents arguments not in the schema. Fix: Pydantic schema (Phase 3).

## Folder structure

```
08-agents/
├── README.md
├── 01_agent_from_scratch.py    ← Complete agent — NO frameworks
├── 02_agent_memory.py          ← 4 memory strategies compared
└── requirements.txt
```

Prereqs: Phase 3 (tool schemas), Phase 4 (async).
Next: Phase 9 (MCP — the standard way to give agents tools).
