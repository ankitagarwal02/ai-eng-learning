# Phase 12 — LangGraph

> **Real-life analogy:** LangGraph is a flowchart that executes itself. Boxes are AI nodes;
> arrows are runtime-decided edges. Unlike LangChain's linear chains, LangGraph loops back,
> branches, pauses for human review, and resumes.

---

## When you outgrow LangChain

LangChain is great for `A | B | C`. When you need:
- **Loops** (agent keeps trying until a condition is met)
- **Branches decided at runtime** (route to specialist based on state)
- **Human-in-the-loop pauses** ("wait for my approval, then continue")
- **Persistent state across calls** (checkpoints, resume from failure)

… you want LangGraph.

## StateGraph core

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]   # accumulates across nodes

def node_a(state: State):
    return {"messages": ["from A"]}

def node_b(state: State):
    return {"messages": ["from B"]}

g = StateGraph(State)
g.add_node("a", node_a)
g.add_node("b", node_b)
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", END)
app = g.compile()
```

## The 3 things LangGraph gives you

1. **Cyclical graphs** — an agent can loop back to a node ("plan → act → observe → replan").
2. **Checkpoints (MemorySaver)** — full state snapshotted every step; resume from any point.
3. **Interrupts** — pause before a node runs, wait for human input, then resume with `Command(resume=...)`.

## Folder structure

```
12-langgraph/
├── README.md
├── 01_langgraph_basics.py     ← StateGraph, nodes, conditional edges
├── 02_agent_graph.py          ← Full agent as a graph (loops)
├── 03_human_in_the_loop.py    ← Pause → human approves → resume
└── requirements.txt
```

Prereqs: Phase 11 (LangChain), Phase 8 (agent basics).
Next: Phase 13 (Memory — often persisted via LangGraph checkpoints).
