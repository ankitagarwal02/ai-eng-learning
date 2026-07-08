"""
02_agent_graph.py — Full ReAct agent expressed as a LangGraph
==============================================================

WHAT THIS FILE TEACHES
----------------------
Why LangGraph beats LangChain's AgentExecutor for real production agents:

  • Cycles (agent revisits planning after each tool result)
  • Explicit state that persists across turns
  • Checkpointing (resume interrupted runs)
  • Clear routing rules — no hidden magic
  • Easy to add human-in-the-loop (see 03_human_in_the_loop.py)

The graph:

     START ──▶ agent ──▶ (has tool_calls?)
                          │  yes
                          ▼
                        tools ──┐
                          ▲     │
                          └─────┘  cycle: after tools, back to agent
                          │  no
                          ▼
                         END

HOW TO RUN
----------
    pip install langgraph
    python 02_agent_graph.py

REAL-WORLD SCENARIO
-------------------
Same task as Phase 8's from-scratch agent:
  "Find the weather in Paris, convert 22°C to Fahrenheit, email me the result."
Now expressed as a state machine that could be paused, resumed, and audited.
"""

import os
import json
import asyncio
from typing import Annotated

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from typing_extensions import TypedDict
    _has_lg = True
except ImportError:
    _has_lg = False

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────
# Annotated[list, operator.add] makes list fields ACCUMULATE across nodes,
# rather than being replaced. Essential for message history.
import operator

if _has_lg:
    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]
        iterations: int


    # ─────────────────────────────────────────────────────────────
    # Tools (same as Phase 8)
    # ─────────────────────────────────────────────────────────────
    def get_weather(city: str) -> dict:
        fake = {"Paris": 22, "Tokyo": 28}
        return {"city": city, "temp_c": fake.get(city, 20), "condition": "sunny"}

    def calculate(expression: str) -> dict:
        import ast, operator as op
        ops = {ast.Add: op.add, ast.Sub: op.sub,
               ast.Mult: op.mul, ast.Div: op.truediv}
        def _e(n):
            if isinstance(n, ast.Constant): return n.value
            if isinstance(n, ast.BinOp): return ops[type(n.op)](_e(n.left), _e(n.right))
        return {"result": _e(ast.parse(expression, mode="eval").body)}

    def send_email(to: str, body: str) -> dict:
        return {"status": "sent", "to": to}

    TOOLS = {"get_weather": get_weather, "calculate": calculate, "send_email": send_email}


    # ─────────────────────────────────────────────────────────────
    # Mock LLM node — same deterministic behavior as Phase 8
    # ─────────────────────────────────────────────────────────────
    def agent_node(state: AgentState) -> dict:
        """Decide: emit tool_calls OR emit final content."""
        called_tools = set()
        for m in state["messages"]:
            if m.get("role") == "tool":
                called_tools.add(m.get("tool_name"))

        user = next((m["content"] for m in state["messages"] if m["role"] == "user"), "").lower()

        if "get_weather" not in called_tools and "paris" in user:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "w1", "name": "get_weather", "args": {"city": "Paris"}}],
                }],
                "iterations": state["iterations"] + 1,
            }

        if "calculate" not in called_tools and any(m.get("tool_name") == "get_weather" for m in state["messages"]):
            weather_result = next(
                json.loads(m["content"]) for m in state["messages"]
                if m.get("tool_name") == "get_weather"
            )
            return {
                "messages": [{
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "name": "calculate",
                                     "args": {"expression": f"{weather_result['temp_c']}*9/5+32"}}],
                }],
                "iterations": state["iterations"] + 1,
            }

        if "send_email" not in called_tools and "email" in user:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "e1", "name": "send_email",
                                     "args": {"to": "me@example.com",
                                              "body": "Paris: 22°C (71.6°F)"}}],
                }],
                "iterations": state["iterations"] + 1,
            }

        return {
            "messages": [{"role": "assistant", "content": "Done. Weather checked, converted, emailed."}],
            "iterations": state["iterations"] + 1,
        }


    def tools_node(state: AgentState) -> dict:
        """Execute all tool_calls in the LAST assistant message."""
        last_assistant = None
        for m in reversed(state["messages"]):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                last_assistant = m
                break
        if not last_assistant:
            return {"messages": []}

        new_messages = []
        for tc in last_assistant["tool_calls"]:
            fn = TOOLS[tc["name"]]
            try:
                result = fn(**tc["args"])
            except Exception as e:
                result = {"error": str(e)}
            new_messages.append({
                "role": "tool",
                "tool_name": tc["name"],
                "tool_call_id": tc["id"],
                "content": json.dumps(result),
            })
        return {"messages": new_messages}


    def route(state: AgentState) -> str:
        """After the agent node, decide: tools if there are pending calls, else END."""
        if state["iterations"] > 8:
            return END       # safety cap
        last = state["messages"][-1] if state["messages"] else None
        if last and last.get("role") == "assistant" and last.get("tool_calls"):
            return "tools"
        return END


    # ─────────────────────────────────────────────────────────────
    # Build the graph
    # ─────────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")   # ← the cycle!

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)


    # ─────────────────────────────────────────────────────────────
    # Run
    # ─────────────────────────────────────────────────────────────
    def run(user_message: str, thread_id: str = "session-1"):
        config = {"configurable": {"thread_id": thread_id}}
        print(f"\n👤 USER: {user_message}\n")

        state = {
            "messages": [{"role": "user", "content": user_message}],
            "iterations": 0,
        }

        for i, event in enumerate(app.stream(state, config=config)):
            node, patch = next(iter(event.items()))
            print(f"── Step {i+1}: node='{node}' ──")
            for m in patch.get("messages", []):
                role = m.get("role")
                if role == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        print(f"    🤖 tool_call {tc['name']}({tc['args']})")
                elif role == "assistant":
                    print(f"    🤖 FINAL: {m['content']}")
                elif role == "tool":
                    print(f"    🔧 result[{m['tool_name']}]: {m['content']}")

        final = app.get_state(config).values
        print(f"\n✅ End state: {len(final['messages'])} messages, {final['iterations']} iterations")
        return final


    run("Find the weather in Paris, convert 22C to Fahrenheit, and email me the result.")


else:
    print("Install: pip install langgraph")
    print("""
This file demonstrates the same tool-using agent as Phase 8's from-scratch
version, but expressed as a LangGraph:

  START → agent ─(has tool_calls?)→ tools → agent → ... → END

Key advantages over 08-agents/01_agent_from_scratch.py:

  • Automatic state persistence via MemorySaver (or Postgres/Redis in prod)
  • Resume from any checkpoint with thread_id
  • Clean routing rules — no hidden state
  • Foundation for human-in-the-loop (03_human_in_the_loop.py)

Same agent, but production-grade infrastructure.
""")


print("""

✅ Summary — Agent as a StateGraph:

  • Nodes: agent (LLM call), tools (execute tool_calls)
  • Edge: agent → tools (if tool_calls) OR agent → END (otherwise)
  • Cycle: tools → agent  (agent decides again after seeing tool results)
  • Checkpointing: MemorySaver persists state, resume by thread_id

Compare to the loop from Phase 8:
    while not done:
        response = llm(...)
        if response.tool_calls:
            run tools; append; continue
        else: return

LangGraph gives you the SAME thing but with:
  • Free persistence
  • Free tracing
  • Free HITL support
  • Runtime-modifiable graph (add nodes without restart)

Prefer LangGraph for anything longer than a single-step Q&A agent.
""")
