"""
03_human_in_the_loop.py — Pause → human approves → resume
==========================================================

SCENARIO
--------
Agent proposes sending an email. Before the send_email node executes, the
graph pauses. A human reviews (e.g., in a UI), then the graph resumes with
Command(resume=True) or Command(resume=False).
"""

import os
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt, Command
    from typing_extensions import TypedDict
    _has = True
except ImportError:
    _has = False


class State(TypedDict):
    draft: str
    approved: bool
    sent: bool


def compose_email(state: State) -> dict:
    draft = "Dear Customer, we regret to inform you that..."
    print(f"  📝 Composed draft: {draft[:60]}...")
    return {"draft": draft}

def human_review(state: State) -> dict:
    """This node PAUSES. When resumed, receives the human's decision."""
    if _has:
        approval = interrupt({"draft": state["draft"], "reason": "Please approve before sending."})
        return {"approved": bool(approval)}
    return {"approved": True}

def send_email(state: State) -> dict:
    if state["approved"]:
        print("  📨 Email SENT.")
        return {"sent": True}
    print("  🚫 Send skipped (not approved).")
    return {"sent": False}


if _has:
    g = StateGraph(State)
    g.add_node("compose", compose_email)
    g.add_node("review", human_review)
    g.add_node("send", send_email)

    g.add_edge(START, "compose")
    g.add_edge("compose", "review")
    g.add_edge("review", "send")
    g.add_edge("send", END)

    memory = MemorySaver()
    app = g.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "session-1"}}

    print("STEP 1: run until interrupt")
    for chunk in app.stream({"draft": "", "approved": False, "sent": False}, config=config):
        print(f"  chunk: {chunk}")

    print("\nSTEP 2: human reviews & resumes with approval=True")
    for chunk in app.stream(Command(resume=True), config=config):
        print(f"  chunk: {chunk}")

    print(f"\nFinal state: {app.get_state(config).values}")
else:
    print("Install: pip install langgraph")
    print("""
Pattern preview:

  compose → [PAUSE for human review] → send

  interrupt(...) suspends the graph and returns to your app.
  Command(resume=<value>) continues from that exact point.
  MemorySaver persists state across pauses.
""")

print("""
✅ Human-in-the-loop pattern:

  • MemorySaver (or PostgresSaver/RedisSaver in prod) enables checkpointing
  • interrupt() inside a node pauses the graph
  • Command(resume=...) delivers the human decision back
  • Same thread_id continues the same session

Real use cases: expense approval, code deployment, medical decisions, any
high-stakes agent action that must be reviewed before execution.
""")
