"""
01_langgraph_basics.py — StateGraph, nodes, conditional edges
==============================================================

SCENARIO
--------
Customer support routing graph. Classify intent → route to specialist handler:
    classify → handle_refund | handle_complaint | handle_general → format_response
"""

import os

try:
    from langgraph.graph import StateGraph, START, END
    from typing_extensions import TypedDict
    _has_lg = True
except ImportError:
    _has_lg = False


class State(TypedDict):
    message: str
    intent: str
    reply: str


def classify_intent(state: State) -> dict:
    msg = state["message"].lower()
    if any(w in msg for w in ("refund", "money back", "cancel")):
        intent = "refund"
    elif any(w in msg for w in ("broken", "not working", "issue", "problem")):
        intent = "complaint"
    else:
        intent = "general"
    print(f"  classify_intent → {intent}")
    return {"intent": intent}

def handle_refund(state: State) -> dict:
    return {"reply": "I've initiated your refund. It will arrive in 5-7 business days."}

def handle_complaint(state: State) -> dict:
    return {"reply": "I'm sorry to hear that. Let me escalate to a specialist."}

def handle_general(state: State) -> dict:
    return {"reply": "Thanks for reaching out. How can I help?"}

def format_response(state: State) -> dict:
    return {"reply": f"[intent={state['intent']}] {state['reply']}"}


def route_by_intent(state: State) -> str:
    return {
        "refund":    "handle_refund",
        "complaint": "handle_complaint",
        "general":   "handle_general",
    }[state["intent"]]


if _has_lg:
    g = StateGraph(State)
    g.add_node("classify", classify_intent)
    g.add_node("handle_refund", handle_refund)
    g.add_node("handle_complaint", handle_complaint)
    g.add_node("handle_general", handle_general)
    g.add_node("format", format_response)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_by_intent, {
        "handle_refund":    "handle_refund",
        "handle_complaint": "handle_complaint",
        "handle_general":   "handle_general",
    })
    for h in ("handle_refund", "handle_complaint", "handle_general"):
        g.add_edge(h, "format")
    g.add_edge("format", END)

    app = g.compile()

    for msg in [
        "I want a refund for my last purchase.",
        "My device is broken and doesn't turn on.",
        "How do I export my data?",
    ]:
        print(f"\nInput: {msg}")
        result = app.invoke({"message": msg})
        print(f"Final: {result['reply']}")
else:
    print("Install: pip install langgraph")
    print("\nPreview of the routing graph:")
    print("""
    START
      │
      ▼
    classify
      │
      ├── refund    → handle_refund   ──┐
      ├── complaint → handle_complaint ──┼──▶ format ──▶ END
      └── general   → handle_general   ──┘
""")
print("""
✅ LangGraph basics:

  • TypedDict state — nodes return partial dicts, merged into state
  • add_conditional_edges: runtime routing
  • START and END sentinels
  • .compile() → runnable app
""")
