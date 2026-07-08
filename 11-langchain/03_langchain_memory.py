"""
03_langchain_memory.py — Message history + session-based memory
================================================================

WHAT THIS FILE TEACHES
----------------------
  • ChatMessageHistory (in-memory)
  • RunnableWithMessageHistory (auto history injection)
  • Session-based memory (different history per session_id)
  • ConversationSummaryMemory pattern (implemented manually)
  • Memory + RAG (inject BOTH retrieved docs AND history)

HOW TO RUN
----------
    pip install langchain langchain-core
    python 03_langchain_memory.py

REAL-WORLD SCENARIO
-------------------
A customer support bot that remembers the customer's name, order number, and
what they've already tried — across 8 conversation turns.
"""

import os

try:
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    _has_lc = True
except ImportError:
    _has_lc = False

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


if not _has_lc:
    print("Install: pip install langchain langchain-core")
    exit()


# ─────────────────────────────────────────────────────────────
# SESSION STORAGE
# ─────────────────────────────────────────────────────────────
# In production: swap for Redis / Postgres / DynamoDB-backed history.
_sessions: dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _sessions:
        _sessions[session_id] = InMemoryChatMessageHistory()
    return _sessions[session_id]


# ─────────────────────────────────────────────────────────────
# The "model" — a mock that inspects history + user message
# ─────────────────────────────────────────────────────────────
def mock_support_bot(inputs: dict) -> str:
    """Inputs: {"user_input": str, "history": list[BaseMessage]}."""
    history = inputs.get("history", [])
    q = inputs["user_input"].lower()

    # Extract known facts from history
    facts = {}
    for m in history:
        content = m.content if isinstance(m, BaseMessage) else str(m)
        low = content.lower()
        if "my name is" in low:
            facts["name"] = content.split("my name is")[-1].strip().split()[0].rstrip(".,")
        if "order" in low and "#" in low:
            for token in content.split():
                if token.startswith("#"):
                    facts["order"] = token
                    break
        if "tried" in low:
            facts["already_tried"] = content

    # Respond based on state
    if "my name is" in q:
        name = q.split("my name is")[-1].strip().split()[0].rstrip(".,")
        return f"Nice to meet you, {name.title()}. What can I help you with today?"

    if "#" in q and "order" in q:
        return "Got it — thanks for the order number. What's the issue?"

    if "tried" in q or "does not work" in q or "still" in q:
        name = facts.get("name", "there")
        return (f"Sorry to hear that, {name}. I've noted what you've tried. "
                f"Let me escalate this to a specialist.")

    if "name" in q and "what" in q and facts.get("name"):
        return f"Your name is {facts['name']}."

    if "order" in q and "what" in q and facts.get("order"):
        return f"Your order number is {facts['order']}."

    name_prefix = f"{facts.get('name', 'there').title()}, " if facts.get("name") else ""
    return f"{name_prefix}how can I help you?"


bot_chain = RunnableLambda(mock_support_bot)


# ─────────────────────────────────────────────────────────────
# Wrap with RunnableWithMessageHistory
# ─────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Aria, a support agent. Remember the customer across turns."),
    MessagesPlaceholder("history"),
    ("human", "{user_input}"),
])

# The pattern: turn a plain function into a Runnable that auto-injects history.
def _build_input(inputs: dict) -> dict:
    """Adapter: LangChain gives us {"user_input", "history"}."""
    return inputs

chain = RunnableLambda(_build_input) | bot_chain

with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="history",
)


# ─────────────────────────────────────────────────────────────
# DEMO — one customer, one session, 8 turns
# ─────────────────────────────────────────────────────────────
def run_conversation(session_id: str, turns: list[str]):
    print(f"\n═══ Session: {session_id} ═══")
    for user_input in turns:
        response = with_history.invoke(
            {"user_input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"\n  👤 {user_input}")
        print(f"  🤖 {response}")


run_conversation("alice-2026-07-07-1400", [
    "Hi, my name is Alice.",
    "I have an issue with my order #A-1234.",
    "The tracking shows delivered but I didn't receive it.",
    "I've tried checking with neighbors and the front desk.",
    "What's the next step?",
    "Can you tell me my name?",
    "And what order number did I mention?",
    "OK, please escalate.",
])


# ─────────────────────────────────────────────────────────────
# Session isolation demo — a DIFFERENT customer gets a fresh history
# ─────────────────────────────────────────────────────────────
run_conversation("bob-2026-07-07-1500", [
    "Hi, my name is Bob.",
    "What's my name?",
])


# ─────────────────────────────────────────────────────────────
# Summarization-based memory (implemented manually)
# ─────────────────────────────────────────────────────────────
print("\n\n" + "═" * 78)
print("Summarization memory (when history grows too long)")
print("═" * 78)

def summarize_old_turns(history: BaseChatMessageHistory, keep_recent: int = 4) -> None:
    """Compress everything before the last `keep_recent` turns into one summary."""
    msgs = history.messages
    if len(msgs) <= keep_recent * 2:
        return    # not big enough yet

    old = msgs[:-keep_recent]
    recent = msgs[-keep_recent:]

    # In prod: LLM summarizes `old`. Mock:
    old_text = " ".join(m.content for m in old if hasattr(m, "content"))
    summary = f"[Prior conversation summary: {old_text[:120]}...]"

    history.clear()
    history.add_message(SystemMessage(content=summary))
    for m in recent:
        history.add_message(m)


alice_hist = get_session_history("alice-2026-07-07-1400")
print(f"\nAlice history before: {len(alice_hist.messages)} messages")

summarize_old_turns(alice_hist, keep_recent=4)

print(f"Alice history after:  {len(alice_hist.messages)} messages")
print(f"Summary message:      {alice_hist.messages[0].content[:80]}...")


print("""

✅ Summary — LangChain memory patterns:

  • InMemoryChatMessageHistory   → dev / prototyping
  • Redis / Postgres history      → prod (swap without changing chain code)
  • RunnableWithMessageHistory    → auto-injects history into every call
  • Session isolation via session_id → multi-tenant safe
  • Summarization pattern         → manual for now, but keeps token cost bounded

Related: LangGraph checkpoints (Phase 12) give you the same across-session
persistence but with STATE (not just messages). Use LangGraph when you need
to resume a paused workflow, not just remember chat.

Next: 04_langchain_rag.py — putting memory + retrieval together.
""")
