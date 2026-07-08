"""
05_langchain_agents.py — Tool-using agents with LangChain
==========================================================

WHAT THIS FILE TEACHES
----------------------
  • @tool decorator for defining tools
  • create_openai_tools_agent() / AgentExecutor
  • Custom tool with Pydantic schema
  • verbose=True to see the ReAct trace
  • handle_parsing_errors=True for resilience
  • Agent + memory (RunnableWithMessageHistory around the executor)

HOW TO RUN
----------
    pip install langchain langchain-openai
    python 05_langchain_agents.py

REAL-WORLD SCENARIO
-------------------
A research agent with tools:
  search(query)          — web search
  calculate(expr)        — safe arithmetic
  get_stock_price(ticker) — market data (mock)

User asks: "What is Apple's current PE ratio compared to its 5-year average?"
Agent chains: search → get_stock_price → calculate → answer.
"""

import os
import re

try:
    from langchain_core.tools import tool
    from langchain_core.pydantic_v1 import BaseModel, Field
    _has_lc = True
except ImportError:
    _has_lc = False

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


if not _has_lc:
    print("Install: pip install langchain langchain-openai")
    print("This file demonstrates the LangChain agent pattern.")
    exit()


# ─────────────────────────────────────────────────────────────
# 1. Define tools with @tool decorator
# ─────────────────────────────────────────────────────────────
@tool
def search(query: str) -> str:
    """Search the web for a query. Returns a short snippet."""
    fake = {
        "apple pe ratio": "Apple's current P/E ratio is 28.5.",
        "apple 5-year average pe": "Apple's 5-year average P/E ratio is approximately 24.",
        "tesla revenue": "Tesla Q3 revenue was $25B.",
    }
    q = query.lower()
    for k, v in fake.items():
        if all(w in q for w in k.split()):
            return v
    return f"[MOCK] search results for '{query}': no exact match."


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression. Safe — whitelist only."""
    import ast, operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.USub: operator.neg}

    def _eval(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp):    return ops[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):  return ops[type(n.op)](_eval(n.operand))
        raise ValueError(f"disallowed: {ast.dump(n)}")

    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"[calc error] {e}"


class StockArgs(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")


@tool(args_schema=StockArgs)
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker."""
    prices = {"AAPL": 195.50, "TSLA": 245.20, "MSFT": 415.10, "GOOGL": 175.80}
    return f"${prices.get(ticker.upper(), 'unknown')}"


TOOLS = [search, calculate, get_stock_price]


# ─────────────────────────────────────────────────────────────
# 2. Simulate the agent loop (MOCK_MODE — no real LangChain executor call)
# ─────────────────────────────────────────────────────────────
# In production, this section is replaced by:
#
#   from langchain.agents import create_openai_tools_agent, AgentExecutor
#   from langchain_openai import ChatOpenAI
#   from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#
#   prompt = ChatPromptTemplate.from_messages([
#       ("system", "You are a research assistant with access to tools."),
#       ("human", "{input}"),
#       MessagesPlaceholder("agent_scratchpad"),
#   ])
#   agent = create_openai_tools_agent(ChatOpenAI(model="gpt-4o-mini"), TOOLS, prompt)
#   executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True,
#                            handle_parsing_errors=True, max_iterations=6)
#   result = executor.invoke({"input": "..."})

def mock_agent_run(user_input: str) -> dict:
    """Deterministic mock that shows the trace of a multi-step LangChain agent."""
    trace = []
    print(f"\n👤 USER: {user_input}\n")

    low = user_input.lower()
    if "apple" in low and "pe" in low:
        print("[step 1] LLM decides to search for Apple's current PE")
        r1 = search.invoke({"query": "Apple PE ratio"})
        print(f"   ↳ search returned: {r1}")
        trace.append(("search", r1))

        print("\n[step 2] LLM decides to search for 5-year average")
        r2 = search.invoke({"query": "Apple 5-year average PE"})
        print(f"   ↳ search returned: {r2}")
        trace.append(("search", r2))

        # Extract the numbers
        current = re.search(r"P/E ratio is (\d+\.?\d*)", r1).group(1)
        avg = re.search(r"P/E ratio is approximately (\d+\.?\d*)", r2).group(1)

        print(f"\n[step 3] LLM decides to compute ratio {current}/{avg}")
        r3 = calculate.invoke({"expression": f"{current}/{avg}"})
        print(f"   ↳ calculate returned: {r3}")
        trace.append(("calculate", r3))

        answer = (
            f"Apple's current P/E ratio is {current}, and its 5-year average is {avg}. "
            f"The current ratio is {r3} times its 5-year average — modestly above average."
        )
        print(f"\n🤖 FINAL: {answer}")
        return {"output": answer, "trace": trace}

    if "aapl" in low or "stock" in low:
        print("[step 1] LLM decides to get_stock_price for AAPL")
        r = get_stock_price.invoke({"ticker": "AAPL"})
        print(f"   ↳ {r}")
        answer = f"AAPL is currently trading at {r}"
        print(f"\n🤖 FINAL: {answer}")
        return {"output": answer, "trace": [("get_stock_price", r)]}

    return {"output": "I need more context to help.", "trace": []}


# ─────────────────────────────────────────────────────────────
# Run demos
# ─────────────────────────────────────────────────────────────
mock_agent_run("What is Apple's current PE ratio compared to its 5-year average?")
print("\n" + "═" * 78)
mock_agent_run("What is AAPL trading at today?")


print("""

✅ Summary — LangChain agents:

  • @tool decorator + Pydantic args_schema → typed tool definitions
  • create_openai_tools_agent = LLM + prompt + tools = agent
  • AgentExecutor runs the ReAct loop with verbose tracing
  • handle_parsing_errors=True to survive malformed LLM outputs
  • max_iterations to prevent infinite loops

Production tips:
  • ALWAYS log the full trace (executor.invoke returns intermediate_steps)
  • Use LangSmith for automatic trace capture
  • Add early-exit heuristics (agent has enough info → stop)
  • Combine with RunnableWithMessageHistory for multi-turn agents

For CYCLIC workflows (agent revisits its own planning), LangGraph is the
better tool — see Phase 12.
""")
