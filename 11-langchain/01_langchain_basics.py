"""
01_langchain_basics.py — LangChain fundamentals with LCEL
==========================================================

STATUS: Starter — shows the core LCEL idiom. Expand as you build.

WHAT THIS TEACHES
-----------------
• ChatOpenAI initialization
• ChatPromptTemplate.from_messages / .from_template
• StrOutputParser
• The LCEL pipe: prompt | model | parser
• .invoke(), .stream(), .batch()
• RunnableLambda for custom Python functions
• .with_config(run_name=...) for tracing

HOW TO RUN
----------
    pip install langchain langchain-openai
    python 01_langchain_basics.py

SCENARIO
--------
A customer-intent classifier chain: raw customer message →
    COMPLAINT / QUESTION / REFUND_REQUEST / COMPLIMENT
"""

import os

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import StrOutputParser
    from langchain_core.runnables import RunnableLambda
    _has_lc = True
except ImportError:
    _has_lc = False


MOCK_MODE = not os.getenv("OPENAI_API_KEY")


if not _has_lc:
    print("Install: pip install langchain langchain-openai")
    print("\nHere's the pattern this file demonstrates:\n")
    print("""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer intent classifier."),
    ("user", "Classify this message: {msg}\\n\\nRespond with one word: "
             "COMPLAINT, QUESTION, REFUND_REQUEST, or COMPLIMENT.")
])

chain = prompt | model | StrOutputParser()

for msg in ["Where's my order?", "I want my money back!", "Great product!"]:
    print(chain.invoke({"msg": msg}))
""")
else:
    # Real LangChain code path
    if MOCK_MODE:
        # In MOCK_MODE we can't call the real API — show a minimal chain
        # that uses only RunnableLambda so it works without an API key.
        classify = RunnableLambda(lambda x: (
            "REFUND_REQUEST" if "money back" in x["msg"].lower() else
            "COMPLAINT"      if "problem" in x["msg"].lower() else
            "COMPLIMENT"     if "great" in x["msg"].lower() else
            "QUESTION"
        ))
        chain = classify

        for msg in ["Where's my order?", "I want my money back!",
                    "Great product!", "There's a problem with the invoice."]:
            print(f"  '{msg}' → {chain.invoke({'msg': msg})}")
    else:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a customer intent classifier."),
            ("user", "Classify this message: {msg}\n\nRespond with one word: "
                     "COMPLAINT, QUESTION, REFUND_REQUEST, or COMPLIMENT."),
        ])
        chain = (prompt | model | StrOutputParser()).with_config(run_name="intent-classifier")

        for msg in ["Where's my order?", "I want my money back!", "Great product!"]:
            print(f"  '{msg}' → {chain.invoke({'msg': msg})}")

    print("""
✅ LangChain basics:

  • Compose with |: prompt | model | parser
  • .invoke(x), .stream(x), .batch([x1, x2, x3])
  • RunnableLambda wraps any Python function → participates in a chain
  • .with_config(run_name=...) makes traces searchable in LangSmith

Next: 02_lcel_chains.py (Parallel, Passthrough, Branch, Fallbacks).
""")
