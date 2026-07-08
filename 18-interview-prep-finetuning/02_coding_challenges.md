# Coding Challenges (with hints)

Solve each without an LLM framework. Full solutions are elsewhere in the roadmap.

---

**1. Implement `retry_with_backoff` decorator.**  See `01-python-fundamentals/02_functions_and_decorators.py`.

**2. Build a rate limiter that allows N calls per minute.**  See §2 of same file.

**3. Compute cosine similarity between two vectors — no numpy.**
```python
def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(y*y for y in b) ** 0.5
    return dot / (na * nb) if na*nb else 0.0
```

**4. Return top-K most similar strings from a list.**  See `02-ai-ml-basics/02_text_and_tokens.py`.

**5. Implement a sliding-window memory that returns last N messages.**  See `08-agents/02_agent_memory.py`.

**6. Implement a token counter using tiktoken and estimate cost.**  See Phase 16 §cost.

**7. Write a JSON extraction function that survives markdown fences and prose.**  See `03-llm-fundamentals/02_structured_output.py` §2.

**8. Given a schema and free text, extract structured data with Pydantic.**  See `03-llm-fundamentals/02_structured_output.py`.

**9. Build a ReAct agent loop with 4 tools.**  See `08-agents/01_agent_from_scratch.py`.

**10. Implement BM25 for keyword search over a list of docs.**  See `06-vector-databases/04_vector_db_patterns.py`.

**11. Hybrid search: combine BM25 + cosine into one score.**  Same file.

**12. Implement recency scoring for memory entries.**
```python
from datetime import datetime, timedelta
import math
def recency(last_accessed: datetime, half_life_days: float = 20) -> float:
    days = (datetime.now() - last_accessed).days
    return math.exp(-days / half_life_days)
```

**13. Given a golden set, compute precision/recall/F1 per class.**  See `14-evals/01_exact_match_eval.py`.

**14. Detect prompt injection with 10 regex patterns.**  See `04-advanced-prompting-security/02_injection_defense.py`.

**15. Design a StateGraph with 3 nodes and a conditional edge.**  See `12-langgraph/01_langgraph_basics.py`.
