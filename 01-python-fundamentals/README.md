# Phase 1 — Python Fundamentals for AI Engineers

> **Real-life analogy:** Python is to AI what English is to international business. Every major
> AI paper, open-source library, and job posting assumes it. Your goal in this phase is fluency —
> syntax should be invisible so you can focus on the *idea*.

---

## Why Python?

```
┌────────────────────────────────────────────────────────────────┐
│  85% of published ML/AI research code:  Python                 │
│  OpenAI, Anthropic, Google official SDKs primary language:     │
│      Python (with TypeScript/JS as secondary)                  │
│  Hugging Face, LangChain, LlamaIndex, ChromaDB core language:  │
│      Python                                                    │
│  Data scientists' lingua franca:                               │
│      Python (pandas, numpy, matplotlib, sklearn)               │
└────────────────────────────────────────────────────────────────┘
```

Every other language you'd pick has to *translate* AI concepts through wrappers. Python is
native to them.

---

## Concept Table

| Term | Plain-English definition |
|------|--------------------------|
| **List comprehension** | Compact `for`-loop that builds a list in one line |
| **Dict comprehension** | Same idea, but produces a dict `{k: v for ... in ...}` |
| **Decorator** | A function that wraps another function to add behavior (like a middleware) |
| **Closure** | A function that remembers variables from its enclosing scope |
| **`*args` / `**kwargs`** | Accept any number of positional / keyword arguments |
| **`async` / `await`** | Non-blocking concurrency — do many I/O things at once |
| **`asyncio.gather`** | Run several async coroutines in parallel and wait for all |
| **Pydantic `BaseModel`** | Auto-validating typed data class — the LLM/API workhorse |
| **`@field_validator`** | Custom check on one field of a Pydantic model |
| **Type hint** | Annotation like `str`, `List[int]`, `Optional[User]` that tools & humans read |
| **`with` block** | Auto-cleanup context (open a file, close on exit) |
| **`@dataclass`** | Auto-generates `__init__`, `__repr__`, `__eq__` from field declarations |

---

## Code Patterns

### 1. The "safe API call" pattern

```python
@retry(max_attempts=3, backoff=2.0)
def call_llm(prompt: str) -> str:
    return client.chat.completions.create(...).choices[0].message.content
```

### 2. The "parallel I/O" pattern

```python
results = await asyncio.gather(*[call_llm(p) for p in prompts])
```

### 3. The "typed data contract" pattern

```python
class ExtractedInvoice(BaseModel):
    vendor: str
    total: float = Field(gt=0)
    line_items: List[LineItem]

parsed = ExtractedInvoice.model_validate_json(llm_raw_output)
```

### 4. The "config from env" pattern

```python
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
MOCK_MODE = not API_KEY
```

---

## Use Case Matrix

| Scenario | Right technique |
|----------|-----------------|
| Filter 10k tickets by priority | List comprehension |
| Wrap every LLM call with retry logic | Decorator |
| Call GPT-4o + Claude + Gemini together | `asyncio.gather` |
| Parse LLM's JSON output safely | Pydantic `model_validate_json` |
| Read API key without hardcoding | `dotenv` + `os.getenv` |
| Log every call with timing info | Decorator + `logging` |
| Iterate through a huge dataset lazily | Generator (`yield`) |
| Ensure a file closes on exception | `with open(...) as f:` |

---

## The Async I/O Advantage

```
Serial (blocking):        Parallel (asyncio):
GPT-4o    ████░░░░░░       GPT-4o    ████░░░░░░
                                   ┃
Claude    ░░░░████░░       Claude    ████░░░░░░
                                   ┃  (all start together)
Gemini    ░░░░░░░░██       Gemini    ████░░░░░░

Total: 10s                 Total: ~4s (slowest one)
```

That single change is the difference between a chatbot that feels *snappy* and one that feels
*sluggish* in production.

---

## Folder structure

```
01-python-fundamentals/
├── README.md                          ← You are here
├── 01_data_types_and_control.py       ← lists/dicts/sets, comprehensions, control flow
├── 02_functions_and_decorators.py     ← closures, decorators, @retry, functools
├── 03_classes_and_oop.py              ← OOP, dunder methods, dataclasses, ABC
├── 04_async_programming.py            ← async/await, gather, httpx.AsyncClient
├── 05_pydantic_and_typing.py          ← BaseModel, validators, LLM output parsing
├── 06_file_and_env_handling.py        ← dotenv, pathlib, logging, argparse
├── requirements.txt
└── projects/
    └── ticket_classifier/
        └── classifier.py              ← Mini-project: classify 20 support tickets
```

Run every file with `python <filename>.py`. No API key needed — `MOCK_MODE` is on by default.
