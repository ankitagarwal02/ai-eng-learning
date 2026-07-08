"""
05_pydantic_and_typing.py — Pydantic + type hints: the LLM output contract
==========================================================================

WHAT THIS FILE TEACHES
----------------------
• Python type hints: str, int, Optional, List, Dict, Union, Literal.
• Pydantic BaseModel: auto-validating typed data classes.
• Field(gt=0, description=...) constraints.
• Nested models + composition.
• .model_dump(), .model_validate(), .model_validate_json().
• @field_validator + @model_validator.
• Why Pydantic matters: LLM structured output, FastAPI, agent tool schemas.

HOW TO RUN
----------
    python 05_pydantic_and_typing.py

REAL-WORLD SCENARIO
-------------------
Parse an ExtractedInvoice with nested LineItem objects from a raw JSON string
that an LLM returned. Validate that total == sum of line items. Handle a
malformed payload and recover.
"""

import os
import json
from datetime import date
from decimal import Decimal
from typing import List, Optional, Union, Literal, Dict, Any

# Pydantic is the de-facto standard for structured data in modern Python AI code.
try:
    from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
    _has_pydantic = True
except ImportError:
    _has_pydantic = False
    print("⚠️  pydantic not installed. Run: pip install pydantic>=2.0")
    print("    (Continuing with fallback dataclass-only demo.)")

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Type hints — the free documentation
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Type hints")
print("=" * 70)

def summarize(text: str, max_tokens: int = 200) -> str:
    """Type hints tell IDEs, linters, and future-you what to expect.
    They are checked by mypy/pyright at development time, NOT at runtime."""
    return text[:max_tokens]

# Rich type hints:
def classify(
    text: str,
    categories: List[str],
    threshold: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
) -> Union[str, None]:
    """
    text        : plain string
    categories  : list of strings — e.g. ["billing", "outage"]
    threshold   : float, defaults to 0.5
    metadata    : optional dict OR None
    return      : str OR None
    """
    return categories[0] if categories else None

# `Literal` — restrict to a specific set of allowed values:
Priority = Literal["low", "medium", "high", "critical"]

def route(pri: Priority) -> str:
    return {"low": "queue", "medium": "queue", "high": "on-call", "critical": "pager"}[pri]

print(f"Type-hinted call: route('high') = {route('high')}")
# Static-checkers will flag route("urgent") as an error, at edit time.


if not _has_pydantic:
    print("\n(Skipping Pydantic sections — install pydantic to run them.)")
else:

    # ─────────────────────────────────────────────────────────────
    # SECTION 2: Basic BaseModel
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 2: Basic BaseModel")
    print("=" * 70)

    class User(BaseModel):
        id: int
        email: str
        is_active: bool = True     # default
        signup_date: Optional[date] = None

    # Pydantic COERCES compatible types by default:
    u = User(id="42", email="a@b.com", signup_date="2026-01-15")
    print(f"User: {u}")
    print(f"id is int? {isinstance(u.id, int)}   signup_date is date? {isinstance(u.signup_date, date)}")

    # Invalid input raises a rich error:
    try:
        User(id="not-a-number", email="x@y.com")
    except ValidationError as e:
        print(f"\nValidation error caught:\n{e}")


    # ─────────────────────────────────────────────────────────────
    # SECTION 3: Field constraints
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 3: Field constraints")
    print("=" * 70)

    class ChatRequest(BaseModel):
        model: Literal["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"] = "gpt-4o-mini"
        message: str = Field(..., min_length=1, max_length=4000, description="User message")
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        max_tokens: int = Field(1024, gt=0, le=128000)
        top_p: float = Field(1.0, gt=0.0, le=1.0)

    good = ChatRequest(message="hi", temperature=0.5)
    print(f"OK: {good}")

    try:
        ChatRequest(message="", temperature=3.0)
    except ValidationError as e:
        print(f"\nRejected bad request:\n{e}")


    # ─────────────────────────────────────────────────────────────
    # SECTION 4: Nested models — the LLM structured output pattern
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 4: Nested models — ExtractedInvoice")
    print("=" * 70)

    class LineItem(BaseModel):
        description: str = Field(..., min_length=1)
        quantity: int = Field(..., gt=0)
        unit_price: float = Field(..., gt=0)

        @property
        def total(self) -> float:
            return self.quantity * self.unit_price

    class ExtractedInvoice(BaseModel):
        invoice_id: str = Field(..., pattern=r"^INV-\d{4,}$")
        vendor: str
        issue_date: date
        line_items: List[LineItem] = Field(..., min_length=1)
        subtotal: float = Field(..., ge=0)
        tax: float = Field(..., ge=0)
        total: float = Field(..., ge=0)
        notes: Optional[str] = None

        @field_validator("vendor")
        @classmethod
        def vendor_title_case(cls, v: str) -> str:
            """Normalize vendor to title-case."""
            return v.strip().title()

        @model_validator(mode="after")
        def check_totals_match(self) -> "ExtractedInvoice":
            """Cross-field invariant: subtotal + tax must equal total (within 1 cent)."""
            computed = round(self.subtotal + self.tax, 2)
            if abs(computed - self.total) > 0.01:
                raise ValueError(
                    f"total ({self.total}) != subtotal + tax ({computed})"
                )
            return self

    # Simulate LLM output as a raw JSON string:
    llm_output_good = """
    {
        "invoice_id": "INV-2026-0042",
        "vendor": "acme corp",
        "issue_date": "2026-07-01",
        "line_items": [
            {"description": "Consulting", "quantity": 10, "unit_price": 150.0},
            {"description": "Licence",    "quantity": 1,  "unit_price": 500.0}
        ],
        "subtotal": 2000.0,
        "tax": 200.0,
        "total": 2200.0,
        "notes": "Net-30"
    }
    """

    invoice = ExtractedInvoice.model_validate_json(llm_output_good)
    print(f"Parsed vendor (normalized): {invoice.vendor}")
    for li in invoice.line_items:
        print(f"  - {li.description:<12} qty={li.quantity:<3} @ ${li.unit_price:>7.2f} = ${li.total:>8.2f}")
    print(f"Total: ${invoice.total}")


    # ─────────────────────────────────────────────────────────────
    # SECTION 5: Handling bad LLM output — total mismatch
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 5: Cross-field validation catches an LLM hallucination")
    print("=" * 70)

    llm_output_bad = """
    {
        "invoice_id": "INV-2026-0043",
        "vendor": "widget co",
        "issue_date": "2026-07-02",
        "line_items": [
            {"description": "Item A", "quantity": 2, "unit_price": 100.0}
        ],
        "subtotal": 200.0,
        "tax": 20.0,
        "total": 9999.0
    }
    """

    try:
        ExtractedInvoice.model_validate_json(llm_output_bad)
    except ValidationError as e:
        print("Caught invalid invoice — the LLM hallucinated the total.")
        print(f"  → {e.errors()[0]['msg']}")


    # ─────────────────────────────────────────────────────────────
    # SECTION 6: .model_dump() — going back to dict / JSON
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 6: Serialization — .model_dump / .model_dump_json")
    print("=" * 70)

    as_dict = invoice.model_dump()
    as_json = invoice.model_dump_json(indent=2)
    print(f"Round-trip dict keys: {list(as_dict.keys())}")
    print(f"\nSerialized JSON (first 200 chars):\n{as_json[:200]}...")


    # ─────────────────────────────────────────────────────────────
    # SECTION 7: Agent tool schema — Pydantic → OpenAI function spec
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 7: BaseModel → OpenAI tool schema")
    print("=" * 70)

    class SearchTool(BaseModel):
        """Search the company knowledge base."""
        query: str = Field(..., description="The search query")
        top_k: int = Field(5, ge=1, le=20, description="Number of results")

    schema = SearchTool.model_json_schema()
    print(f"JSON schema (feed this to OpenAI's tools parameter):")
    print(json.dumps(schema, indent=2))


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
Pydantic is the contract you enforce between untrusted text (LLM output,
HTTP requests, user input) and your trusted code:

  • Type hints alone give docs + IDE help + static checks
  • BaseModel adds runtime validation + coercion
  • Field(gt=..., ge=..., min_length=...) declares invariants
  • @field_validator/@model_validator handle custom + cross-field checks
  • .model_validate_json() parses AND validates in one call
  • .model_json_schema() gives you the OpenAI-compatible tool schema

Rule: if the data crossed a trust boundary (LLM, HTTP, disk, other service),
run it through a Pydantic model before using it. Never trust raw dicts.

Next file: 06_file_and_env_handling.py — .env, pathlib, logging, argparse.
""")
