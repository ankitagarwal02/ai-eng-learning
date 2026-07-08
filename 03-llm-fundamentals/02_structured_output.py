"""
02_structured_output.py — Reliable structured output from LLMs
==============================================================

WHAT THIS FILE TEACHES
----------------------
• Why "please respond in JSON" is FRAGILE (5-20% parse failures in prod).
• OpenAI `response_format={"type": "json_object"}` — guaranteed JSON syntax.
• OpenAI Structured Outputs with a Pydantic schema — guaranteed matches schema.
• `instructor` library pattern.
• Handling partial/malformed JSON: fallback + retry strategies.
• Nested structured output: extract full order with line items from a paragraph.

HOW TO RUN
----------
    pip install pydantic
    python 02_structured_output.py

REAL-WORLD SCENARIO
-------------------
Extract a structured MedicalRecord (patient name, age, diagnosis, medications, follow-up
date) from a free-text doctor's note. Include a case where the LLM returns a bad total
and validation catches it, then retry succeeds.
"""

import os
import json
import re
from typing import Optional, List
from datetime import date

try:
    from pydantic import BaseModel, Field, field_validator, ValidationError
except ImportError:
    print("Install pydantic: pip install pydantic>=2.0")
    raise

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Why plain "respond in JSON" is fragile
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: The fragile way — bare prompt asking for JSON")
print("=" * 70)

# Real LLMs sometimes wrap JSON in markdown fences, prefix "Sure! Here's your JSON:",
# or truncate mid-object. These are the failure modes:
BAD_OUTPUTS = [
    'Sure! Here is the extraction:\n```json\n{"name": "Alice", "age": 42}\n```',
    '{"name": "Bob", "age": 55,',                                   # truncated
    '{"name": Alice, "age": 42}',                                    # missing quotes
    "Here you go: {\"name\": \"Carla\", \"age\": 71}",              # prefix text
    '{"name": "Dan", "age": "thirty"}',                              # wrong type
]

for i, raw in enumerate(BAD_OUTPUTS, 1):
    try:
        parsed = json.loads(raw)
        print(f"  {i}. ✅ Parsed: {parsed}")
    except json.JSONDecodeError as e:
        print(f"  {i}. ❌ Parse failed: {e.msg[:40]}...  raw={raw[:50]!r}")


# ─────────────────────────────────────────────────────────────
# SECTION 2: The extract-json fallback pattern
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: Robust JSON extraction with regex fallback")
print("=" * 70)

def extract_json_object(text: str) -> Optional[dict]:
    """Find the first {...} block, try to parse. This is your safety net when the
    model wraps JSON in prose or markdown fences."""
    # Strip common markdown fences:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")
    # Greedy match for the outermost {} :
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

for i, raw in enumerate(BAD_OUTPUTS, 1):
    parsed = extract_json_object(raw)
    print(f"  {i}. {'✅' if parsed else '❌'}  {str(parsed)[:60]}")


# ─────────────────────────────────────────────────────────────
# SECTION 3: The RIGHT way — response_format + Pydantic schema
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: Pydantic schema for structured output")
print("=" * 70)

class Medication(BaseModel):
    name: str
    dose: str = Field(..., description="e.g. '500mg', '10ml'")
    frequency: str = Field(..., description="e.g. 'twice daily'")

class MedicalRecord(BaseModel):
    patient_name: str
    age: int = Field(..., gt=0, lt=130)
    diagnosis: str
    medications: List[Medication] = Field(default_factory=list)
    follow_up_date: Optional[date] = None
    notes: Optional[str] = None

    @field_validator("patient_name")
    @classmethod
    def name_not_placeholder(cls, v: str) -> str:
        if v.lower() in {"unknown", "n/a", "patient", ""}:
            raise ValueError("patient_name must be a real name")
        return v.strip()


# The doctor's note we want to parse:
doctor_note = """
Patient: Alice Johnson, age 42.
Presented today with persistent headache and mild hypertension.
Prescribed: Ibuprofen 400mg twice daily; Lisinopril 10mg once daily.
Diagnosis: tension headache with mild hypertension.
Follow-up in 2 weeks — 2026-07-21.
"""

# In real code, you'd use one of these APIs:
#
#   # OpenAI Structured Outputs (guaranteed schema match):
#   response = client.beta.chat.completions.parse(
#       model="gpt-4o-2024-08-06",
#       messages=[...],
#       response_format=MedicalRecord,
#   )
#   record = response.choices[0].message.parsed
#
#   # instructor library:
#   record = instructor.patch(OpenAI()).chat.completions.create(
#       model="gpt-4o-mini",
#       response_model=MedicalRecord,
#       messages=[...],
#   )

# In MOCK_MODE we simulate the LLM's JSON output directly:
mock_llm_json = json.dumps({
    "patient_name": "Alice Johnson",
    "age": 42,
    "diagnosis": "tension headache with mild hypertension",
    "medications": [
        {"name": "Ibuprofen",  "dose": "400mg", "frequency": "twice daily"},
        {"name": "Lisinopril", "dose": "10mg",  "frequency": "once daily"},
    ],
    "follow_up_date": "2026-07-21",
    "notes": "Persistent headache, mild hypertension.",
})

record = MedicalRecord.model_validate_json(mock_llm_json)
print(f"Parsed record: {record.patient_name}, age {record.age}")
print(f"  Diagnosis: {record.diagnosis}")
print(f"  Meds: {[m.name for m in record.medications]}")
print(f"  Follow-up: {record.follow_up_date}")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Validation catches a hallucination — retry logic
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: Bad LLM output → validation error → retry")
print("=" * 70)

# LLM hallucinated: age is a string, patient name is "Unknown", missing medications.
hallucinated_json = json.dumps({
    "patient_name": "Unknown",       # fails custom validator
    "age": "forty-two",              # wrong type
    "diagnosis": "headache",
    "medications": [],
})

def extract_medical_record(note: str, max_retries: int = 2) -> MedicalRecord:
    """The production pattern: try to parse, on failure, feed the ERROR back into
    the retry prompt so the LLM knows exactly what to fix."""
    last_error: str = ""
    for attempt in range(max_retries + 1):
        # Simulate the LLM call: first attempt returns garbage, retry succeeds.
        raw = hallucinated_json if attempt == 0 else mock_llm_json

        try:
            return MedicalRecord.model_validate_json(raw)
        except ValidationError as e:
            last_error = str(e)
            print(f"  Attempt {attempt+1}: ValidationError")
            # Real production: build a corrective retry prompt like:
            #   "Your previous JSON failed validation with these errors:\n"
            #   f"{last_error}\n"
            #   "Please output valid JSON matching the schema."
    raise RuntimeError(f"Failed to extract after retries: {last_error}")

good_record = extract_medical_record(doctor_note)
print(f"✅ Final: {good_record.patient_name}, age {good_record.age}")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Nested extraction — Order with LineItems
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: Nested structured output — full order extraction")
print("=" * 70)

class LineItem(BaseModel):
    sku: str = Field(..., min_length=1)
    name: str
    quantity: int = Field(..., gt=0)
    unit_price: float = Field(..., gt=0)

class Order(BaseModel):
    order_id: str = Field(..., pattern=r"^ORD-\d+$")
    customer_email: str
    line_items: List[LineItem] = Field(..., min_length=1)
    subtotal: float
    tax: float
    total: float

    @field_validator("customer_email")
    @classmethod
    def email_shape(cls, v: str) -> str:
        if "@" not in v or "." not in v:
            raise ValueError("Invalid email")
        return v.lower()


order_text = """
Order ORD-77123 from alice@example.com contains:
  - SKU BLU-001 Bluetooth headphones x2 at $79.99
  - SKU KEY-042 Mechanical keyboard x1 at $129.00
Subtotal: 288.98. Tax: 23.12. Total: 312.10.
"""

# Simulated LLM extraction:
extracted = Order(
    order_id="ORD-77123",
    customer_email="alice@example.com",
    line_items=[
        LineItem(sku="BLU-001", name="Bluetooth headphones", quantity=2, unit_price=79.99),
        LineItem(sku="KEY-042", name="Mechanical keyboard",   quantity=1, unit_price=129.00),
    ],
    subtotal=288.98,
    tax=23.12,
    total=312.10,
)
print(f"Parsed order: {extracted.order_id} for {extracted.customer_email}")
for li in extracted.line_items:
    print(f"  {li.sku} {li.name:<25} qty={li.quantity}  @ ${li.unit_price:>7.2f}")
print(f"Total: ${extracted.total}")

# Show the JSON schema this Pydantic model produces — you literally pass this to OpenAI:
print("\nOpenAI-ready JSON schema (excerpt):")
schema = Order.model_json_schema()
print(json.dumps({k: schema[k] for k in ("title", "type", "required")}, indent=2))


# ─────────────────────────────────────────────────────────────
# SECTION 6: Comparing the 4 approaches
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: Which structured-output approach when?")
print("=" * 70)
print("""
  1. Prompt-only: 'Please respond in JSON'
       ❌ 5-20% failure rate in production. Never rely on this alone.

  2. OpenAI response_format={"type": "json_object"}
       ✅ Guarantees VALID JSON syntax.
       ❌ Does NOT guarantee your fields, types, or values.

  3. OpenAI Structured Outputs with Pydantic (or JSON schema)
       ✅ Guarantees VALID JSON *matching your schema*.
       ✅ Zero manual parsing — .parsed gives a typed object.
       ⚠️  Slightly higher latency; only certain models support it.

  4. `instructor` library + response_model=YourModel
       ✅ Model-agnostic (works with OpenAI, Anthropic, Gemini via LiteLLM).
       ✅ Auto-retry on validation failure with error feedback.
       ⚠️  One more dependency.

Recommendation:  Approach 3 or 4 for all production extraction. Approach 2 as
a floor if your model doesn't support 3. Never approach 1 alone.
""")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
Structured output is the difference between a demo and a production system:

  • Define a Pydantic model — it doubles as validation AND schema
  • Use response_format=YourModel (OpenAI) or instructor.patch (universal)
  • On validation failure, retry with the error injected into the prompt
  • Validate cross-field invariants with @model_validator
  • Never trust raw LLM strings — always parse through a model first

Phase 4 (Advanced Prompting & Security) builds on this: guardrails validate
LLM output the same way, then decide block/redact/pass.
""")
