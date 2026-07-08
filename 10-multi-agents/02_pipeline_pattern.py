"""
02_pipeline_pattern.py — Fixed sequential pipeline with stage gates
=====================================================================

SCENARIO
--------
Document processing:  Extractor → Classifier → Summarizer → Formatter
Stage 2 only runs if Stage 1 validation passes. Errors are recoverable.
"""


def extractor(raw: str) -> dict:
    return {"text": raw.strip(), "word_count": len(raw.split())}

def classifier(state: dict) -> dict:
    text = state["text"].lower()
    if "invoice" in text or "$" in text:
        state["category"] = "financial"
    elif "meeting" in text or "agenda" in text:
        state["category"] = "meeting"
    else:
        state["category"] = "other"
    return state

def summarizer(state: dict) -> dict:
    state["summary"] = state["text"][:60] + "..."
    return state

def formatter(state: dict) -> str:
    return f"[{state['category'].upper()}] ({state['word_count']} words) {state['summary']}"


def run_pipeline(raw: str) -> str:
    stages = [
        ("extractor", extractor,
            lambda s: s["word_count"] > 3),                   # gate: at least 4 words
        ("classifier", classifier,
            lambda s: s["category"] != "other"),              # gate: recognized category
        ("summarizer", summarizer,
            lambda s: "summary" in s and len(s["summary"]) > 0),
        ("formatter", formatter,
            lambda s: isinstance(s, str)),
    ]

    state = raw
    for name, fn, gate in stages:
        try:
            print(f"  → {name}")
            state = fn(state)
            if not gate(state):
                print(f"  ⚠️  stage gate failed at {name} — skipping downstream")
                break
            print(f"    ✓ state: {str(state)[:80]}...")
        except Exception as e:
            print(f"  ✗ {name} error: {e}")
            break

    return state if isinstance(state, str) else str(state)


if __name__ == "__main__":
    for raw in [
        "Invoice INV-1023 dated 2026-07-01 for $2,200 from Acme Corp.",
        "Sales meeting agenda: Q3 pipeline review, forecast update, and next-quarter goals.",
        "Hello",   # too short — should fail extractor gate
    ]:
        print(f"\nInput: {raw[:60]}...")
        result = run_pipeline(raw)
        print(f"Output: {result}")

    print("\n✅ Pipeline pattern — best for known, structured workflows.")
