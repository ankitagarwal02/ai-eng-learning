# Phase 15 — Guardrails

> **Real-life analogy:** Guardrails are the seatbelts, ABS, and lane-departure warnings in a car.
> They don't prevent you from driving — they catch you when something goes wrong.

---

## The guardrail stack

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT GUARDRAILS (before LLM)                              │
│    • Prompt injection detection (Phase 4)                    │
│    • Topic scope enforcement                                 │
│    • Rate limiting                                           │
├─────────────────────────────────────────────────────────────┤
│              LLM CALL                                        │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT GUARDRAILS (before user sees response)               │
│    • PII detection & redaction                               │
│    • Toxicity / moderation                                   │
│    • Format validation (JSON schema, length caps)            │
│    • Grounding check (answer supported by context)           │
└─────────────────────────────────────────────────────────────┘
```

## Action semantics

```
PASS    → let through unchanged
LOG     → let through but record for review
REDACT  → transform (mask PII, truncate, sanitize) and let through
BLOCK   → refuse; return canned refusal
```

## Folder structure

```
15-guardrails/
├── README.md
├── 01_input_guardrails.py       ← Injection detection, topic scope, rate limit
├── 02_output_guardrails.py      ← PII detection, moderation, format check
├── 03_guardrails_framework.py   ← Ordered stack, GuardrailCheck, metrics
└── requirements.txt
```

Prereqs: Phase 4 (injection defense), Phase 3 (structured output).
