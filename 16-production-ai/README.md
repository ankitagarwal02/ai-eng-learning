# Phase 16 — Production AI

> **Real-life analogy:** Prototyping is a race car for a test track. Production is Le Mans —
> 24 hours of load, weather, traffic, failures. Everything fine at 10 users falls apart at 10,000.

---

## Production checklist (every AI system needs these)

```
✅ Observability
   ─ Structured logging (JSON) for every LLM call
   ─ Trace IDs to link all calls for one user session
   ─ Metrics: TTFT, total latency, cost, tokens, error rate
   ─ Alerts on anomalies

✅ Cost management
   ─ Token counting BEFORE every call
   ─ Model routing (gpt-4o-mini for 80%, gpt-4o for 20%)
   ─ Prompt caching (OpenAI + Anthropic support this)
   ─ Response caching (Redis or similar)
   ─ Batch API for bulk async work (50% discount)

✅ Reliability
   ─ Retries with exponential backoff (Phase 1)
   ─ Fallback models when primary fails
   ─ Circuit breakers
   ─ Rate limit handling

✅ Quality
   ─ Eval suite in CI (Phase 14)
   ─ Red team suite (Phase 4)
   ─ Golden dataset regression check

✅ Safety
   ─ Input + output guardrails (Phase 15)
   ─ PII redaction
   ─ Content moderation
```

## The 3 latency numbers you must know

```
TTFT     Time To First Token    < 500ms feels "instant"; > 2s feels broken
TPOT     Time Per Output Token  streaming — > 100ms is noticeable
Total    end-to-end             sets your SLO
```

## Cost math

```
gpt-4o-mini input:  $0.15 / 1M tokens
gpt-4o-mini output: $0.60 / 1M tokens

gpt-4o input:       $2.50 / 1M tokens  (16× mini)
gpt-4o output:      $10.00 / 1M tokens (16× mini)

Rule: use mini until PROVEN insufficient by evals.
Model routing saves 90-95% typical.
```

## Folder structure

```
16-production-ai/
├── README.md
├── 01_observability.py       ← Structured logs, trace IDs, metrics summary
├── 02_evaluation.py          ← Golden dataset + regression harness
├── 03_guardrails.py          ← Full input+output stack (preview of Phase 15)
├── 04_cost_optimization.py   ← Token counter, model routing, caching, batch
└── requirements.txt
```
