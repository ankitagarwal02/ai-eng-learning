# Design 1 — ChatGPT-style Chatbot at 1M DAU

## Requirements (clarify first)

| Dimension | Assumption |
|-----------|------------|
| Users | 1M DAU, avg 10 turns/session, avg 3 sessions/day |
| Peak QPS | ~3,500 (peak 3× avg) |
| Latency SLO | TTFT p95 < 500ms; total < 3s |
| Correctness | Consumer chat — light hallucination tolerance |
| Cost | Target < $0.005/message |
| Retention | 30 days history for logged-in users |

## Napkin math

```
1M DAU × 30 msg/day    = 30M messages/day
30M / 86,400s          = ~350 QPS average
Peak (3×)              = ~1,050 QPS
Model: gpt-4o-mini, ~200 in-tok + ~150 out-tok = 350 tokens/message
30M msg/day × 350 tok  = 10.5B tokens/day
Cost @ mini pricing    ≈ $3,150/day ≈ $95k/month
```

## Architecture

```
                             ┌─────────────┐
     User ──▶ CDN ──▶ ALB ──▶│  Gateway    │
                             │ (FastAPI)   │
                             └──────┬──────┘
                                    │  auth, rate-limit, input guardrail
                                    ▼
                             ┌─────────────┐
                             │ Orchestrator│───▶ Session store (Redis)
                             │             │    (TTL 1h, sliding window)
                             └──────┬──────┘
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼              ▼
             ┌──────────┐   ┌────────────┐  ┌────────────┐
             │  Router  │   │ Retriever  │  │Prompt cache│
             └────┬─────┘   │(user history│  │  (Redis)   │
                  │         │  vector DB) │  └─────┬──────┘
     ┌────────────┴─────────┐└─────────────┘        │
     ▼                      ▼                       │
 ┌─────────┐          ┌─────────┐                   │
 │gpt-4o-mini│        │ gpt-4o  │                   │
 │  (~80%) │         │  (~20%) │                   │
 └────┬────┘          └────┬────┘                   │
      │                    │                        │
      └──────────┬─────────┘                        │
                 ▼                                  │
         ┌────────────┐                             │
         │ Output      │◀────────────────────────────┘
         │ guardrails  │
         └──────┬─────┘
                │  SSE streaming
                ▼
              User
```

## Critical path latency breakdown

```
Gateway auth + rate limit         5ms
Input guardrail (regex + judge)  20ms
Session load from Redis           5ms
Prompt-cache check                5ms  (30% hit → LLM skipped)
LLM (gpt-4o-mini streaming)     TTFT 300ms
Output guardrail on each chunk    3ms/token
Total TTFT budget                ~350ms   ✅ under 500ms
```

## Scaling considerations

- **Gateway/Orchestrator**: stateless, horizontal scale — 20 pods to handle 1,000 QPS
- **Session store**: Redis Cluster, ~10GB (1M users × 10KB session)
- **Vector DB**: per-user memory shard by user_id — sub-linear growth
- **LLM**: use OpenAI's dynamic capacity + your own provisioned throughput for peak
- **Prompt cache**: 50% off input on cached prefixes (identical system prompts)

## Reliability

- Circuit breaker between orchestrator and LLM (fallback to smaller model on 5xx)
- 3 retries with jittered exponential backoff
- Dead-letter queue for messages that fail all retries → human review

## Cost optimizations applied

1. Model routing: 80% mini (cheap), 20% gpt-4o (only when router says so)
2. Prompt caching: system prompts are stable → 50% off input
3. Response caching: 20-30% queries are near-dupes (Redis, embedding-based hash)
4. Sliding-window memory: last 6 turns instead of full history → 30% fewer input tokens

**Result:** ~$95k/month → ~$25-35k/month with optimizations (~65-70% reduction).

## Follow-up questions interviewer will ask

- How do you handle prompt injection? → Phase 4 defense stack
- How do you evaluate quality changes? → Phase 14 golden set + LLM judge
- How do you keep session context under 128k tokens? → Phase 13 hybrid memory
- What if OpenAI has an outage? → Anthropic Claude fallback, LiteLLM router
- How do you A/B test a new prompt? → shadow traffic, canary 5% → 25% → 100%
