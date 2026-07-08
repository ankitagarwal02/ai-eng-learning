# Design 2 — Customer Support Agent Platform

Multi-tenant SaaS: each customer (company) has their own knowledge base,
branded persona, and safety policies.

## Requirements
- Tenants: 500 companies, ~1000 users each = 500k end-users
- Peak: 100 QPS
- SLO: 3-turn resolution rate > 60%, human escalation < 20%
- Compliance: SOC 2, GDPR, per-tenant data isolation

## Architecture

```
                              ┌─────────────────┐
       User ──▶ Widget ──▶ CDN ─▶│  Auth + Tenant │
                              │  resolver       │
                              └────────┬────────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                 ┌─────────────────┐       ┌────────────────┐
                 │ Tenant config   │       │ LLM Agent      │◀─── Tools:
                 │ (Postgres)       │──▶───│  (LangGraph)   │      • create_ticket
                 │  - prompt         │       │                │      • lookup_order
                 │  - allowed tools  │       └───┬────────────┘      • refund_estimate
                 │  - guardrails     │           │                    • escalate_to_human
                 │  - KB pointer     │           ▼                    (from MCP servers
                 └─────────────────┘   ┌──────────────────┐            per-tenant)
                                        │ Tenant vector DB │
                                        │ (Chroma / Qdrant│
                                        │  sharded by      │
                                        │  tenant_id)     │
                                        └──────────────────┘
```

## Key design decisions

**Isolation:** every DB query filters `WHERE tenant_id = ?`. Vector DB uses per-tenant
collections (or physical isolation for Enterprise plan).

**Per-tenant prompts:** each tenant's persona/tone stored in Postgres, injected into
system prompt at runtime. Base prompt + tenant overrides.

**LangGraph for the agent:** because we need cycles (retry with different retrieval),
human-in-the-loop (escalation approval), and checkpoints (resume interrupted session).

**MCP for tools:** each tenant registers their own MCP servers (their CRM, ticket system,
inventory). Standard protocol → tenants can bring their own connectors.

## Human handoff pattern

```
LangGraph state machine:
    understand → search_kb → draft_answer
                                │
                                ▼
                        confidence_check
                        │             │
                   >0.8 ▼             ▼ <0.5
                    send        escalate_to_human
                    to user           │
                                       ▼
                                human_review (LangGraph interrupt)
                                       │
                                       ▼
                              agent resumes with human note
```

## Guardrails per tenant

- Injection detection (Phase 4) — same regex + judge stack across tenants
- Topic scope: tenant's admin picks allowed topics from a taxonomy
- PII redaction: standard patterns + tenant-defined (e.g., FinBank adds account #s)
- Output moderation: OpenAI Moderation API + tenant-custom rules

## Metrics per tenant (dashboard)

- Resolution rate, avg turn count, escalation rate
- Cost per session
- Guardrail block rate
- User satisfaction (thumbs up/down)

## Sample interviewer follow-ups

- How do you prevent one noisy tenant from starving others? → per-tenant rate limits + weighted queues
- How do tenants safely upload new KB documents? → async pipeline: virus-scan → chunk → embed → index
- How do you ensure eval quality across tenants? → tenant-specific golden sets + weekly regression report
