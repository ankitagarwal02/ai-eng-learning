# Phase 17 — AI System Design (Interview-Grade)

> Interview-focused capstone. Whiteboard-ready designs for the most common
> agentic/AI system design questions.

---

## The AI System Design Framework (use for every question)

```
1. CLARIFY REQUIREMENTS (5 min)
   ─ Scale: QPS, concurrent users, data volume, latency SLO
   ─ Correctness bar: hallucination tolerance, safety requirements
   ─ Cost target: $/month or $/query
   ─ Traffic pattern: peak/average, geographic distribution
   ─ Data: private vs public, PII, retention

2. HIGH-LEVEL COMPONENTS (5 min)
   ─ Gateway (auth, rate limit, guardrails)
   ─ Orchestrator (routing, session management)
   ─ LLM(s) with fallback
   ─ Vector DB (if RAG)
   ─ Tool / MCP servers (if agent)
   ─ Cache (Redis)
   ─ Storage (Postgres, S3)
   ─ Observability (Prometheus, LangSmith)

3. DEEP DIVE ON CRITICAL PATH (10 min)
   ─ Latency budget breakdown
   ─ Cost per query breakdown
   ─ Bottleneck identification & mitigation
   ─ Failure modes

4. SCALE (5 min)
   ─ Horizontal scaling of stateless components
   ─ Sharding vector DB
   ─ Caching strategy at each layer
   ─ Multi-region

5. RELIABILITY & SAFETY (5 min)
   ─ Circuit breakers, fallback models
   ─ Prompt injection defense
   ─ PII redaction
   ─ Eval suite in CI, canary deploys
```

## Five Canonical Designs (in this phase)

1. **Design a ChatGPT clone** (1M DAU) — streaming, model routing, cost
2. **Customer-support agent platform** — multi-tenant, tools, guardrails
3. **Enterprise RAG over 100M documents** — sharding, incremental indexing
4. **Multi-agent code generation system** — parallelism, tool sandboxing
5. **Real-time voice AI** — STT → LLM → TTS, latency budget

## Common latency budgets

```
Chat (streaming):   TTFT < 500ms, first sentence < 1s
Voice AI:           total round trip < 800ms (STT+LLM+TTS)
Agent (multi-tool): each tool step 1-3s, full task < 30s
Batch RAG:          sub-second per doc, thousands/sec throughput
```

## Folder structure

```
17-ai-system-design/
├── README.md
├── 01_chatgpt_clone.md
├── 02_support_agent_platform.md
├── 03_enterprise_rag.md
├── 04_multi_agent_codegen.md
└── 05_realtime_voice_ai.md
```
