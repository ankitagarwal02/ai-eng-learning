"""
01_observability.py — Structured logging, trace IDs, metrics for LLM calls
===========================================================================

Every LLM call gets:
  • request_id       for tracing
  • model, tokens, latency, cost
  • Written as JSON to stdout (LangSmith / Datadog / Grafana ingest)
"""

import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class LLMTrace:
    request_id: str
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    error: str | None = None


def price_for(model: str, in_tok: int, out_tok: int) -> float:
    p = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}[model]
    return in_tok * p[0] / 1e6 + out_tok * p[1] / 1e6


def traced_llm_call(prompt: str, model: str = "gpt-4o-mini") -> tuple[str, LLMTrace]:
    req_id = str(uuid.uuid4())
    start = time.perf_counter()

    # Mock LLM call
    in_tok = len(prompt.split())
    out_text = f"[MOCK] {prompt[:40]}..."
    out_tok = len(out_text.split())

    elapsed = (time.perf_counter() - start) * 1000

    trace = LLMTrace(
        request_id=req_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        model=model,
        prompt_tokens=in_tok,
        completion_tokens=out_tok,
        total_tokens=in_tok + out_tok,
        latency_ms=round(elapsed, 2),
        cost_usd=round(price_for(model, in_tok, out_tok), 6),
    )
    print(json.dumps(asdict(trace)))
    return out_text, trace


if __name__ == "__main__":
    traces: list[LLMTrace] = []
    for prompt in ["hello", "summarize the quarterly report", "translate to French"]:
        _, t = traced_llm_call(prompt)
        traces.append(t)

    print("\n" + "=" * 60)
    print("Metrics summary")
    print("=" * 60)
    total_cost = sum(t.cost_usd for t in traces)
    total_tokens = sum(t.total_tokens for t in traces)
    avg_latency = sum(t.latency_ms for t in traces) / len(traces)
    p95 = sorted(t.latency_ms for t in traces)[int(len(traces) * 0.95)]

    print(f"  calls: {len(traces)}")
    print(f"  total tokens:  {total_tokens}")
    print(f"  total cost:    ${total_cost:.6f}")
    print(f"  avg latency:   {avg_latency:.1f} ms")
    print(f"  p95 latency:   {p95:.1f} ms")

    print("""

✅ Observability minimums for every prod LLM app:

  • request_id links every log line for one user session
  • Structured JSON logs → any log aggregator can slice by model, cost, etc.
  • Track TTFT + total latency separately (streaming vs blocking)
  • Alerts: cost > $X/hour, error rate > 1%, latency p95 > 2s

Real integrations: LangSmith, OpenTelemetry + Grafana, Datadog LLM Observability,
Braintrust, Weights & Biases Weave.
""")
