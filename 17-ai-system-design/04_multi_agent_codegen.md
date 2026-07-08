# Design 4 — Multi-Agent Code Generation System

## Requirements
- Given a natural-language task, generate a working PR with tests
- Support Python, TypeScript
- Latency: 3-15 minutes per task
- Cost: < $3 per task
- Must not leak secrets or run destructive commands

## Multi-agent architecture

```
                    ┌────────────────┐
        User task   │    Planner     │  Break task into sub-tasks
        ──────────▶│  (gpt-4o)      │  Emit DAG of work
                    └────────┬───────┘
                             │
                    ┌────────┴────────┐
                    │  Task Dispatcher│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  ┌──────────┐         ┌───────────┐        ┌──────────┐
  │  Coder   │         │  Tester   │        │  Reviewer│
  │(gpt-4o)  │         │(gpt-4o-mini)│      │(gpt-4o)  │
  └────┬─────┘         └─────┬─────┘        └────┬─────┘
       │                     │                    │
       ▼                     ▼                    ▼
   Sandbox                Sandbox              File diff
   (Docker /              (Docker /            static analysis
   Firecracker)           Firecracker)
       │                     │
       └─────────────────────┴───────────┐
                                          ▼
                                  ┌───────────────┐
                                  │Result Aggregator│
                                  └──────┬────────┘
                                         │
                                         ▼
                                    Open PR
                                    (GitHub)
```

## Key considerations

**Tool sandboxing:** every code-execution tool runs in a locked-down Docker/Firecracker VM.
No network access. Filesystem is a scratch volume. CPU/memory capped.

**Parallelism:** independent sub-tasks (implement + write tests + write docs) run in
parallel via `asyncio.gather`. Dependent tasks (build → test) sequential.

**Failure recovery:** LangGraph checkpoints after each sub-task. If a step fails 3×,
escalate to human (interrupt).

**Cost control:**
- Planner + Reviewer use gpt-4o (need reasoning) — ~$0.50 each
- Coder does bulk work — batch its calls if multiple files
- Tester + Documenter use gpt-4o-mini
- Total: ~$1-3/task typical

## Prompt injection defenses (critical for code-gen)

- Never let the model execute arbitrary shell commands unless in sandbox
- Whitelist of allowed CLI tools (git, python, npm; not rm, curl)
- Scan generated code for hardcoded credentials before opening PR
- Human approval gate for any change to CI/CD, infra config, secrets

## Metrics to track

- Task success rate (tests pass, review passes)
- Avg cost per task, p95 cost
- Human intervention rate
- Time-to-PR
- Generated-code security scan pass rate

## Similar systems in production

- OpenAI's SWE-bench solutions
- Anthropic's Claude Code (this environment!)
- Devin, Cognition AI
- GitHub Copilot Workspace
