# Phase 10 — Multi-Agent Systems

> **Real-life analogy:** A company with departments. The CEO (supervisor) delegates to
> specialists (research, writing, QA). Each is expert at one thing.

---

## Three canonical patterns

### 1. Supervisor

One agent decomposes the task and dispatches to specialists.

```
                ┌── ResearchAgent ──┐
Supervisor ────┼── WriterAgent ────┤──▶ Supervisor aggregates ──▶ Final
                └── FactCheckAgent ─┘
```

Best for: open-ended tasks where the plan isn't known upfront.

### 2. Pipeline

Fixed sequence with stage gates.

```
Extractor ─▶ Classifier ─▶ Summarizer ─▶ Formatter
              (stage gate: only proceed if validation passes)
```

Best for: known workflows (document processing, ETL, batch pipelines).

### 3. Parallel (Fan-out / Fan-in)

All specialists run at once, results merged.

```
        ┌─── LegalRiskAgent ────┐
Query ──┼─── FinancialAgent ────┼──▶ Merger ──▶ Report
        └─── ComplianceAgent ───┘
```

Best for: independent analyses of the same input.

## Folder structure

```
10-multi-agents/
├── README.md
├── 01_supervisor_pattern.py
├── 02_pipeline_pattern.py
├── 03_parallel_agents.py
└── requirements.txt
```

Prereqs: Phase 8 (single agents), Phase 4 (async for parallel).
