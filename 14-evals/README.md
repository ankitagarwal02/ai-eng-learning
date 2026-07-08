# Phase 14 — Evaluation

> **Real-life analogy:** You wouldn't ship a car without crash tests. Evals are your AI's
> crash test — systematic, repeatable, quantitative measurement.

---

## 4 Levels of Eval

```
Level 1  Unit         one function/prompt, exact-match on known inputs
Level 2  Integration  end-to-end pipeline (RAG, agent) on realistic queries
Level 3  Regression   compare new prompt/model vs known baseline
Level 4  A/B          production traffic split, compare metrics live
```

## Metrics by task type

| Task | Metric |
|------|--------|
| Classification | Accuracy / Precision / Recall / F1 |
| Extraction (structured) | Field-level exact-match, JSON validity |
| Open-ended generation | LLM-as-judge (rubric scoring) |
| RAG | RAGAS (faithfulness, answer relevancy, context precision/recall) |
| Agents | Task completion rate, tool-call correctness |

## Quality Gate template

```
if pass_rate < 0.95:               exit(1)   # block deploy
if any(f.severity == "critical"):  exit(1)   # regression on P0 test
if avg_llm_judge_score < 3.5:      exit(1)   # quality below bar
```

## Folder structure

```
14-evals/
├── README.md
├── 01_exact_match_eval.py    ← EvalCase / EvalReport / classifier evaluation
├── 02_llm_as_judge.py        ← Rubric scoring, pairwise comparison
├── 03_ragas_evaluation.py    ← 4 RAGAS metrics from scratch
└── requirements.txt
```
