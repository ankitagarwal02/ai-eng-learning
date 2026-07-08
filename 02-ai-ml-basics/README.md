# Phase 2 — AI / ML Basics

> **Real-life analogy:** Teaching a child to recognize cats by showing 1000 photos vs writing
> explicit rules ("has whiskers AND fur AND four legs..."). ML *finds* patterns from examples;
> you supply the examples and the objective, not the rules.

---

## Why this phase matters for LLM/agentic work

Modern LLM engineering isn't ML from scratch anymore — but **every concept below shows up daily**:

| ML concept | Where it appears in LLM/agent work |
|-----------|-----------------------------------|
| Train / test split | Building golden eval sets |
| Precision / recall / F1 | Evaluating a classifier LLM, a router, a guardrail |
| Overfitting | Prompt over-specialized to a benchmark |
| Feature engineering | Prompt design, retrieved-context assembly |
| Embeddings | Vector search, semantic memory, RAG |
| Cosine similarity | Retrieval scoring, dedup, clustering |
| Tokens | Cost, latency, context-window budgeting |
| Confusion matrix | Debugging why a classifier fails on class X |

You can't skip these. They are the vocabulary.

---

## The ML Workflow (universal)

```
┌───────────┐   ┌───────────┐   ┌────────┐   ┌─────────┐   ┌────────┐
│   Data    │─▶│  Prep /    │─▶│ Train  │─▶│ Evaluate │─▶│ Deploy │
│ Collection│   │ Features   │   │ Model  │   │  (test)  │   │        │
└───────────┘   └───────────┘   └────────┘   └─────────┘   └────────┘
      ▲                                            │
      └────────────── feedback loop ───────────────┘
```

In LLM land the "training" step is often just prompt-engineering or fine-tuning, but
**every other stage is identical**.

---

## Concept Table

| Term | Plain-English |
|------|---------------|
| **Supervised learning** | Learn from labeled examples (spam vs not-spam) |
| **Unsupervised learning** | Find structure in unlabeled data (clustering) |
| **Classification** | Predict a discrete category |
| **Regression** | Predict a continuous number |
| **Overfitting** | Model memorized training data, fails on new data |
| **Underfitting** | Model too simple to capture the pattern |
| **Precision** | Of items you flagged positive, how many were actually positive |
| **Recall** | Of items that were actually positive, how many you caught |
| **F1** | Harmonic mean of precision & recall |
| **Confusion matrix** | Table of TP / FP / TN / FN — the diagnostic ground-truth |
| **Token** | A sub-word unit — "quick" = 1 token, "tokenization" ≈ 3 |
| **Embedding** | Fixed-length vector that captures meaning of text |
| **Cosine similarity** | Angle between two vectors — 1 identical, 0 orthogonal, -1 opposite |
| **Semantic search** | Retrieve by *meaning* using embeddings, not by keyword |

---

## Precision vs Recall — the intuition that always trips people up

```
   Reality:         SPAM         NOT SPAM
Model says SPAM │  ✅ TP      │  ❌ FP  (false alarm)
Model says NOT  │  ❌ FN      │  ✅ TN  (missed spam)
                     recall        precision
```

- **Precision high, recall low**: "I'm cautious — only flag SPAM if I'm sure." Users don't see false alarms, but real spam slips through.
- **Recall high, precision low**: "I flag anything suspicious." Users see false alarms constantly.

**Rule of thumb per domain:**

| Domain | Which matters more? |
|--------|---------------------|
| Spam filter | Precision (avoid blocking real emails) |
| Fraud detection | Recall (avoid missing real fraud) |
| Medical screening (initial) | Recall (never miss a case) |
| Loan approval risk model | Precision (avoid bad loans) |
| RAG retrieval | Both matter — see Phase 14 RAGAS |

---

## Embeddings — the bridge from ML to LLMs

```
Text: "wireless bluetooth headphones"
        │
        ▼  (embedding model)
Vector: [ 0.13, -0.42, 0.87, ..., 0.05 ]   ← 1,536 dimensions typically
```

Two texts are "similar" if their vectors point in a similar direction:

```
     v1: "bluetooth earbuds"     ─┐
                                   ├── cosine ≈ 0.85  (similar)
     v2: "wireless headphones"   ─┘

     v3: "car engine oil"        ─── cosine ≈ 0.12  (unrelated)
```

This one idea unlocks:
- **Semantic search** (Phase 6)
- **RAG retrieval** (Phase 7)
- **Agent long-term memory** (Phase 13)
- **Clustering / topic discovery** (unsupervised)
- **Deduplication of near-identical docs**

---

## Tokens — why they dominate cost & latency

```
Input:  "The quick brown fox jumps over the lazy dog"
Tokens: ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog"]
Count:  9 tokens

Cost formula:
    cost = (input_tokens × input_price + output_tokens × output_price) / 1_000_000

gpt-4o-mini:  input $0.15/1M  output $0.60/1M   (as of 2026)
gpt-4o:       input $2.50/1M  output $10.00/1M

10× cost difference — often 2× quality difference.
Rule: use gpt-4o-mini until you PROVE gpt-4o helps.
```

---

## Use Case Matrix

| Scenario | Which technique |
|----------|-----------------|
| Route customer message to a queue | Classifier (small model + labeled data) |
| Predict tomorrow's sales | Regression |
| Group similar support tickets | Clustering (KMeans on embeddings) |
| Find similar past cases to this one | Cosine similarity on embeddings |
| Deduplicate near-identical docs | Embedding cosine > threshold |
| Estimate LLM cost before calling | `tiktoken` token count × price |
| Diagnose why classifier fails on class X | Confusion matrix |
| A/B test a new prompt | Evaluate on golden set, compare P/R/F1 |

---

## Folder structure

```
02-ai-ml-basics/
├── README.md                     ← You are here
├── 01_ml_core_concepts.py        ← sklearn workflow, spam classifier, confusion matrix
├── 02_text_and_tokens.py         ← tiktoken, embeddings, cosine, semantic search
└── requirements.txt
```

Run each `.py` with `python <file>.py`. MOCK_MODE lets you run without API keys —
embeddings are simulated with deterministic random vectors so the *shape* of the
computation is correct.
