# Fine-tuning Basics

## The decision tree

```
Question: "Should I fine-tune this LLM?"

    Do I have >1000 labeled examples of exactly this task?
    ├── NO   → Prompt-engineer (Phase 3, 4)
    │        └─ Try few-shot, CoT, structured output
    │           If still not good enough:
    │           └─ Add RAG (Phase 7)
    │              If STILL not good enough → collect data → fine-tune
    │
    └── YES  → What are you trying to change?
                ├── Style/tone/format:  FINE-TUNE  (500-5000 examples)
                ├── Domain knowledge:   RAG usually beats fine-tuning
                ├── Cheap inference:    FINE-TUNE a small model
                └── Reasoning/logic:    fine-tuning rarely helps; better prompts do
```

## The 3 flavors of fine-tuning

### 1. SFT (Supervised Fine-Tuning) — what everyone means by "fine-tune"

Given labeled (input, ideal_output) pairs, adjust weights so model matches
ideal outputs. Loss computed only on the output tokens.

### 2. RLHF / DPO — preference-based

Given pairs of (better, worse) responses, train the model to prefer the better one.
DPO (Direct Preference Optimization) is the modern practical approach — no reward model needed.

### 3. Continued pre-training

Feed the model raw text from your domain to learn vocabulary and patterns.
Rarely worth it — RAG usually replaces this.

## When RAG > fine-tuning

| Symptom | Prefer |
|---------|--------|
| Model doesn't know your docs | RAG |
| Docs change weekly | RAG (fine-tune stales fast) |
| Need exact quotes / citations | RAG |
| Need general Q&A over corpus | RAG |

## When fine-tuning > RAG

| Symptom | Prefer |
|---------|--------|
| Need consistent tone/format | Fine-tune |
| Need narrow task (classify to 5 labels) reliably | Fine-tune |
| Latency needs base < 100ms | Fine-tune a small model |
| Cost > $1/query and huge scale | Fine-tune to a smaller model |

## When you need BOTH

Most serious production systems: fine-tune a **small model** for style + task,
then use **RAG** to inject fresh context. Ship a 7B/8B fine-tuned model
running on your own GPU + a vector DB for retrieval.

Result: cheaper than gpt-4o + fresher than base fine-tune.

## Data prep checklist for SFT

1. **De-duplicate** — near-identical examples inflate rare-case importance.
2. **Balance** — if classifying, roughly equal examples per class.
3. **Quality** — manually review 10% of your set; labels wrong → model wrong.
4. **Format** — same messages structure your model expects at inference.
5. **Split** — 80% train / 10% val / 10% test. Never let test leak into train.
6. **Volume** — 500 for style tweaks, 2000-5000 for task specialization,
   10k+ for behavior changes.

Bad training data = bad model. Always spend time here first.
