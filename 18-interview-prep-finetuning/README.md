# Phase 18 — Interview Prep + Fine-Tuning

Capstone: interview drills + when-to-fine-tune knowledge.

---

## Structure

```
18-interview-prep-finetuning/
├── README.md
├── 01_conceptual_questions.md    ← 60+ Q&A on LLMs, RAG, agents, evals
├── 02_coding_challenges.md       ← 15 problems with reference solutions
├── 03_finetuning_basics.md       ← When to fine-tune vs prompt vs RAG
├── 04_lora_qlora_peft.md         ← Adapter methods explained
└── 05_finetuning_recipe.py       ← Dataset prep → training → merge → serve
```

## The "when to fine-tune" decision tree

```
Do you have >1000 labeled examples of the exact task?
   │
   ├── No  → PROMPT ENGINEER (Phase 3, 4) — start here
   │        │
   │        └─ Still not good enough? RAG (Phase 7)
   │            │
   │            └─ Still not good enough? Fine-tune.
   │
   └── Yes → Fine-tune with LoRA/QLoRA:
             • Task-specific format/tone → 500-5000 examples sufficient
             • New domain knowledge → prefer RAG
             • Cheap inference at high scale → SFT small model
```

## The 3 fine-tuning regimes

| Method | GPU cost | Data needs | When |
|--------|---------|-----------|------|
| **Full fine-tuning** | High (8×A100+) | 10k+ examples | Rarely worth it |
| **LoRA** | Medium (1×A100) | 1k-10k | Most common |
| **QLoRA** | Low (1×3090) | 1k-10k | Broke researcher / home |

LoRA freezes the base model and trains small "adapter" matrices. QLoRA is LoRA
on top of a 4-bit quantized base — same quality, 4× less GPU memory.

---

## Top 20 Interview Questions (see 01_conceptual_questions.md for full 60+)

1. Explain transformer attention in 60 seconds.
2. What are the pros/cons of temperature vs top-p?
3. RAG vs fine-tuning — when do you pick each?
4. Explain the ReAct loop.
5. What is prompt injection, and what are 3 defenses?
6. How would you evaluate a RAG system?
7. Draw the architecture of a customer support agent for 100 tenants.
8. Given a golden set of 100 examples, walk me through prompt A/B testing.
9. What's HNSW? When does it fail?
10. Explain the difference between semantic search and hybrid search.
11. What is a cross-encoder? Why not use it for all retrieval?
12. What's the difference between short-term and semantic agent memory?
13. Design a scalable episodic memory system.
14. What is LangGraph and when do you need it over LangChain?
15. How do you prevent an agent from looping forever?
16. Explain MCP in 2 minutes.
17. What are the 4 RAGAS metrics?
18. What's the cost formula for an OpenAI call?
19. Give 5 concrete cost-optimization techniques.
20. Walk me through a prod-shape LLM observability stack.

Each answer sketch is in `01_conceptual_questions.md`.
