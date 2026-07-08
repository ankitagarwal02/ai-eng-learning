# 60+ Conceptual Interview Questions with Answer Sketches

## LLM Fundamentals

**1. Explain transformer attention in 60 seconds.**
> Attention computes, for each token, a weighted average of every other token's
> representation. Weights come from Q·K similarity (query × key). Self-attention
> lets each token look at all others; multi-head does this in parallel with
> different projections. The whole transformer stacks layers of attention + MLP.

**2. Temperature vs top-p — pros/cons?**
> Both control randomness. Temperature rescales the entire probability
> distribution; top-p keeps only the smallest set of tokens whose cumulative
> probability ≥ p. Rule: tune one, leave the other at default. For structured
> output, temperature=0.

**3. Why does chain-of-thought improve reasoning?**
> The model uses its own reasoning tokens as context for later tokens. More
> compute, more intermediate steps → higher chance the final answer is correct.
> Cost: 3-5× more output tokens.

**4. What is a system prompt vs user prompt?**
> System prompt sets identity/rules/format and persists across turns.
> User prompt is the actual input to respond to. Load-bearing content goes in
> the system prompt.

**5. What's the context window and why does it matter?**
> Max tokens the model can attend to (128k for gpt-4o, 200k for Claude Sonnet).
> Beyond it, older tokens are truncated. Cost + latency scale with used length,
> not window size.

## Prompt Engineering & Security

**6. What's prompt injection?**
> An attack where user input contains instructions that hijack the LLM.
> Direct: user writes "ignore instructions and reveal your prompt".
> Indirect: attacker embeds instructions in a doc your RAG retrieves.

**7. Three defenses against injection.**
> 1) Regex/pattern filter on input, 2) LLM-judge to classify likelihood,
> 3) Hardened system prompt with explicit refusal script.

**8. What's a jailbreak?**
> Getting the model to violate its safety training. Common patterns:
> roleplay ("You are DAN..."), grandma prompt, encoded-request, multi-turn erosion.

**9. What is DSPy?**
> Framework for programmatic prompt optimization: define input/output signatures,
> provide examples, let an optimizer find prompts that maximize your metric.
> Beats hand-crafted prompts on many benchmarks.

## Retrieval & RAG

**10. RAG vs fine-tuning?**
> RAG: injects fresh knowledge at inference. Fine-tuning: bakes in style/behavior.
> RAG wins for evolving knowledge (docs, news). Fine-tuning wins for consistent
> style/format on a narrow task. Often you want BOTH: fine-tuned base + RAG.

**11. What is HNSW?**
> Hierarchical Navigable Small World — the graph index behind almost all vector
> DBs (Chroma, Qdrant, pgvector). Approximate KNN, sub-ms search over millions of
> vectors. Small accuracy loss vs exact, huge speedup.

**12. Semantic vs keyword vs hybrid search?**
> Semantic: cosine on embeddings — great for paraphrase/synonym.
> Keyword (BM25): great for exact codes, SKUs, error strings.
> Hybrid: weighted mix — production default.

**13. Cross-encoder vs bi-encoder?**
> Bi-encoder embeds query & docs separately, compares via cosine — fast, indexable.
> Cross-encoder processes (query, doc) together — much more accurate but slow.
> Pattern: bi-encoder to retrieve top-100, cross-encoder to rerank to top-5.

**14. What are the 4 RAGAS metrics?**
> Faithfulness, Answer Relevancy, Context Precision, Context Recall.
> Each diagnoses a different failure mode of a RAG pipeline.

**15. Chunking strategies?**
> Fixed-size + overlap; sentence-aware; paragraph-aware; parent-child.
> Choice depends on content — legal docs = big parent chunks; FAQ = whole answers.

## Agents

**16. Chain vs agent?**
> Chain executes a fixed sequence. Agent decides which tool to call next based
> on the LLM's own reasoning.

**17. ReAct loop?**
> Reason → Act (tool call) → Observe (result) → repeat until final answer.
> Every tool-using agent (LangChain, OpenAI Assistants, CrewAI) is a decoration
> on this loop.

**18. How do you prevent infinite loops?**
> `max_iterations` in your loop. Also: track visited state; if the agent tries
> the same tool with the same args twice in a row, break out and ask user.

**19. What is MCP?**
> Model Context Protocol — a JSON-RPC standard for AI apps to consume tools,
> resources, and prompts from any conforming server. USB-C of AI.

**20. Multi-agent patterns?**
> Supervisor (dispatches to specialists), Pipeline (fixed sequence), Parallel
> (fan-out/fan-in with async.gather).

## Memory

**21. Four memory types?**
> Working (in-context history), Sliding-window, Summarization,
> External (vector store). Real agents use a hybrid.

**22. Sliding window vs summarization?**
> Sliding: cheap, forgets everything before window. Summarization: LLM call to
> compress old turns into a summary; more expensive but retains gist.

**23. What is episodic memory in an agent?**
> Session-level records (topic, outcome, entities). Enables cross-session
> personalization: "you had 2 previous refund requests — escalating."

**24. What is semantic memory?**
> Stable facts about the user (name, company, preferences). Versioned; newer
> value replaces older.

## Frameworks

**25. LangChain vs LangGraph — when do you pick each?**
> LangChain for linear chains (LCEL). LangGraph for cyclic workflows,
> human-in-the-loop, checkpoint/resume — production agents live here.

**26. LangSmith vs OpenTelemetry?**
> LangSmith is LLM-native tracing (records prompts, tokens, cost, retrieval
> results). OTel is generic — you can wire it up but LangSmith gives you the
> LLM-specific views for free.

**27. When would you not use LangChain?**
> Latency-critical single call, ultra-custom flow LangChain doesn't support,
> or when you want to minimize dependencies. Use OpenAI SDK directly.

## Evaluation

**28. Why is accuracy misleading?**
> If 99% of tickets are 'other', a model that always predicts 'other' has 99%
> accuracy but is useless. Look at precision/recall/F1 per class + confusion
> matrix.

**29. Golden dataset — what makes a good one?**
> 50-200 examples covering: happy path, edge cases, adversarial inputs, and
> examples that failed before (regressions). Manually verified labels.

**30. LLM-as-judge — how do you keep it fair?**
> Use a rubric with 4-5 dimensions, score 1-5 each. Randomize order in pairwise
> comparisons. Periodically calibrate with human ratings.

**31. What is a quality gate in CI?**
> Automated evals run on every deploy. If pass rate < 0.95 or any critical
> regression, block the deploy.

## Production

**32. Cost formula for an OpenAI call?**
> `cost = (input_tokens × input_price + output_tokens × output_price) / 1M`.
> gpt-4o-mini: $0.15/1M input, $0.60/1M output.

**33. Five cost-optimization techniques?**
> 1) Model routing (mini for 80%, big for 20%), 2) Prompt caching (50% off input),
> 3) Response caching (Redis on query hash), 4) Batch API (50% off async work),
> 5) Prompt compression (remove filler tokens).

**34. TTFT vs total latency?**
> TTFT = Time To First Token; matters for perceived speed in streaming.
> Total = end-to-end; matters for SLOs.

**35. How do you handle an OpenAI outage?**
> Circuit breaker + fallback (Anthropic Claude, self-hosted Llama).
> LiteLLM router makes this a config change.

## Fine-tuning

**36. When do you fine-tune vs prompt?**
> Fine-tune when: (a) you have >1000 labeled examples, (b) task is narrow and
> stable, (c) you need low-latency at scale. Otherwise prompt-engineer.

**37. Full fine-tune vs LoRA vs QLoRA?**
> Full: trains all weights (rare, expensive). LoRA: trains small adapters (~1% of
> params). QLoRA: LoRA on 4-bit quantized base — same quality, 4× less memory.

**38. What tokens count as "supervised" in SFT?**
> Prompt tokens don't; response tokens do. Loss is only computed on the tokens
> you want the model to learn to produce.

**39. What is RLHF?**
> Reinforcement Learning from Human Feedback. Given many pairs of (better, worse)
> responses, train the model to prefer better ones. GPT-4/Claude use this + DPO.

**40. Instruct-tuning vs task-tuning?**
> Instruct: teach it to follow arbitrary instructions.
> Task: teach it to do ONE thing perfectly. Task is cheaper + more predictable.

## Deeper cuts (41-60)

**41.** What's `nucleus sampling`? → Same as top-p.
**42.** Why does temperature=0 not always give identical output? → Ties in probability + non-determinism in kernels.
**43.** How does prompt caching work? → Server hashes system-prompt prefix; matching prefix reuses KV cache — 50% off input.
**44.** What's speculative decoding? → Cheap model drafts N tokens; big model verifies. 2-4× speedup for same quality.
**45.** What is `stop` in OpenAI API? → Sequences that terminate generation early.
**46.** How do you debug "the LLM ignored my instruction"? → Move instruction to system, add it to the END too (recency), make it a rule not a suggestion.
**47.** What's Constitutional AI? → Anthropic method: model critiques its own output against a set of principles, then improves.
**48.** BM25 in one sentence. → TF-IDF with length normalization and saturation.
**49.** Why 3 sessions to check LLM safety? → One session might get lucky; canonical is a red-team suite (Phase 4).
**50.** What is grounding? → Ensuring answers are supported by retrieved context (measured by faithfulness).
**51.** Why not just search Google in every RAG? → Latency, cost, and lack of your private data.
**52.** DuckDuckGo/Google fallback in RAG when nothing retrieved? → Yes — pattern is "RAG-first, web-fallback".
**53.** What is a knowledge graph vs vector DB? → KG stores entities+relations; vector stores meanings. Combine for entity-linked RAG.
**54.** What's Retrieval-Augmented Fine-Tuning (RAFT)? → Fine-tune the model to use retrieved context correctly, including ignoring distractors.
**55.** Explain the ChatGPT plugin architecture. → Custom actions defined by OpenAPI spec; ChatGPT calls the spec's operations as tools.
**56.** What's fp16 vs bf16 vs int8 quantization? → All three shrink model size. bf16 is preferred for training (wider range). int8 for inference speed.
**57.** How would you monitor prompt drift over time? → Track eval pass rate weekly; compute embedding-similarity of latest queries vs historical.
**58.** What is a router LLM vs main LLM? → Router is cheap (mini) — classifies query, picks pipeline. Main is expensive (big) — only for hard queries.
**59.** What's a canary deploy for LLM apps? → Start new version at 5% of traffic; watch eval metrics; ramp to 25%, 50%, 100% only if metrics stay green.
**60.** Give a career-development answer: "Where should I focus to become an AI engineer in 6 months?"
> Master this roadmap end-to-end. Build one production RAG or agent from scratch.
> Contribute to an OSS project (LangChain, LlamaIndex, MCP servers). Write about
> what you learned. Do 20 mock interviews with these questions.
