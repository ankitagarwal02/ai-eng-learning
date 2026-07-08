# Phase 3 — LLM Fundamentals

> **Real-life analogy:** An LLM is a world-class consultant who read the entire internet before
> falling asleep and remembers most of it. Your prompt is your briefing document. Better
> briefing → better advice. Vague briefing → vague advice.

---

## How an LLM actually works (30-second version)

```
   ┌─────────────────────────────────────────────────────────────────┐
   │  Input tokens ──▶ Transformer stack (attention + MLP layers) ──▶│
   │                                                                 │
   │  For each next token:                                           │
   │    1. Compute probabilities over the ~100k vocabulary           │
   │    2. Sample (temperature/top-p controls how)                   │
   │    3. Append token, feed back in, repeat                        │
   │                                                                 │
   │  Stops when: <|endoftext|> emitted OR max_tokens hit            │
   └─────────────────────────────────────────────────────────────────┘
```

**Key implication:** every response is generated one token at a time, conditioned on all
previous tokens (your prompt + what it has said so far). This is why:
- Longer prompts cost more.
- Instructions early in the prompt influence the whole response.
- "Chain-of-thought" works — the model uses its own reasoning tokens as context.

---

## The Prompting Hierarchy (increasing power)

```
                ┌───────────────────────────────────────┐
       Level 5  │  ReAct (Thought → Action → Observe)   │  ← Agents
                ├───────────────────────────────────────┤
       Level 4  │  Chain-of-Thought / Self-consistency  │  ← Complex reasoning
                ├───────────────────────────────────────┤
       Level 3  │  Few-shot (3-5 examples in prompt)    │  ← Format learning
                ├───────────────────────────────────────┤
       Level 2  │  One-shot + role + constraints        │  ← Consistent output
                ├───────────────────────────────────────┤
       Level 1  │  Zero-shot (just ask directly)        │  ← Simple tasks
                └───────────────────────────────────────┘
```

Start at the bottom. Only escalate when quality demands it — each level costs more tokens.

---

## Temperature & Top-p (sampling controls)

```
temperature = 0.0     Deterministic — same input → same output.  Use for:
                      classification, extraction, code generation.

temperature = 0.7     Creative but coherent.  Use for:
                      chat, writing assistance, brainstorming.

temperature = 1.5     Wild — often incoherent, hallucinates. Rarely useful.

top_p = 1.0           Consider all tokens.
top_p = 0.1           Consider only the top 10% probability mass.
                      (An alternative to temperature — usually don't tune both.)
```

**Rule of thumb**: for STRUCTURED output, set `temperature=0`. Set higher only when
"creativity" is the explicit goal.

---

## System vs User Prompts

```
role:  "system"       ← Set the assistant's identity, rules, format.
                        Persists for the whole conversation.
                        Users cannot see it (usually).
                        LOAD-BEARING — put your most important instructions here.

role:  "user"         ← The actual input to respond to.

role:  "assistant"    ← Previous replies from the model (for multi-turn).

role:  "tool"         ← Tool call results (Phase 8+).
```

---

## The 8 Prompting Techniques (covered in `01_prompting_techniques.py`)

| # | Technique | One-line pitch | Cost |
|---|-----------|----------------|------|
| 1 | Zero-shot | Just ask | ⚡ cheap |
| 2 | One-shot | Show one example, then ask | ⚡ cheap |
| 3 | Few-shot | Show 3-5 examples | 🟡 medium |
| 4 | Chain-of-Thought (CoT) | Ask model to "think step by step" | 🟡 medium |
| 5 | System prompt engineering | Load persona + constraints in system role | ⚡ cheap |
| 6 | Role prompting | "You are an expert X..." | ⚡ cheap |
| 7 | Self-consistency | Ask 3× at temp>0, majority-vote | 🔴 3× cost |
| 8 | ReAct | Interleave Thought/Action/Observation | 🔴 many calls |

---

## Structured Output (covered in `02_structured_output.py`)

The 4 approaches, worst-to-best reliability:

```
1. "Please respond in JSON..."            ← fragile, 5-20% parse failures in prod
2. response_format={"type": "json_object"}  ← guaranteed valid JSON syntax
3. response_format={"type": "json_schema"}  ← guaranteed matches your schema
4. Tool-calling with a Pydantic schema      ← same as #3 + fits agent pattern
```

For any structured extraction, jump straight to #3 or #4.

---

## Concept Table

| Term | Plain-English |
|------|---------------|
| **Zero/one/few-shot** | 0, 1, or several examples included in the prompt |
| **Chain-of-Thought (CoT)** | Model reasons step-by-step before answering |
| **Self-consistency** | Sample same prompt multiple times, take majority answer |
| **ReAct** | Thought → Action (tool) → Observation → repeat |
| **System prompt** | Instructions that scope the whole conversation |
| **Temperature** | Randomness knob (0 = deterministic, 1 = creative) |
| **Top-p** | Nucleus sampling — considers only top-p probability mass |
| **Structured output** | Guaranteed JSON matching a schema |
| **Hallucination** | Model confidently invents wrong facts |
| **Context window** | Max tokens the model can attend to (128k for 4o, 200k for Sonnet) |

---

## Use Case Matrix

| Scenario | Right technique |
|----------|-----------------|
| Classify customer intent | Zero-shot + Structured output |
| Extract fields from an invoice | Structured output with Pydantic schema |
| Solve a math word problem | Chain-of-Thought |
| Rewrite in a specific brand voice | Few-shot (3 examples in that voice) |
| Multi-step reasoning task | Self-consistency (3x, majority vote) |
| Look up + answer with tools | ReAct (Phase 8) |
| Multi-turn assistant | Rich system prompt + user/assistant history |

---

## Folder structure

```
03-llm-fundamentals/
├── README.md                    ← You are here
├── 01_prompting_techniques.py   ← All 8 techniques, MOCK_MODE safe
├── 02_structured_output.py      ← 4 approaches, ExtractedMedicalRecord example
└── requirements.txt
```
