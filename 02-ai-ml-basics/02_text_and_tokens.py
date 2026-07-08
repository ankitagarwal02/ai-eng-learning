"""
02_text_and_tokens.py — Tokens, embeddings, cosine similarity, semantic search
===============================================================================

WHAT THIS FILE TEACHES
----------------------
• How text is tokenized (BPE intuition).
• `tiktoken` for exact token counting (falls back to word-based estimate).
• Why token count drives cost AND latency.
• Embeddings — what they are, why similar text is "close" in vector space.
• Cosine similarity implemented from scratch with numpy.
• Building a mini semantic search over 10 documents.

HOW TO RUN
----------
    pip install numpy tiktoken   # tiktoken optional but recommended
    python 02_text_and_tokens.py

REAL-WORLD SCENARIO
-------------------
You have 10,000 product reviews. Show how to:
  1. Count total tokens to estimate GPT-4o-mini cost
  2. Embed reviews and find the 3 most similar to a search query
  3. Explain why semantic search beats keyword search on paraphrased queries
"""

import os
import numpy as np

try:
    import tiktoken
    _has_tiktoken = True
except ImportError:
    _has_tiktoken = False
    print("(tiktoken not installed — falling back to word-count estimate.)")

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# SECTION 1: What is a token?
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: What is a token?")
print("=" * 70)
print("""
GPT tokenizers split text into subword pieces using BPE
(Byte-Pair Encoding). Common words are 1 token; rare words split into pieces.

Examples (approximate for cl100k_base tokenizer):
   "The"        → 1 token
   "quick"      → 1 token
   "tokenization" → 2 tokens  ["token", "ization"]
   "AGI"        → 1 token
   "obfuscation" → 3 tokens   ["ob", "fuscation"]   (long, uncommon)

Rules of thumb (English):
   ~ 4 chars     ≈ 1 token
   ~ 0.75 words  ≈ 1 token
   1024 tokens   ≈ 3 paragraphs of text
""")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Exact token counting with tiktoken
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 2: Token counting")
print("=" * 70)

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Exact token count via tiktoken, else word-based estimate."""
    if _has_tiktoken:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return int(len(text.split()) / 0.75)   # rough fallback

samples = [
    "Hi.",
    "The quick brown fox jumps over the lazy dog.",
    "Tokenization is fundamental to understanding LLM costs.",
    "Please summarize the attached quarterly earnings report and highlight risks.",
]
for s in samples:
    print(f"  {count_tokens(s):>4} tokens  '{s}'")

# Multi-turn conversation counting — real chat costs:
messages = [
    {"role": "system", "content": "You are a helpful support agent."},
    {"role": "user",   "content": "My order didn't arrive."},
    {"role": "assistant", "content": "I'm sorry to hear that. Can you share your order number?"},
    {"role": "user",   "content": "ORD-12345, placed on July 3rd."},
]
total = sum(count_tokens(m["content"]) for m in messages) + len(messages) * 4  # ~4 tokens per message overhead
print(f"\nFull conversation: {total} tokens")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Cost estimation
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: Cost estimation for 10,000 product reviews")
print("=" * 70)

# Simulate 10,000 reviews (avg 80 tokens each = 800k tokens):
n_reviews = 10_000
avg_input_tokens = 80
avg_output_tokens = 40   # short classification answer

# Prices per 1M tokens (2026 approximation):
PRICES = {
    "gpt-4o-mini":       {"input": 0.15, "output": 0.60},
    "gpt-4o":            {"input": 2.50, "output": 10.00},
    "claude-3-5-haiku":  {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
}

print(f"Volume: {n_reviews:,} reviews × ({avg_input_tokens} in + {avg_output_tokens} out) tokens\n")
print(f"{'Model':<22} {'Input $':>10} {'Output $':>10} {'Total $':>10}")
for model, p in PRICES.items():
    in_cost = n_reviews * avg_input_tokens * p["input"] / 1_000_000
    out_cost = n_reviews * avg_output_tokens * p["output"] / 1_000_000
    total_c = in_cost + out_cost
    print(f"  {model:<20} {in_cost:>10.2f} {out_cost:>10.2f} {total_c:>10.2f}")

print("""
Takeaway: gpt-4o costs ~16× gpt-4o-mini. Use mini until proven insufficient.
""")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Embeddings — what they ARE
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 4: Embeddings")
print("=" * 70)
print("""
An embedding is a fixed-length vector of floats that captures the *meaning*
of a piece of text. Similar meaning → similar direction in vector space.

Text A: 'wireless bluetooth earbuds'
Text B: 'cordless in-ear headphones'
Text C: 'car engine oil'

emb(A) · emb(B) → high (both talk about listening devices)
emb(A) · emb(C) → low  (unrelated topics)

Real embedding models: OpenAI text-embedding-3-small (1536 dims),
sentence-transformers/all-MiniLM-L6-v2 (384 dims), Voyage voyage-3 (1024 dims).

In MOCK_MODE we generate DETERMINISTIC pseudo-embeddings so the *shape* of the
computation is correct, without needing an API key.
""")


def mock_embed(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic 'embedding' from a hash — good enough to show the math.
    Real embeddings capture semantics; ours captures character n-grams.
    """
    rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim)
    # Layer in some pseudo-semantic clustering: shared words → shared shift.
    for word in text.lower().split():
        wrng = np.random.default_rng(seed=abs(hash(word)) % (2**32))
        v += wrng.standard_normal(dim) * 0.3
    # Normalize to unit length so cosine ≡ dot product:
    return v / np.linalg.norm(v)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity — angle between two vectors, 1..-1."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─────────────────────────────────────────────────────────────
# SECTION 5: Semantic search over 10 product descriptions
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 5: Semantic search on 10 product descriptions")
print("=" * 70)

products = [
    "Wireless Bluetooth headphones with noise cancellation, 30-hour battery, foldable design.",
    "Sports earbuds for running, sweat-proof, secure fit, integrated heart-rate monitor.",
    "Gaming mouse RGB, 16000 DPI, 8 programmable buttons, ergonomic grip.",
    "Mechanical keyboard with hot-swap switches, RGB backlight, wireless connectivity.",
    "USB-C hub 7-in-1: HDMI, 3× USB-A, SD card, ethernet, 100W passthrough.",
    "Portable Bluetooth speaker, waterproof IP67, 24 hours play time, deep bass.",
    "Standing desk mat with anti-fatigue foam, non-slip base, rounded edges.",
    "Ultrawide 34-inch curved monitor, 3440×1440, 144Hz, HDR400, USB-C 90W.",
    "External SSD 2TB, USB-C 20Gbps, rugged aluminum case, IP55 water/dust resistant.",
    "Wireless charger 15W fast charging pad, MagSafe compatible, LED indicator.",
]

# Embed all products (one-time cost — cache in production):
product_vectors = np.array([mock_embed(p) for p in products])

def search(query: str, k: int = 3) -> list[tuple[float, str]]:
    q = mock_embed(query)
    sims = product_vectors @ q     # matrix-vector dot product = many cosines at once
    top_idx = np.argsort(sims)[::-1][:k]
    return [(float(sims[i]), products[i]) for i in top_idx]


queries = [
    "wireless headphones for gym",
    "keyboard for programming",
    "monitor for productivity",
]
for q in queries:
    print(f"\n🔎 Query: '{q}'")
    for score, prod in search(q, k=3):
        print(f"  sim {score:+.3f}  {prod[:70]}...")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Semantic vs keyword — where each wins
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: Semantic search beats keyword search")
print("=" * 70)

def keyword_search(query: str, docs: list[str], k: int = 3) -> list[str]:
    words = set(query.lower().split())
    scored = [(sum(1 for w in words if w in d.lower()), d) for d in docs]
    scored.sort(key=lambda t: -t[0])
    return [d for score, d in scored if score > 0][:k]

comparison_queries = [
    ("cordless music player for jogging",
     "Keyword misses because docs use 'earbuds' / 'running' / 'Bluetooth' — not 'cordless music'."),
    ("device to watch movies on a big screen",
     "Keyword misses the ultrawide monitor because docs don't say 'movies'."),
    ("charge my phone without a cable",
     "Keyword misses the wireless charger because docs say 'MagSafe' / 'charging pad'."),
]

for q, why in comparison_queries:
    print(f"\n🔎 Query: '{q}'")
    print(f"    Why keyword fails: {why}")
    print("    Keyword top hits:", keyword_search(q, products, 3) or ["(none)"])
    print("    Semantic top hits:")
    for score, prod in search(q, k=3):
        print(f"       sim {score:+.3f}  {prod[:60]}...")


# ─────────────────────────────────────────────────────────────
# SECTION 7: Deduplication with cosine
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7: Deduplication with embeddings")
print("=" * 70)

near_dupes = [
    "Refund my last payment please",
    "Please issue a refund for the payment I made",   # paraphrase of the above
    "How do I export my project data?",
    "Show me how to export project data",             # paraphrase
    "The system is completely down since 3pm.",       # unrelated
]

vecs = np.array([mock_embed(t) for t in near_dupes])
sims = vecs @ vecs.T
print("Pairwise cosine similarity matrix:")
for i, row in enumerate(sims):
    line = "  ".join(f"{v:+.2f}" for v in row)
    print(f"  {i}: [{line}]  {near_dupes[i][:35]}...")

# Naive dedup: for each doc, keep only if not already >0.85 similar to a kept doc.
THRESHOLD = 0.85
kept: list[int] = []
for i in range(len(near_dupes)):
    if not any(sims[i, j] > THRESHOLD for j in kept):
        kept.append(i)

print(f"\nAfter dedup (threshold {THRESHOLD}), kept {len(kept)} of {len(near_dupes)}:")
for i in kept:
    print(f"  • {near_dupes[i]}")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
Everything you need for the LLM/RAG stack:

  • Tokens: use tiktoken for exact counts; ~4 chars ≈ 1 token as a shortcut
  • Cost = tokens × price/1M — model choice can 10× your bill
  • Embeddings turn text into direction-in-space
  • Cosine similarity = np.dot(a,b) when vectors are unit-normalized
  • Semantic search wins on paraphrase & synonym queries
  • Deduplication uses the same cosine trick

Phase 3 (LLM Fundamentals) uses these primitives directly:
  - Prompting techniques all send tokens
  - Structured output parses LLM tokens back into structure
  - Retrieval (Phase 6+) uses the exact embedding math you just built
""")
