"""
01_embeddings_deep_dive.py — Embeddings, cosine, top-K from scratch
====================================================================

WHAT THIS FILE TEACHES
----------------------
• OpenAI text-embedding-3-small in real mode; deterministic mock in MOCK_MODE.
• Cosine similarity from scratch and via numpy dot-product.
• Normalizing vectors (why unit-norm makes cosine == dot product).
• Embedding a batch of documents efficiently.
• Finding top-K most similar with np.argsort.

HOW TO RUN
----------
    pip install numpy
    python 01_embeddings_deep_dive.py

REAL-WORLD SCENARIO
-------------------
Index 20 support-ticket summaries. Given a new incoming ticket, find the 3 most
similar past tickets — this is the "similar cases" feature of every support tool.
"""

import os
import numpy as np

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# The embedding function
# ─────────────────────────────────────────────────────────────
def embed(texts: list[str], dim: int = 128) -> np.ndarray:
    """Embed a list of texts. Real mode uses OpenAI; MOCK_MODE uses a
    deterministic pseudo-embedding that respects word overlap."""
    if MOCK_MODE:
        vectors = []
        for t in texts:
            base = np.random.default_rng(seed=abs(hash(t)) % (2**32)).standard_normal(dim)
            # Word-based drift — shared words → similar vectors
            for w in t.lower().split():
                base += np.random.default_rng(seed=abs(hash(w)) % (2**32)).standard_normal(dim) * 0.4
            base /= np.linalg.norm(base)
            vectors.append(base)
        return np.array(vectors)

    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])


# ─────────────────────────────────────────────────────────────
# SECTION 1: What an embedding looks like
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: What an embedding is")
print("=" * 70)

v = embed(["The quick brown fox jumps over the lazy dog"])
print(f"Shape: {v.shape}")
print(f"First 8 dims: {v[0][:8].round(4)}")
print(f"Norm (should ≈ 1.0):  {np.linalg.norm(v[0]):.6f}")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Cosine similarity, from scratch
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: Cosine similarity")
print("=" * 70)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

pairs = [
    ("bluetooth headphones",           "wireless earbuds"),         # near-synonym
    ("machine learning model",          "neural network training"),  # related
    ("banana",                          "quantum physics"),          # unrelated
    ("The dog barks loudly",            "The dog barks loudly"),    # identical
]

for a, b in pairs:
    va, vb = embed([a, b])
    print(f"  cosine('{a[:30]}', '{b[:30]}') = {cosine(va, vb):+.3f}")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Why unit-norm makes cosine == dot-product
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: Unit norm & dot product")
print("=" * 70)
print("""
For unit vectors (‖v‖ = 1):
    cosine(a, b) = (a · b) / (‖a‖ · ‖b‖) = a · b

Since our embed() normalizes to unit length, we can use plain matrix-vector
multiplication to compute *all* similarities against a query in one operation.
""")

# Demonstrate: 20 doc vectors × query vector → 20 similarity scores in one dot-product
docs = ["doc-" + str(i) for i in range(20)]
doc_matrix = embed(docs)                # shape: (20, 128)
query_vec = embed(["doc-3"])[0]         # shape: (128,)

sims = doc_matrix @ query_vec           # shape: (20,)  — ONE matmul
print(f"Similarities against 'doc-3':")
for name, sim in zip(docs, sims):
    bar = "█" * int(max(sim, 0) * 40)
    print(f"  {name:<7} {sim:+.3f} {bar}")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Top-K with argsort
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: Top-K most similar")
print("=" * 70)

def topk(query: str, docs: list[str], k: int = 3) -> list[tuple[float, str]]:
    q = embed([query])[0]
    corpus = embed(docs)
    sims = corpus @ q                            # cosine because unit-normed
    top_idx = np.argsort(sims)[::-1][:k]         # descending
    return [(float(sims[i]), docs[i]) for i in top_idx]


# ─────────────────────────────────────────────────────────────
# SECTION 5: Real-world scenario — similar past tickets
# ─────────────────────────────────────────────────────────────
past_tickets = [
    "Customer's card was double-charged on the monthly subscription.",
    "User can't log in — 500 error on the login page.",
    "How do I export my project data as CSV format?",
    "Requesting refund for the annual plan I just cancelled.",
    "Dashboard reports are returning 502 gateway errors.",
    "Where can I find my invoice for last month?",
    "Site is completely down — none of my links load.",
    "Feature request: dark mode for the mobile app.",
    "Can you add SSO integration with Okta?",
    "My subscription renewed but I meant to cancel it.",
    "I love this product, it's saved my team hours per week.",
    "Getting 401 unauthorized on API endpoints suddenly.",
    "How do I invite additional teammates to my workspace?",
    "Please downgrade my plan from Pro to Starter.",
    "The docs seem out of date — is there a v2 quickstart?",
    "Bulk delete of projects is missing from the UI.",
    "Payment failed with 'card declined' but my card is fine.",
    "How do webhooks work? Where can I find examples?",
    "Absolutely amazing tool, recommending to everyone.",
    "Random 503 errors on the API since 3pm today.",
]

new_ticket = "My credit card was charged twice for this month"
print(f"\nNew ticket: {new_ticket!r}\n")
print("Top 3 similar past tickets:")
for score, ticket in topk(new_ticket, past_tickets, k=3):
    print(f"  sim {score:+.3f}  →  {ticket}")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Visualization intuition (no actual plotting)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: Cluster intuition (PCA teaser)")
print("=" * 70)
print("""
If you ran PCA to reduce 128 → 2 dims and plotted our 20 tickets, you'd see
roughly 4 clusters emerge:

    ↑
    │  refund/billing ●●●
    │        outage ●●●
y   │              howto ●●●
    │                       feature ●
    │                                    praise ●●
    └────────────────────────────────────────────→  x

That's the topological structure the vector DB is indexing — not "which words
overlap" but "which meanings are close in space".

In production, use `sklearn.decomposition.PCA` or `umap-learn` to visualize
your corpus. Great for spotting duplicate or misclassified docs.
""")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
The three primitives your entire RAG/memory stack rests on:

  • embed(text) → unit vector
  • cosine(a, b) == a · b  when both are unit-length
  • topk = np.argsort(matrix @ query)[::-1][:k]

Everything from here (semantic search, ChromaDB, RAG, agent memory) is a
production wrapper around these same three operations.

Next: 02_semantic_search.py — build a real search engine over 20 products.
""")
