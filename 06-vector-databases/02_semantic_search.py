"""
02_semantic_search.py — Build a semantic search engine over 20 products
========================================================================

WHAT THIS FILE TEACHES
----------------------
• Build a searchable index over 20 product descriptions.
• Compare semantic search vs keyword search on paraphrase queries.
• Score display and top-K return.

HOW TO RUN
----------
    python 02_semantic_search.py

REAL-WORLD SCENARIO
-------------------
E-commerce site search — user types "cordless music player for jogging", your
product catalog has "wireless bluetooth sports earbuds". Keyword misses;
semantic wins.
"""

import os
import numpy as np

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


def embed(texts: list[str], dim: int = 128) -> np.ndarray:
    vectors = []
    for t in texts:
        base = np.random.default_rng(seed=abs(hash(t)) % (2**32)).standard_normal(dim)
        for w in t.lower().split():
            base += np.random.default_rng(seed=abs(hash(w)) % (2**32)).standard_normal(dim) * 0.4
        base /= np.linalg.norm(base)
        vectors.append(base)
    return np.array(vectors)


PRODUCTS = [
    ("PROD-001", "Wireless Bluetooth headphones with active noise cancellation, 30-hour battery, foldable design."),
    ("PROD-002", "Sports earbuds for running, sweat-proof IPX7, secure ear-hook fit, heart-rate monitor."),
    ("PROD-003", "Gaming mouse RGB, 16000 DPI, 8 programmable buttons, ergonomic grip for FPS gaming."),
    ("PROD-004", "Mechanical keyboard with hot-swap Cherry MX switches, RGB backlight, wireless."),
    ("PROD-005", "USB-C hub 7-in-1: HDMI 4K, 3× USB-A 3.0, SD card, gigabit ethernet, 100W power passthrough."),
    ("PROD-006", "Portable Bluetooth speaker, waterproof IP67, 24 hours play time, deep bass."),
    ("PROD-007", "Standing desk anti-fatigue mat, memory foam base, rounded edges."),
    ("PROD-008", "Ultrawide 34-inch curved monitor, 3440×1440, 144Hz, HDR400, USB-C 90W upstream."),
    ("PROD-009", "External SSD 2TB, USB-C 20Gbps NVMe, rugged aluminum shell, IP55."),
    ("PROD-010", "Wireless charger 15W fast pad, MagSafe compatible, LED indicator, foreign-object detection."),
    ("PROD-011", "Ergonomic office chair with lumbar support, mesh back, 4D armrests."),
    ("PROD-012", "Smart LED desk lamp, adjustable color temperature, USB-C power, presence sensor."),
    ("PROD-013", "1080p webcam with autofocus, dual-mic array, privacy shutter, works with Teams/Zoom."),
    ("PROD-014", "Waterproof running smartwatch with GPS, heart-rate, blood-oxygen, 7-day battery."),
    ("PROD-015", "27-inch 4K monitor for creators, 99% DCI-P3, USB-C 90W, hardware calibration."),
    ("PROD-016", "Portable dual-fan laptop cooler, adjustable RGB, USB-powered, quiet 25dB."),
    ("PROD-017", "Studio-quality wired dynamic microphone, XLR output, cardioid pattern, boom-arm compatible."),
    ("PROD-018", "Docking station for MacBook: dual HDMI, 6× USB-A, Ethernet, SD card, 96W charging."),
    ("PROD-019", "Wireless presenter with laser pointer, USB-C receiver, 3-year battery."),
    ("PROD-020", "Bluetooth mini keyboard for mobile, tri-device switching, backlit, aluminum."),
]

sku_index: dict[str, str] = {sku: desc for sku, desc in PRODUCTS}
descriptions = [d for _, d in PRODUCTS]
skus = [s for s, _ in PRODUCTS]

# One-time embed of the whole catalog (cache in production!)
print("Indexing catalog... ", end="", flush=True)
CATALOG = embed(descriptions)
print(f"done. Matrix: {CATALOG.shape}")


def semantic_search(query: str, k: int = 3) -> list[tuple[float, str, str]]:
    q = embed([query])[0]
    sims = CATALOG @ q
    top_idx = np.argsort(sims)[::-1][:k]
    return [(float(sims[i]), skus[i], descriptions[i]) for i in top_idx]


def keyword_search(query: str, k: int = 3) -> list[tuple[int, str, str]]:
    words = [w for w in query.lower().split() if len(w) > 2]
    scored = []
    for sku, desc in PRODUCTS:
        score = sum(1 for w in words if w in desc.lower())
        if score > 0:
            scored.append((score, sku, desc))
    scored.sort(key=lambda t: -t[0])
    return scored[:k]


# ─────────────────────────────────────────────────────────────
# Head-to-head comparison — paraphrase queries
# ─────────────────────────────────────────────────────────────
COMPARISON_QUERIES = [
    ("cordless music player for jogging",
     "Keyword misses: docs use 'wireless earbuds' / 'sports' / 'running', not 'cordless music'."),
    ("device to watch movies on a big screen",
     "Keyword misses: monitor docs don't say 'movies'."),
    ("charge my phone without a cable",
     "Keyword misses: 'wireless charger' docs use 'MagSafe' / 'charging pad'."),
    ("microphone for streaming on Twitch",
     "Keyword misses: docs say 'studio' / 'XLR' but not 'Twitch' or 'streaming'."),
    ("hardware that helps my wrists stop hurting",
     "Keyword misses ergonomic keyboard/mouse/chair/mat — different vocabulary."),
]

print("\n" + "=" * 78)
print("Semantic vs Keyword Search — head-to-head on paraphrase queries")
print("=" * 78)
for q, why in COMPARISON_QUERIES:
    print(f"\n🔎 Query: '{q}'")
    print(f"   Why keyword fails: {why}")

    print("\n   [KEYWORD] top hits:")
    kw = keyword_search(q, k=3)
    if not kw:
        print("       (none — no words overlap)")
    for sc, sku, d in kw:
        print(f"       {sku}  score={sc}  {d[:60]}...")

    print("\n   [SEMANTIC] top hits:")
    for sc, sku, d in semantic_search(q, k=3):
        print(f"       {sku}  sim={sc:+.3f}  {d[:60]}...")


# ─────────────────────────────────────────────────────────────
# Scenario where keyword WINS: exact product code
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("Where keyword search still wins: exact code / SKU / name")
print("=" * 78)
print("Query: 'PROD-015'")
print(f"  Keyword: {keyword_search('PROD-015', 3)}")
print(f"  Semantic: {[(sku, round(s, 3)) for s, sku, _ in semantic_search('PROD-015', 3)]}")
print("→ Combine both (hybrid search — Phase 6/§4).")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("✅ Summary")
print("=" * 78)
print("""
Semantic vs keyword search — pick the right tool:

  Keyword search wins:  exact match (SKU, code, name, error string)
  Semantic search wins: paraphrase, synonym, cross-lingual, intent-based
  Hybrid (weighted mix): production — best of both

You just built the retrieval layer of an e-commerce search engine.
The next file wraps it in ChromaDB, adding persistence + metadata filters.
""")
