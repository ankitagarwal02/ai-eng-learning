"""
03_chromadb_local.py — ChromaDB in-memory and persistent
=========================================================

WHAT THIS FILE TEACHES
----------------------
• Install and initialize ChromaDB (in-memory + persistent).
• client.create_collection() / get_or_create_collection()
• collection.add(), .upsert(), .query(), .delete()
• Metadata filtering with where= clauses.
• Collection stats and inspection.

HOW TO RUN
----------
    pip install chromadb
    python 03_chromadb_local.py

REAL-WORLD SCENARIO
-------------------
Index 30 FAQ answers. Query them semantically. Filter by product area
("billing" vs "features"). Show the full production shape: init, index, query,
update, delete, stats.
"""

import os
import shutil
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    _has_chroma = True
except ImportError:
    _has_chroma = False
    print("Install chromadb: pip install chromadb")

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


if not _has_chroma:
    print("\nCannot demo without chromadb installed.")
    raise SystemExit(0)


# ─────────────────────────────────────────────────────────────
# SECTION 1: In-memory client (throw-away)
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: In-memory ChromaDB")
print("=" * 70)

# Chroma uses a built-in embedding function unless you pass your own.
# For MOCK_MODE / speed, we use their default (all-MiniLM-L6-v2 sentence-transformer).
client = chromadb.EphemeralClient()   # in-memory, dies when process exits

collection = client.get_or_create_collection(
    name="faq",
    metadata={"description": "Product FAQ answers"},
)
print(f"Collection: {collection.name}  count={collection.count()}")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Add 30 FAQ entries
# ─────────────────────────────────────────────────────────────
FAQ = [
    # (id, question, answer, area)
    ("F01", "How do I export my data?",         "Export from Settings → Data → Export CSV/JSON.",         "howto"),
    ("F02", "How do I change my email?",        "Profile → Account → change email; confirm via link.",    "howto"),
    ("F03", "How do I invite teammates?",       "Workspace settings → Members → Invite by email.",        "howto"),
    ("F04", "How to reset my password?",        "Login page → 'Forgot password' → check email.",          "howto"),
    ("F05", "How do I enable dark mode?",       "User settings → Appearance → Dark mode toggle.",         "howto"),
    ("F06", "How do webhooks work?",             "Configure a webhook URL in Integrations; we POST JSON.", "howto"),
    ("F07", "What's the refund policy?",         "Refunds available within 30 days of purchase.",          "billing"),
    ("F08", "How do I get an invoice?",          "Billing → Invoices tab → download PDF.",                 "billing"),
    ("F09", "How do I change payment method?",   "Billing → Payment methods → add/remove card.",           "billing"),
    ("F10", "Why was my card charged twice?",    "Duplicate charges usually clear in 3-5 days; contact if not.", "billing"),
    ("F11", "How to cancel my subscription?",    "Billing → Subscription → Cancel; retains access until period end.", "billing"),
    ("F12", "Can I downgrade my plan?",          "Yes — Billing → Change plan; downgrade takes effect next cycle.",   "billing"),
    ("F13", "Do you support SSO?",               "Yes — SAML SSO on Business plan, Okta/OneLogin/Azure AD.", "features"),
    ("F14", "Do you have a mobile app?",         "iOS and Android apps available on app stores.",         "features"),
    ("F15", "Do you support Slack integration?", "Yes — install our Slack app from the Integrations page.", "features"),
    ("F16", "Is there an API?",                  "Full REST API with keys under Developer settings.",     "features"),
    ("F17", "Do you support GraphQL?",           "GraphQL endpoint /api/graphql; requires API key.",      "features"),
    ("F18", "What data retention do you offer?", "Standard: 90 days. Enterprise: configurable up to 7 years.", "features"),
    ("F19", "How is my data encrypted?",         "TLS in transit; AES-256 at rest.",                       "security"),
    ("F20", "Are you SOC 2 compliant?",          "Yes — SOC 2 Type II. Report available under NDA.",       "security"),
    ("F21", "Is data isolated per customer?",    "Yes — logical isolation via tenant IDs; Enterprise: physical isolation option.", "security"),
    ("F22", "Do you support 2FA?",               "TOTP-based 2FA available in Security settings.",         "security"),
    ("F23", "How do I report a security issue?", "Email security@ or submit via our HackerOne program.",  "security"),
    ("F24", "The site is down — is there an outage?", "Check our status page at status.example.com.",   "outage"),
    ("F25", "I'm getting 502 errors.",           "Usually a transient issue; check status page. If persistent, contact support.", "outage"),
    ("F26", "Login is failing.",                  "Clear browser cookies for our domain or try incognito. If continues, contact support.", "outage"),
    ("F27", "Reports aren't loading.",           "Usually resolves in <5 min. If not, refresh and check status page.", "outage"),
    ("F28", "How do I upgrade my plan?",          "Billing → Change plan → select new tier → confirm.",    "billing"),
    ("F29", "Can I try before paying?",           "14-day free trial, no credit card required.",           "billing"),
    ("F30", "Do you have a partner program?",    "Yes — partners@example.com or apply at /partners.",     "features"),
]

ids       = [f[0] for f in FAQ]
docs      = [f"Q: {f[1]}\nA: {f[2]}" for f in FAQ]
metadatas = [{"area": f[3], "faq_id": f[0]} for f in FAQ]

collection.add(ids=ids, documents=docs, metadatas=metadatas)
print(f"After add: count={collection.count()}")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Basic query
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: Basic query")
print("=" * 70)

def pretty(results, header):
    print(f"\n{header}")
    if not results["ids"] or not results["ids"][0]:
        print("  (no matches)")
        return
    for i, (id_, doc, dist, meta) in enumerate(zip(
        results["ids"][0], results["documents"][0],
        results["distances"][0], results["metadatas"][0]
    )):
        print(f"  {i+1}. {id_} [{meta['area']}] dist={dist:.3f}")
        print(f"     {doc.splitlines()[0][:60]}...")

pretty(
    collection.query(query_texts=["I want to leave my subscription"], n_results=3),
    "Query: 'I want to leave my subscription' (no filter)"
)


# ─────────────────────────────────────────────────────────────
# SECTION 4: Metadata filtering
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: Metadata filter — restrict to 'security' area")
print("=" * 70)

pretty(
    collection.query(
        query_texts=["how do you protect my information"],
        n_results=3,
        where={"area": "security"},
    ),
    "Query: 'how do you protect my information' WHERE area=security"
)

pretty(
    collection.query(
        query_texts=["compliance"],
        n_results=3,
        where={"area": {"$in": ["security", "features"]}},
    ),
    "Query: 'compliance' WHERE area IN (security, features)"
)


# ─────────────────────────────────────────────────────────────
# SECTION 5: Upsert (update-or-insert)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: Upsert — safely update existing")
print("=" * 70)

# Update F04's answer to include SMS reset:
collection.upsert(
    ids=["F04"],
    documents=["Q: How to reset my password?\nA: Login page → 'Forgot password' → receive email OR SMS. Link expires in 30 min."],
    metadatas=[{"area": "howto", "faq_id": "F04", "updated": True}],
)
print(f"After upsert: count={collection.count()}  (unchanged — upsert of existing id)")

pretty(
    collection.query(query_texts=["forgot my password"], n_results=1),
    "Query after upsert: 'forgot my password'"
)


# ─────────────────────────────────────────────────────────────
# SECTION 6: Delete
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: Delete")
print("=" * 70)

collection.delete(ids=["F30"])
print(f"After delete F30: count={collection.count()}")

# Bulk delete by metadata filter:
collection.delete(where={"area": "outage"})
print(f"After bulk delete area=outage: count={collection.count()}")


# ─────────────────────────────────────────────────────────────
# SECTION 7: Persistent client
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7: Persistent client (survives process restart)")
print("=" * 70)

persist_dir = Path("./chroma_persistent")
if persist_dir.exists():
    shutil.rmtree(persist_dir)

pclient = chromadb.PersistentClient(path=str(persist_dir))
pcol = pclient.get_or_create_collection(name="persisted_faq")
pcol.add(ids=["P01"], documents=["persisted forever"], metadatas=[{"kind": "test"}])
print(f"Persistent count: {pcol.count()}")
print(f"Files on disk: {list(persist_dir.rglob('*'))[:5]}")


# ─────────────────────────────────────────────────────────────
# SECTION 8: Inspect / stats
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 8: Collection stats")
print("=" * 70)
print(f"Collections: {[c.name for c in client.list_collections()]}")
print(f"'faq' count: {collection.count()}")
sample = collection.peek(limit=2)
print(f"Sample IDs from peek(): {sample['ids']}")


# ─────────────────────────────────────────────────────────────
# Cleanup persistent files
# ─────────────────────────────────────────────────────────────
if persist_dir.exists():
    shutil.rmtree(persist_dir)


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
The ChromaDB API you'll use every day:

  create/get collection      → .get_or_create_collection(name=...)
  add docs                    → .add(ids=, documents=, metadatas=)
  update in place             → .upsert(ids=, ...)
  semantic query              → .query(query_texts=, n_results=, where=)
  metadata filter operators   → $eq, $ne, $in, $nin, $lt, $gt, $and, $or
  delete                      → .delete(ids=) or .delete(where=)
  stats                       → .count(), .peek(), .list_collections()

Same API concepts work on Qdrant, Weaviate, Pinecone, pgvector — different
client names, same operations.

Next: 04_vector_db_patterns.py — chunking, hybrid search, re-ranking.
""")
