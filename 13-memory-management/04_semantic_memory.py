"""
04_semantic_memory.py — User profile facts with versioning
============================================================

Extracts stable facts about the user across turns. Later turns automatically
include the profile in system context so the agent can personalize responses.

SCENARIO
--------
Turn 1: User: "I'm Alice, a software engineer at FinBank."
Turn 3: User: "I prefer replies in bullet points."
Turn 5: User: "Actually, I moved to Stripe last month."   ← conflict resolution!

Later turns receive a system message:
   "User: Alice; role: software engineer; company: Stripe; format: bullets."
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class Fact:
    key: str
    value: str
    confidence: float = 1.0
    source: str = "user"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1


class UserProfile:
    def __init__(self):
        self.facts: dict[str, Fact] = {}
        self.changelog: list[dict] = []

    def set(self, key: str, value: str, source: str = "user", confidence: float = 1.0):
        if key in self.facts:
            prev = self.facts[key]
            if prev.value == value:
                return
            self.changelog.append({
                "at": datetime.now().isoformat(),
                "key": key,
                "from": prev.value,
                "to": value,
                "source": source,
            })
            new = Fact(key=key, value=value, confidence=confidence, source=source,
                       created_at=prev.created_at,
                       updated_at=datetime.now(),
                       version=prev.version + 1)
        else:
            new = Fact(key=key, value=value, confidence=confidence, source=source)
        self.facts[key] = new

    def get(self, key: str, default=None):
        f = self.facts.get(key)
        return f.value if f else default

    def get_all(self, min_confidence: float = 0.0) -> dict[str, str]:
        return {k: f.value for k, f in self.facts.items() if f.confidence >= min_confidence}

    def to_context_string(self) -> str:
        if not self.facts:
            return ""
        parts = [f"{k}={v}" for k, v in self.get_all(min_confidence=0.7).items()]
        return "User profile: " + "; ".join(parts)


# ─────────────────────────────────────────────────────────────
# Simple pattern-based fact extractor (mock LLM)
# ─────────────────────────────────────────────────────────────
import re

FACT_PATTERNS = [
    (re.compile(r"i'?m\s+(\w+)[,\.]", re.IGNORECASE),         "name"),
    (re.compile(r"my\s+name\s+is\s+(\w+)", re.IGNORECASE),   "name"),
    (re.compile(r"software\s+engineer|data\s+scientist|manager|analyst", re.IGNORECASE), "role"),
    (re.compile(r"at\s+([A-Z][A-Za-z]+)", ),                  "company"),
    (re.compile(r"moved\s+to\s+([A-Z][A-Za-z]+)", ),          "company"),
    (re.compile(r"bullet\s+point|bulleted",   re.IGNORECASE), "format"),
    (re.compile(r"risk[- ]averse",            re.IGNORECASE), "risk_tolerance"),
]

def extract_facts(text: str) -> dict[str, str]:
    out = {}
    for pattern, key in FACT_PATTERNS:
        m = pattern.search(text)
        if m:
            val = m.group(1) if m.groups() else m.group(0)
            if key == "role":
                val = m.group(0).lower()
            if key == "format":
                val = "bullets"
            out[key] = val
    return out


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    profile = UserProfile()

    conversation = [
        "Hi, I'm Alice, a software engineer at FinBank.",
        "Can you help me set up MLOps for a new project?",
        "I prefer replies in bullet points.",
        "I'm risk-averse when it comes to production changes.",
        "Actually, I moved to Stripe last month.",
    ]

    for i, turn in enumerate(conversation, 1):
        facts = extract_facts(turn)
        for k, v in facts.items():
            profile.set(k, v)
        print(f"\nTurn {i}: {turn}")
        if facts:
            print(f"  Extracted: {facts}")
        print(f"  Context now: {profile.to_context_string()}")

    print("\n─── Change log ───")
    for entry in profile.changelog:
        print(f"  {entry['key']}: {entry['from']} → {entry['to']}  ({entry['at'][:19]})")

    print("""
✅ Semantic memory pattern:

  • Facts have version numbers — conflicts overwrite the older value
  • Changelog preserves the history (crucial for compliance / debug)
  • .to_context_string() injects the profile into every LLM call

Result: agent adapts its style + content to each user, across sessions.
""")
