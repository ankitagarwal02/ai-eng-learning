"""
03_episodic_memory.py — Session-level episodic memory
======================================================

Records each conversation/session as an Episode with topic, outcome, entities,
and timestamp. Later sessions can look up past episodes by topic, entity, or
recency to personalize.

SCENARIO
--------
Alice's support history over 6 months. On her 7th call, agent sees she has
had 2 prior refund requests → escalates directly instead of asking again.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class Episode:
    id: str
    user_id: str
    session_id: str
    timestamp: datetime
    topic: str
    summary: str
    outcome: str   # resolved | escalated | unresolved
    sentiment: str # positive | neutral | negative
    entities: dict[str, str] = field(default_factory=dict)   # e.g. order_id, product
    tags: list[str] = field(default_factory=list)
    turn_count: int = 0


class EpisodicMemory:
    def __init__(self):
        self.episodes: list[Episode] = []

    def record(self, ep: Episode):
        self.episodes.append(ep)

    def get_recent(self, n: int = 5, user_id: Optional[str] = None) -> list[Episode]:
        eps = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        return sorted(eps, key=lambda e: e.timestamp, reverse=True)[:n]

    def get_by_topic(self, topic: str, user_id: Optional[str] = None) -> list[Episode]:
        return [e for e in self.episodes
                if e.topic == topic and (user_id is None or e.user_id == user_id)]

    def get_in_range(self, days: int, user_id: Optional[str] = None) -> list[Episode]:
        cutoff = datetime.now() - timedelta(days=days)
        return [e for e in self.episodes
                if e.timestamp >= cutoff and (user_id is None or e.user_id == user_id)]

    def search_entities(self, key: str, value: str) -> list[Episode]:
        return [e for e in self.episodes if e.entities.get(key) == value]

    def format_for_context(self, user_id: str, max_episodes: int = 3) -> str:
        recent = self.get_recent(max_episodes, user_id=user_id)
        if not recent:
            return ""
        lines = [f"Recent history for {user_id}:"]
        for e in recent:
            lines.append(f"  - {e.timestamp.strftime('%Y-%m-%d')}: {e.topic} ({e.outcome}, {e.sentiment})")
            lines.append(f"      {e.summary}")
        return "\n".join(lines)

    def stats(self, user_id: Optional[str] = None) -> dict:
        eps = self.episodes if user_id is None else [e for e in self.episodes if e.user_id == user_id]
        return {
            "total": len(eps),
            "resolved": sum(1 for e in eps if e.outcome == "resolved"),
            "unresolved": sum(1 for e in eps if e.outcome == "unresolved"),
            "escalated": sum(1 for e in eps if e.outcome == "escalated"),
            "avg_turns": (sum(e.turn_count for e in eps) / len(eps)) if eps else 0,
            "negative_count": sum(1 for e in eps if e.sentiment == "negative"),
        }


# ─────────────────────────────────────────────────────────────
# Demo — 6 episodes over 6 months for user "alice"
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    now = datetime.now()
    mem = EpisodicMemory()

    for months_ago, topic, summary, outcome, sentiment, entities, turns in [
        (6, "onboarding", "Set up her workspace, invited teammates.", "resolved", "positive", {}, 5),
        (5, "billing",    "Question about invoice — resolved quickly.", "resolved", "neutral",  {"invoice_id": "INV-2001"}, 3),
        (4, "refund",     "Requested refund for double-charge in June.", "resolved", "negative", {"invoice_id": "INV-2050"}, 6),
        (2, "outage",     "Reported dashboard 502 error; issue was investigated.", "resolved", "neutral", {}, 4),
        (1, "refund",     "Second refund request — annual plan renewal.", "resolved", "negative", {"invoice_id": "INV-2200"}, 5),
        (0, "billing",    "New question about payment methods.",  "resolved", "neutral", {}, 2),
    ]:
        mem.record(Episode(
            id=f"E{100+months_ago}",
            user_id="alice",
            session_id=f"S{100+months_ago}",
            timestamp=now - timedelta(days=months_ago * 30),
            topic=topic, summary=summary, outcome=outcome, sentiment=sentiment,
            entities=entities, turn_count=turns,
        ))

    # Now Alice calls in for another refund:
    print("New session: Alice: 'I want to request a refund.'")
    print()

    prior_refunds = mem.get_by_topic("refund", user_id="alice")
    print(f"→ Prior refund episodes: {len(prior_refunds)}")
    for e in prior_refunds:
        print(f"    {e.timestamp.strftime('%Y-%m-%d')}  {e.summary}")

    print("\n" + "=" * 78)
    print("Context injected into agent's system prompt:")
    print("=" * 78)
    print(mem.format_for_context("alice", max_episodes=3))

    print("\n" + "=" * 78)
    print("Stats for alice:", mem.stats(user_id="alice"))

    print("""

✅ Result — WITHOUT episodic memory:
    Agent: "Sure, I can help with a refund. This is your first refund request."
   WITH episodic memory:
    Agent: "I see you've had 2 previous refund requests. Given the pattern,
            let me escalate this directly to a specialist and offer a
            resolution beyond a simple refund."

That's the difference between a chatbot and a memory-aware agent.
""")
