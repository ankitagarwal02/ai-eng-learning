"""
03_classes_and_oop.py — OOP for AI Engineers: inheritance, dunders, dataclasses, ABCs
======================================================================================

WHAT THIS FILE TEACHES
----------------------
• Class definition, __init__, instance methods, classmethods, staticmethods.
• Inheritance and method overriding (polymorphism).
• Dunder methods: __repr__, __str__, __len__, __eq__.
• Properties (@property, @setter) — computed / validated attributes.
• Abstract base classes (abc.ABC) — enforcing contracts.
• @dataclass — the modern way to define value objects.

HOW TO RUN
----------
    python 03_classes_and_oop.py

REAL-WORLD SCENARIO
-------------------
Build an AIAssistant base class with two subclasses:
    - CustomerSupportBot (empathetic, ticket-aware)
    - CodeReviewBot     (technical, terse, uses code style rules)

Show polymorphism: same .respond(msg) call, different behavior per subclass.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Dataclasses — value objects the modern way
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Dataclasses")
print("=" * 70)

@dataclass
class Message:
    """A conversation turn. @dataclass auto-generates __init__, __repr__, __eq__."""
    role: str           # "system" | "user" | "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)   # avoids mutable-default pitfall
    metadata: dict = field(default_factory=dict)

m1 = Message(role="user", content="How do I reset my password?")
m2 = Message(role="user", content="How do I reset my password?")
print(f"m1: {m1}")
print(f"m1 == m2? {m1 == m2}  (dataclass gives us __eq__ for free)")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Abstract base class — the contract
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: Abstract base class")
print("=" * 70)

class AIAssistant(ABC):
    """Contract every AI assistant must obey.

    Using abc.ABC forces subclasses to implement `respond()` — you'll get a
    TypeError at instantiation if they don't.  This is how mature codebases
    prevent 'forgot to implement' bugs.
    """

    def __init__(self, name: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.model = model
        self._history: list[Message] = []
        self._total_tokens = 0

    # -----  Abstract method — subclass MUST override  -----
    @abstractmethod
    def respond(self, user_message: str) -> str:
        """Produce a reply. Each subclass implements its own style."""

    # -----  Concrete method — shared by all subclasses  -----
    def remember(self, msg: Message) -> None:
        self._history.append(msg)
        self._total_tokens += len(msg.content.split())     # rough token estimate

    # -----  Property — computed attribute with @property  -----
    @property
    def turn_count(self) -> int:
        """Number of user turns in history."""
        return sum(1 for m in self._history if m.role == "user")

    @property
    def token_estimate(self) -> int:
        return self._total_tokens

    # -----  classmethod — alternate constructor  -----
    @classmethod
    def from_config(cls, config: dict):
        """Alternate constructor from a dict — common pattern in AI SDKs."""
        return cls(name=config["name"], model=config.get("model", "gpt-4o-mini"))

    # -----  staticmethod — utility that doesn't need self/cls  -----
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough tiktoken-style estimate: ~0.75 words/token."""
        return int(len(text.split()) / 0.75)

    # -----  Dunder methods — make the object play nice  -----
    def __len__(self) -> int:
        """`len(bot)` returns number of user turns."""
        return self.turn_count

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r}, turns={self.turn_count})"


# ─────────────────────────────────────────────────────────────
# SECTION 3: Two concrete subclasses — polymorphism
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: CustomerSupportBot + CodeReviewBot")
print("=" * 70)

class CustomerSupportBot(AIAssistant):
    """Empathetic tone, always references the ticket, offers escalation."""

    def __init__(self, name: str = "SupportBot", **kwargs):
        super().__init__(name=name, **kwargs)
        self.system_prompt = (
            "You are a warm, empathetic customer support agent. "
            "Always acknowledge the customer's frustration before solving."
        )

    def respond(self, user_message: str) -> str:
        self.remember(Message("user", user_message))
        # MOCK response — showing the STYLE, not real LLM output.
        reply = (
            f"I understand this is frustrating — thank you for reaching out. "
            f"Regarding: '{user_message[:40]}...'. "
            f"Let me help you resolve this."
        )
        self.remember(Message("assistant", reply))
        return reply


class CodeReviewBot(AIAssistant):
    """Terse, technical, focused on style rules."""

    STYLE_RULES = ["PEP 8", "no bare except", "type hints on public API", "no globals in tests"]

    def __init__(self, name: str = "ReviewBot", **kwargs):
        super().__init__(name=name, **kwargs)
        self.system_prompt = (
            f"You are a code reviewer. Follow: {', '.join(self.STYLE_RULES)}. "
            "Be direct. Cite line numbers."
        )

    def respond(self, user_message: str) -> str:
        self.remember(Message("user", user_message))
        # MOCK — pretend to lint the input:
        issues = []
        if "except:" in user_message:
            issues.append("Bare except found — specify exception type.")
        if "print(" in user_message and "def " in user_message:
            issues.append("Remove debug print() in library code.")
        reply = "\n".join(f"- {i}" for i in issues) or "LGTM — no style issues detected."
        self.remember(Message("assistant", reply))
        return reply


# Uncomment to see the ABC in action — this WILL fail:
# class BrokenBot(AIAssistant): pass
# BrokenBot("x")   # TypeError: Can't instantiate abstract class BrokenBot with abstract method respond

# Same call, different behavior — polymorphism:
bots: list[AIAssistant] = [
    CustomerSupportBot(model="gpt-4o-mini"),
    CodeReviewBot(model="gpt-4o"),
]

test_message = "My order didn't arrive. def process(): try: x=1; except: print(x)"
for bot in bots:
    print(f"\n--- {bot} ---")
    print(f"Reply: {bot.respond(test_message)}")

# Show the dunder __len__ working:
print(f"\nSupportBot has {len(bots[0])} user turn(s) recorded.")
print(f"ReviewBot has {len(bots[1])} user turn(s) recorded.")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Properties with validation
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: @property with validation")
print("=" * 70)

class ChatConfig:
    """Show how @property + @setter enforce invariants."""

    def __init__(self, temperature: float = 0.2):
        self.temperature = temperature   # this calls the setter

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"Temperature must be in [0.0, 2.0], got {value}")
        self._temperature = value

cfg = ChatConfig(temperature=0.7)
print(f"Set to 0.7 → {cfg.temperature}")

try:
    cfg.temperature = 5.0
except ValueError as e:
    print(f"Set to 5.0 → blocked: {e}")


# ─────────────────────────────────────────────────────────────
# SECTION 5: classmethod alternate constructor
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: from_config classmethod")
print("=" * 70)

config = {"name": "AlphaBot", "model": "gpt-4o"}
alpha = CustomerSupportBot.from_config(config)
print(f"Built from config: {alpha}")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Static utility — no instance needed
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: staticmethod utility")
print("=" * 70)

approx = AIAssistant.estimate_tokens("The quick brown fox jumps over the lazy dog.")
print(f"Estimated tokens: {approx}")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ Summary")
print("=" * 70)
print("""
The OOP toolkit you'll use in every real AI framework:

  • @dataclass for value objects (Message, Config)
  • ABC + @abstractmethod to enforce a contract across subclasses
  • Polymorphism: `for bot in bots: bot.respond(x)` — one call, N behaviors
  • @property/@setter for validated, computed attributes
  • classmethod for from_config alt constructors
  • staticmethod for pure utilities
  • __len__, __repr__, __eq__ to make objects Pythonic

Now compare: LangChain's BaseChatModel, OpenAI SDK's AsyncClient, Pydantic's
BaseModel — all use exactly these patterns.

Next file: 04_async_programming.py — Parallel LLM calls with asyncio.
""")
