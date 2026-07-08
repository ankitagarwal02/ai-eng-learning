"""
02_output_guardrails.py — Full output-side guardrail stack
=============================================================

WHAT THIS FILE TEACHES
----------------------
  • PII regex detection & redaction (email, phone, SSN, credit card, IP, IBAN)
  • Content moderation (mocked toxicity categories)
  • Response length enforcement
  • JSON schema validation
  • Full lifecycle: check → transform → deliver

HOW TO RUN
----------
    python 02_output_guardrails.py

REAL-WORLD SCENARIO
-------------------
5 different LLM responses, some containing PII, some too long, some malformed.
Each is run through the stack. Actions taken are logged.
"""

import os
import re
import json
from dataclasses import dataclass, field
from enum import Enum

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


class Action(str, Enum):
    PASS = "PASS"
    REDACT = "REDACT"
    BLOCK = "BLOCK"
    LOG = "LOG"


@dataclass
class OutputResult:
    original: str
    transformed: str
    action: Action
    tags: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# PII detection + redaction
# ─────────────────────────────────────────────────────────────
PII_PATTERNS = [
    (re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),                            "EMAIL"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "PHONE"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                                  "SSN"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"),                                "CARD"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),                             "IP"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),                       "IBAN"),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    tags = []
    for pattern, label in PII_PATTERNS:
        if pattern.search(text):
            tags.append(label)
            text = pattern.sub(f"[{label}_REDACTED]", text)
    return text, tags


# ─────────────────────────────────────────────────────────────
# Toxicity / moderation (mock)
# ─────────────────────────────────────────────────────────────
TOXIC_KEYWORDS = {
    "hate":         ["idiot", "moron", "stupid"],
    "self-harm":    ["kill yourself", "end it all"],
    "adult":        ["explicit sexual content marker"],   # not going to hardcode real terms
    "violence":     ["threaten", "beat him up"],
}


def moderate(text: str) -> tuple[Action, list[str]]:
    low = text.lower()
    flags = []
    for category, words in TOXIC_KEYWORDS.items():
        if any(w in low for w in words):
            flags.append(category)
    if flags:
        return Action.BLOCK, flags
    return Action.PASS, []


# ─────────────────────────────────────────────────────────────
# Length enforcement
# ─────────────────────────────────────────────────────────────
def enforce_length(text: str, max_chars: int = 2000) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "…", True


# ─────────────────────────────────────────────────────────────
# JSON schema validation
# ─────────────────────────────────────────────────────────────
def validate_json_shape(text: str, required_keys: list[str]) -> tuple[bool, str]:
    """If the response is *supposed* to be JSON, check shape."""
    text = text.strip()
    if not text.startswith("{"):
        return False, "not JSON"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e.msg}"
    missing = [k for k in required_keys if k not in obj]
    if missing:
        return False, f"missing keys: {missing}"
    return True, "ok"


# ─────────────────────────────────────────────────────────────
# The full output stack
# ─────────────────────────────────────────────────────────────
def output_stack(text: str, expected_json_keys: list[str] | None = None,
                 max_chars: int = 2000) -> OutputResult:
    tags: list[str] = []
    transformed = text
    action = Action.PASS

    # Layer 1: Moderation (BLOCK on hit)
    m_action, m_flags = moderate(text)
    if m_action == Action.BLOCK:
        return OutputResult(
            original=text,
            transformed="I can't provide that response. If you'd like, I can try a different phrasing.",
            action=Action.BLOCK,
            tags=[f"moderation:{f}" for f in m_flags],
        )

    # Layer 2: PII redaction (REDACT if any found)
    transformed, pii_tags = redact_pii(transformed)
    if pii_tags:
        action = Action.REDACT
        tags.extend([f"pii:{t}" for t in pii_tags])

    # Layer 3: Length
    transformed, truncated = enforce_length(transformed, max_chars)
    if truncated:
        if action != Action.REDACT:
            action = Action.REDACT
        tags.append("truncated")

    # Layer 4: JSON validation (optional)
    if expected_json_keys is not None:
        ok, reason = validate_json_shape(transformed, expected_json_keys)
        if not ok:
            return OutputResult(
                original=text,
                transformed=json.dumps({"error": f"schema validation failed: {reason}"}),
                action=Action.BLOCK,
                tags=[*tags, f"json-invalid:{reason}"],
            )
        tags.append("json-ok")

    return OutputResult(original=text, transformed=transformed, action=action, tags=tags)


# ─────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────
TESTS = [
    ("O01", "Your account is fine. Contact support at support@example.com or 555-123-4567.", None),
    ("O02", "Verified. Your card 4242 4242 4242 4242 was charged $99.",                       None),
    ("O03", "Login successful from IP 192.168.1.100. SSN 123-45-6789 on file.",              None),
    ("O04", "OK — moving on.",                                                                 None),
    ("O05", "You're a total idiot for asking that.",                                           None),
    ("O06", "A" * 2500,                                                                        None),
    ("O07", '{"status":"ok","id":"abc"}',                                                     ["status", "id"]),
    ("O08", '{"status":"ok"}',                                                                ["status", "id"]),
    ("O09", 'plain text where JSON was expected',                                             ["status"]),
    ("O10", "Standard clean response about our refund policy.",                               None),
]


def main():
    print("=" * 90)
    print(f"{'ID':<5}{'Action':<8}{'Tags':<40}Sample of output")
    print("=" * 90)

    for tid, text, req_keys in TESTS:
        r = output_stack(text, expected_json_keys=req_keys)
        tags_str = ",".join(r.tags)[:38]
        preview = r.transformed[:60].replace("\n", " ")
        print(f"{tid:<5}{r.action.value:<8}{tags_str:<40}{preview}")


if __name__ == "__main__":
    main()
    print("""

✅ Summary — Output guardrail stack:

  1. Moderation           BLOCK toxicity, adult, self-harm, violence
  2. PII redaction        REDACT emails, phone, SSN, card, IP, IBAN
  3. Length cap           REDACT (truncate) if over max_chars
  4. JSON schema          BLOCK if structured output expected but malformed

Combined with Phase 15 §01 (input guardrails), you have:

    Layer 0  Rate limit (input)
    Layer 1  Injection regex (input)
    Layer 2  Injection LLM judge (input)
    Layer 3  Topic scope (input)
    Layer 4  Content moderation (output)
    Layer 5  PII redaction (output)
    Layer 6  Length cap (output)
    Layer 7  Schema validation (output)

Best practice: separate stacks for input vs output; run them ALWAYS, not on
sample. Measure block-rate and redaction-rate per model — sudden spikes
signal a compromised prompt or new attack pattern.
""")
