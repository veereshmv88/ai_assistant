"""
ai/intent_parser.py — Voice Intent Classification
===================================================
Two-tier intent classification:

Tier 1 — Rule-based fast path (no LLM round-trip):
  Matches common command patterns via regex/keywords.
  Latency: < 1ms

Tier 2 — LLM classification (Ollama llama3.2):
  For complex, ambiguous queries.
  Latency: 1-5 seconds on Pi

Intent types:
  DESCRIBE    — "What do I see?", "What's in front of me?"
  NAVIGATE    — "Guide me to the bus stop", "How do I get to the exit?"
  READ        — "Read this sign", "What does this say?"
  FACE        — "Who is in front of me?", "Do I know this person?"
  CURRENCY    — "What note is this?", "How much money am I holding?"
  EMERGENCY   — "Help!", "SOS", "Call for help"
  REMEMBER    — "Remember this person as Mom"
  STOP_NAV    — "Stop navigation", "Cancel route"
  UNKNOWN     — Unclassified

Provides:
  • async classify(text)   → Intent
"""

import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


class IntentType(str, Enum):
    DESCRIBE   = "describe"
    NAVIGATE   = "navigate"
    READ       = "read"
    FACE       = "face"
    CURRENCY   = "currency"
    EMERGENCY  = "emergency"
    REMEMBER   = "remember"
    STOP_NAV   = "stop_nav"
    STATUS     = "status"
    UNKNOWN    = "unknown"


@dataclass
class Intent:
    type: IntentType
    text: str                          # original user text
    entity: Optional[str] = None      # extracted entity (e.g., destination name)
    confidence: float = 1.0
    method: str = "rule"              # "rule" | "llm"


# ── Pattern library (ordered by specificity) ──────────────────────────────────
_PATTERNS: list[tuple[IntentType, list[str]]] = [
    (IntentType.EMERGENCY, [
        r"\bsos\b", r"\bhelp\b", r"\bemergency\b", r"\bcall for help\b",
        r"\bi('m| am) (lost|stuck|hurt|injured|in danger)\b",
        r"\bcall.*(police|ambulance|doctor)\b",
    ]),
    (IntentType.READ, [
        r"\bread\b", r"\bwhat does (this|that|it) say\b",
        r"\bwhat('s| is) written\b", r"\bread (the )?(sign|text|label|board)\b",
        r"\bspell\b",
    ]),
    (IntentType.FACE, [
        r"\bwho (is|are)\b", r"\bdo i know\b", r"\brecognize\b",
        r"\bperson (in front|ahead|there)\b", r"\bsomeone (there|here)\b",
    ]),
    (IntentType.CURRENCY, [
        r"\bwhat (note|money|rupee|currency|bill|coin)\b",
        r"\bhow much.*(holding|hand|this)\b",
        r"\bidentify.*(money|currency|note)\b",
        r"\b(500|100|200|50|20|10) rupee\b",
    ]),
    (IntentType.REMEMBER, [
        r"\bremember (this|that) (person|face) as\b",
        r"\bsave (this|that) (person|face)\b",
        r"\bthis is my\b",
        r"\blearn (this|that) face\b",
    ]),
    (IntentType.STOP_NAV, [
        r"\bstop (navigation|guiding|route)\b",
        r"\bcancel (route|navigation|directions)\b",
        r"\bi('ve| have) arrived\b",
        r"\bstop (following|navigating)\b",
    ]),
    (IntentType.NAVIGATE, [
        r"\b(guide|navigate|take|lead|bring) me (to|toward)\b",
        r"\bhow do i get to\b",
        r"\bdirections? to\b",
        r"\bwhere is\b",
        r"\bfind.*(bus stop|exit|entrance|door|shop|hospital|pharmacy)\b",
        r"\bgo to\b",
    ]),
    (IntentType.DESCRIBE, [
        r"\bwhat (is|are|do you see|can you see)\b",
        r"\bwhat('s| is) (in front|ahead|around|near|behind|to (my )?(left|right))\b",
        r"\bdescribe\b", r"\btell me (about|what)\b",
        r"\bam i safe\b", r"\bis it safe\b",
        r"\bcan i cross\b", r"\bclear\b", r"\bpath\b",
        r"\bany (obstacle|danger|hazard)\b",
        r"\blook\b",
    ]),
    (IntentType.STATUS, [
        r"\bstatus\b", r"\bhow are you\b", r"\bare you (ok|okay|working|running)\b",
        r"\bsystem (check|status|health)\b",
    ]),
]


class IntentParser:
    """
    Fast rule-based intent classifier with LLM fallback.
    """

    def __init__(self, cfg: Config, llm=None):
        self.cfg = cfg
        self._llm = llm
        # Pre-compile patterns
        self._compiled = [
            (intent, [re.compile(p, re.IGNORECASE) for p in patterns])
            for intent, patterns in _PATTERNS
        ]

    async def initialise(self):
        log.info("Intent parser ready.")

    async def cleanup(self):
        pass

    # ── Classification ────────────────────────────────────────────────────────
    async def classify(self, text: str) -> Intent:
        """Classify user text into an Intent."""
        if not text or not text.strip():
            return Intent(type=IntentType.UNKNOWN, text=text, confidence=0.0)

        # Tier 1: rule-based
        intent = self._rule_classify(text)
        if intent.type != IntentType.UNKNOWN:
            return intent

        # Tier 2: LLM classification
        if self._llm and self._llm.is_ready():
            return await self._llm_classify(text)

        return Intent(type=IntentType.UNKNOWN, text=text, confidence=0.3)

    def _rule_classify(self, text: str) -> Intent:
        """Apply regex patterns in priority order."""
        text_lower = text.lower().strip()

        for intent_type, patterns in self._compiled:
            for pattern in patterns:
                if pattern.search(text_lower):
                    entity = self._extract_entity(text_lower, intent_type)
                    log.debug(f"Intent (rule): {intent_type.value} — '{text[:50]}'")
                    return Intent(
                        type=intent_type,
                        text=text,
                        entity=entity,
                        confidence=0.95,
                        method="rule",
                    )

        return Intent(type=IntentType.UNKNOWN, text=text, confidence=0.0)

    async def _llm_classify(self, text: str) -> Intent:
        """Use LLM to classify ambiguous queries."""
        intent_list = ", ".join(t.value for t in IntentType if t != IntentType.UNKNOWN)
        prompt = (
            f"Classify this spoken query from a blind person into one of these intents: {intent_list}.\n"
            f"Query: '{text}'\n"
            f"Reply with ONLY the intent name (one word from the list above)."
        )
        try:
            response = await self._llm.query(prompt)
            response = response.strip().lower()
            intent_type = IntentType(response) if response in IntentType._value2member_map_ else IntentType.UNKNOWN
            entity = self._extract_entity(text.lower(), intent_type)
            log.debug(f"Intent (LLM): {intent_type.value} — '{text[:50]}'")
            return Intent(
                type=intent_type,
                text=text,
                entity=entity,
                confidence=0.8,
                method="llm",
            )
        except Exception as e:
            log.warning(f"LLM intent classification error: {e}")
            return Intent(type=IntentType.UNKNOWN, text=text, confidence=0.0)

    # ── Entity extraction ─────────────────────────────────────────────────────
    @staticmethod
    def _extract_entity(text: str, intent: IntentType) -> Optional[str]:
        """Extract key entity from text based on intent type."""
        if intent == IntentType.NAVIGATE:
            # Extract destination: "guide me to the BUS STOP" → "bus stop"
            match = re.search(
                r"(?:to|toward|find|get to|go to)\s+(?:the\s+)?(.+?)(?:\s*$|\s+please)",
                text, re.IGNORECASE
            )
            return match.group(1).strip() if match else None

        if intent == IntentType.REMEMBER:
            # "Remember this person as MOM" → "Mom"
            match = re.search(r"(?:as|named?)\s+(\w+)", text, re.IGNORECASE)
            return match.group(1).title() if match else None

        return None

    def is_ready(self) -> bool:
        return True
