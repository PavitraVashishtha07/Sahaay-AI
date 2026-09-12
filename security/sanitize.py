"""
Sahaay AI — Security & Risk Gateway: LLM Input Sanitizer (Risk 7 & 8)

Guards all free-text fields (user chat messages, merchant names, profile notes)
against Prompt Injection, Jailbreak attempts, and Control Token Smuggling before
they can reach conversational or reasoning LLM components.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Tuple

# High-risk prompt injection and jailbreak phrases (case-insensitive)
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|previous\s+|prior\s+|above\s+|the\s+)*instructions?",
    r"disregard\s+(?:all\s+|previous\s+|prior\s+|the\s+)*(?:instructions?|rules?|prompts?)",
    r"forget\s+(?:all\s+|everything\s+|previous\s+|prior\s+|the\s+)*(?:instructions?|rules?|prompts?)",
    r"you\s+are\s+now\s+(?:in\s+)?(?:developer\s+mode|unrestricted|dan|jailbreak)",
    r"bypass\s+(?:all\s+)?(?:safety|security|rules|guardrails)",
    r"system\s+(?:prompt|override|command)",
    r"act\s+as\s+(?:an?\s+)?unrestricted",
    r"output\s+only\s+(?:the\s+)?(?:flag|password|key|token|system)",
    r"show\s+(?:me\s+)?(?:the\s+)?system\s+prompt",
    r"approve\s+(?:the\s+)?loan\s+unconditionally",
    r"grant\s+admin\s+access",
]

COMPILED_INJECTION_REGEX = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# Code injection & markup patterns
DANGEROUS_MARKUP = [
    r"```[a-zA-Z0-9_-]*",
    r"~~~[a-zA-Z0-9_-]*",
    r"<\s*script[^>]*>.*?<\s*/\s*script\s*>",
    r"<\s*iframe[^>]*>.*?<\s*/\s*iframe\s*>",
    r"javascript\s*:",
    r"data\s*:\s*text/html",
]

COMPILED_MARKUP_REGEX = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in DANGEROUS_MARKUP]


def remove_control_and_zero_width_chars(text: str) -> str:
    """Removes null bytes, zero-width spaces, and hidden unicode smuggling characters."""
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Keep standard letters, numbers, punctuation, symbols, and normal whitespace (space, newline, tab)
        if ch in "\n\r\t ":
            cleaned.append(ch)
        elif cat in ("Cc", "Cf", "Cs", "Co", "Cn"):
            # Control, Format (e.g. zero-width), Surrogate, Private Use, Unassigned
            continue
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def is_suspicious_llm_input(text: str) -> bool:
    """Returns True if the text contains high-risk prompt injection patterns."""
    if not text:
        return False
    
    clean_text = remove_control_and_zero_width_chars(text)
    for pattern in COMPILED_INJECTION_REGEX:
        if pattern.search(clean_text):
            return True
    return False


def sanitize_for_llm(text: str, max_length: int = 1000) -> str:
    """
    Sanitizes user messages or dynamic database text (e.g. merchant names)
    before feeding to the conversational LLM layer.
    
    1. Removes invisible / zero-width characters and control tokens.
    2. Strips prompt-injection patterns (e.g. 'ignore previous instructions').
    3. Strips code fences and script tags.
    4. Truncates excessively long inputs to bounded length.
    """
    if not text:
        return ""

    # Step 1: Strip control & hidden characters
    sanitized = remove_control_and_zero_width_chars(text)

    # Step 2: Strip markup & code fences
    for pattern in COMPILED_MARKUP_REGEX:
        sanitized = pattern.sub(" ", sanitized)

    # Step 3: Neutralize prompt injection overrides
    for pattern in COMPILED_INJECTION_REGEX:
        sanitized = pattern.sub("[FILTERED_INJECTION_ATTEMPT]", sanitized)

    # Step 4: Normalize repeated spaces
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    # Step 5: Bounded length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].strip()

    return sanitized
