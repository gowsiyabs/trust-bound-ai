"""Pattern matching utilities"""

import re
from typing import List, Optional


def check_patterns(text: str, patterns: List[str]) -> Optional[str]:
    """Check if text matches any pattern"""
    text_lower = text.lower()
    
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return f"Matched pattern: {pattern}"
    
    return None


def has_jailbreak_attempt(text: str, patterns: List[str]) -> bool:
    """Check for jailbreak attempt"""
    return check_patterns(text, patterns) is not None


def has_injection_attempt(text: str, patterns: List[str]) -> bool:
    """Check for injection attempt"""
    return check_patterns(text, patterns) is not None


def has_pii_extraction(text: str, patterns: List[str]) -> bool:
    """Check for high-risk PII extraction attempts.

    We require multiple suspicious signals to avoid blocking legitimate
    business disclosure questions (e.g., headquarters address in a 10-K).
    """
    text_lower = text.lower()
    hits = 0
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            hits += 1

    # Two-or-more signals reduces false positives from single words like
    # "address" or "phone" in legitimate finance/compliance questions.
    return hits >= 2
