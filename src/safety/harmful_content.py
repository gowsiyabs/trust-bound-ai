"""Harmful content detection"""

# Phrase-level patterns that indicate the model is GENERATING harmful
# instructions rather than DESCRIBING corporate misconduct from a filing.
# Single words like "fraud", "illegal", "scam" are intentionally excluded:
# they appear routinely in legitimate 10-K risk-factor and litigation text.
HARMFUL_PHRASES = [
    "here is how to commit fraud",
    "step by step to scam",
    "instructions for illegal",
    "how to illegally obtain",
    "how to manipulate the market",
    "i will help you cheat",
    "how to launder money",
    "guide to evading taxes",
]


def is_harmful_content(text: str) -> bool:
    """Check for harmful content"""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in HARMFUL_PHRASES)


def contains_medical_advice(text: str) -> bool:
    """Check if response contains medical advice"""
    advice_patterns = [
        "you should take", "i recommend taking",
        "try this medication", "stop taking",
        "you have", "you are diagnosed"
    ]
    
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in advice_patterns)
