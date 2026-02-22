"""Harmful content detection"""

HARMFUL_KEYWORDS = [
    "fraud", "scam", "illegal", "cheat", "lie",
    "fake claim", "false information", "manipulate",
]


def is_harmful_content(text: str) -> bool:
    """Check for harmful content"""
    text_lower = text.lower()
    
    for keyword in HARMFUL_KEYWORDS:
        if keyword in text_lower:
            return True
    
    return False


def contains_medical_advice(text: str) -> bool:
    """Check if response contains medical advice"""
    advice_patterns = [
        "you should take", "i recommend taking",
        "try this medication", "stop taking",
        "you have", "you are diagnosed"
    ]
    
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in advice_patterns)
