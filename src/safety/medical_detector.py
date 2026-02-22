"""Medical advice detection"""

MEDICAL_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "bleeding", 
    "suicide", "overdose", "seizure", "unconscious",
    "diagnosis", "prescribe", "medication for",
    "what medication", "which drug", "should i take",
]

DIAGNOSIS_PATTERNS = [
    r"do\s+i\s+have\s+(a\s+)?(\w+)",
    r"is\s+it\s+(a\s+)?(\w+\s+)?(disease|condition|illness)",
    r"diagnose\s+me",
]


def is_medical_advice_request(text: str) -> bool:
    """Check if query requests medical advice"""
    import re
    
    text_lower = text.lower()
    
    for keyword in MEDICAL_KEYWORDS:
        if keyword in text_lower:
            return True
    
    for pattern in DIAGNOSIS_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    
    return False
