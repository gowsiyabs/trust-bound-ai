"""Jailbreak detection patterns"""

JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|above|prior)",
    r"forget\s+(all\s+)?(previous|above|prior)",
    r"(you\s+are|you're)\s+now\s+in\s+developer\s+mode",
    r"(you\s+are|you're)\s+now\s+in\s+admin\s+mode",
    r"pretend\s+you\s+are\s+(a\s+)?(doctor|lawyer|therapist)",
    r"roleplay\s+as\s+(a\s+)?(doctor|lawyer|therapist)",
    r"act\s+as\s+if\s+you\s+(are|were)",
    r"new\s+instructions?:",
    r"override\s+your",
    r"bypass\s+(all\s+)?restrictions?",
]

INJECTION_PATTERNS = [
    r"(SELECT|INSERT|UPDATE|DELETE|DROP)\s+",
    r"<script[^>]*>",
    r"javascript:",
    r"onerror\s*=",
    r"onclick\s*=",
    r"\{%\s*",
    r"\{\{\s*",
]

PII_EXTRACTION_PATTERNS = [
    # Explicit extraction intent
    r"(list|dump|exfiltrate|export|show|give|tell|reveal)\s+(me\s+)?(all|full|entire|everyone'?s|customer|users?)",
    # Personal data targets
    r"(customer|employee|user|person|people|client|patient).{0,40}(names?|emails?|phone|address|ssn|social\s+security|dob|date\s+of\s+birth)",
    # Bulk data source language
    r"(database|records?|table|dataset|spreadsheet|csv).{0,30}(customer|employee|user|person|client|patient)",
]
