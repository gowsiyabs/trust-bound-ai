"""Input validator - modularized version"""

from typing import Dict, Any
from .patterns import JAILBREAK_PATTERNS, INJECTION_PATTERNS, PII_EXTRACTION_PATTERNS
from .pattern_matcher import has_jailbreak_attempt, has_injection_attempt, has_pii_extraction
from .medical_detector import is_medical_advice_request

from ..utils.config import SafetyConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class InputValidator:
    """Validates user input for safety."""

    def __init__(self, config: SafetyConfig = None):
        from ..utils.config import get_global_config
        self.config = config or get_global_config().safety
        self.max_length = 5000
        logger.info("Input validator initialized")
    
    def validate(self, query: str) -> Dict[str, Any]:
        """Validate input query"""
        
        if not query or not query.strip():
            return self._invalid("Empty query")
        
        if len(query) > self.max_length:
            return self._invalid(f"Query too long (max {self.max_length})")
        
        if has_jailbreak_attempt(query, JAILBREAK_PATTERNS):
            logger.warning("Jailbreak attempt detected")
            return self._invalid("Invalid request detected")
        
        if has_injection_attempt(query, INJECTION_PATTERNS):
            logger.warning("Injection attempt detected")
            return self._invalid("Invalid request format")
        
        if has_pii_extraction(query, PII_EXTRACTION_PATTERNS):
            logger.warning("PII extraction attempt")
            return self._invalid("Cannot provide confidential information")
        
        if is_medical_advice_request(query):
            logger.warning("Medical-adjacent query detected; passing through with flag")
            # Not hard-blocked; let the RAG answer and rely on output filtering.
            # Return valid but annotate so downstream can attach a disclaimer.
            return {"is_valid": True, "medical_flag": True}
        
        return {"is_valid": True}
    
    def _invalid(self, reason: str) -> Dict[str, Any]:
        """Format invalid response"""
        return {
            "is_valid": False,
            "reason": reason
        }
