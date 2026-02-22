"""Output filter - modularized version"""

from typing import Dict, Any, List
from .pii_detector import detect_pii, redact_pii
from .hallucination_checker import check_hallucination, has_refusal_language
from .harmful_content import is_harmful_content, contains_medical_advice

from ..utils.config import SafetyConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OutputFilter:
    """Filters LLM output for safety."""

    # LOCATION appears frequently in legitimate public company disclosures
    # (e.g., principal executive office address in SEC filings).
    BLOCKED_PII_TYPES = {
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "CRYPTO",
        "IP_ADDRESS",
        "US_BANK_NUMBER",
        "IBAN_CODE",
        "US_PASSPORT",
        "MEDICAL_LICENSE",
    }

    def __init__(self, config: SafetyConfig = None, llm=None):
        from ..utils.config import get_global_config
        self.config = config or get_global_config().safety
        self.llm = llm
        self.pii_threshold = 0.5
        logger.info("Output filter initialized")
    
    def filter(self, answer: str, query: str, context_nodes: List) -> Dict[str, Any]:
        """Filter output for safety"""
        
        if has_refusal_language(answer):
            return {"blocked": False}
        
        if self.config.enable_pii_detection:
            pii_result = self._check_pii(answer)
            if pii_result["blocked"]:
                return pii_result
        
        if self.config.enable_hallucination_detection and self.llm:
            hall_result = self._check_hallucination(query, context_nodes, answer)
            if hall_result["blocked"]:
                return hall_result
        
        if is_harmful_content(answer):
            logger.warning("Harmful content detected")
            return {"blocked": True, "reason": "Harmful content detected"}
        
        if contains_medical_advice(answer):
            logger.warning("Medical advice detected")
            return {"blocked": True, "reason": "Cannot provide medical advice"}
        
        return {"blocked": False}
    
    def _check_pii(self, answer: str) -> Dict[str, Any]:
        """Check for PII leakage"""
        pii_entities = detect_pii(answer, self.pii_threshold)

        blocked_entities = [
            e for e in pii_entities
            if e.get("type", "").upper() in self.BLOCKED_PII_TYPES
        ]

        if blocked_entities:
            logger.warning(f"PII detected: {blocked_entities}")
            return {
                "blocked": True,
                "reason": "PII detected in response",
                "pii_types": [e["type"] for e in blocked_entities]
            }
        
        return {"blocked": False}
    
    def _check_hallucination(self, query: str, nodes: List, answer: str) -> Dict[str, Any]:
        """Check for hallucination"""
        context = "\n".join([n.get_content() for n in nodes])
        
        result = check_hallucination(query, context, answer, self.llm)
        
        if result["is_hallucination"] and result["confidence"] > 0.7:
            logger.warning("Hallucination detected")
            return {
                "blocked": True,
                "reason": "Answer not supported by sources"
            }
        
        return {"blocked": False}
