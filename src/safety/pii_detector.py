"""PII detection using Presidio"""

from typing import List, Dict
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from ..utils.logging import get_logger

logger = get_logger(__name__)

_analyzer = None
_anonymizer = None


def get_analyzer():
    """Get or create analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
    return _analyzer


def get_anonymizer():
    """Get or create anonymizer"""
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _anonymizer


def detect_pii(text: str, threshold: float = 0.5) -> List[Dict]:
    """Detect PII in text"""
    try:
        analyzer = get_analyzer()
        results = analyzer.analyze(
            text=text,
            language='en',
            score_threshold=threshold
        )
        return [{"type": r.entity_type, "score": r.score} for r in results]
    except Exception as e:
        logger.error(f"PII detection failed: {e}")
        return []


def redact_pii(text: str, threshold: float = 0.5) -> str:
    """Redact PII from text"""
    try:
        analyzer = get_analyzer()
        anonymizer = get_anonymizer()
        
        results = analyzer.analyze(text=text, language='en', score_threshold=threshold)
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        
        return anonymized.text
    except Exception as e:
        logger.error(f"PII redaction failed: {e}")
        return text
