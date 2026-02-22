"""Hallucination detection using LLM-as-judge"""

from typing import Dict, Any, Optional
from llama_index.core.llms import LLM

from ..rag.prompts import HALLUCINATION_CHECK_PROMPT
from ..utils.logging import get_logger

logger = get_logger(__name__)


def check_hallucination(question: str, context: str, answer: str, 
                        llm: Optional[LLM] = None) -> Dict[str, Any]:
    """Check if answer is faithful to context"""
    
    if not llm:
        return {"is_hallucination": False, "confidence": 0.0}
    
    try:
        prompt = HALLUCINATION_CHECK_PROMPT.format(
            question=question,
            context=context[:2000],
            answer=answer[:1000]
        )
        
        response = llm.complete(prompt)
        response_text = str(response).lower()
        
        is_faithful = "yes" in response_text[:50]
        
        return {
            "is_hallucination": not is_faithful,
            "explanation": str(response),
            "confidence": 0.8 if ("yes" in response_text or "no" in response_text) else 0.5
        }
        
    except Exception as e:
        logger.error(f"Hallucination check failed: {e}")
        return {"is_hallucination": False, "confidence": 0.0}


def has_refusal_language(text: str) -> bool:
    """Check if response contains refusal"""
    refusal_phrases = [
        "i don't have", "i cannot", "i'm unable",
        "no information", "not in the", "don't know"
    ]
    
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in refusal_phrases)
