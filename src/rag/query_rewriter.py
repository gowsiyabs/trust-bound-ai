"""Query rewriting utilities"""

from typing import Optional
from llama_index.core.llms import LLM

from .prompts import QUERY_REWRITE_PROMPT_TEMPLATE
from ..utils.logging import get_logger

logger = get_logger(__name__)


def rewrite_query(query: str, llm: Optional[LLM] = None) -> str:
    """Rewrite query for better retrieval"""
    if not llm:
        return query
    
    try:
        prompt = QUERY_REWRITE_PROMPT_TEMPLATE.format(query=query)
        response = llm.complete(prompt)
        rewritten = str(response).strip()
        
        if rewritten and len(rewritten) > 5:
            logger.debug(f"Rewritten: {query} -> {rewritten}")
            return rewritten
        
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}")
    
    return query
