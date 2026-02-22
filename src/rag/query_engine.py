"""Query engine - dataset-agnostic RAG query engine."""

from typing import Dict, Any, Optional
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.prompts import PromptTemplate

from .prompts import DOCUMENT_QA_PROMPT_TEMPLATE
from .response_formatter import (
    format_sources,
    calculate_confidence,
    format_query_response,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RAGQueryEngine:
    """Query engine with safety integration"""
    
    def __init__(self, index: VectorStoreIndex, retriever, llm):
        self.index = index
        self.retriever = retriever
        self.llm = llm
        self.qa_prompt = PromptTemplate(DOCUMENT_QA_PROMPT_TEMPLATE)
        logger.info("Query engine initialized")
    
    def query(self, query_text: str, input_validator=None, 
              output_filter=None) -> Dict[str, Any]:
        """Execute query with optional safety checks"""
        
        disclaimer = None
        if input_validator:
            validation = input_validator.validate(query_text)
            if not validation.get("is_valid", True):
                return self._blocked_response(validation["reason"])
            if validation.get("medical_flag"):
                disclaimer = "Note: this system provides document information only, not medical advice."

        try:
            retrieved = self.retriever.retrieve(query_text)
            answer = self._generate_answer(query_text, retrieved)
            if disclaimer:
                answer = f"{answer}\n\n_{disclaimer}_"

            if output_filter:
                filter_result = output_filter.filter(answer, query_text, retrieved)
                if filter_result["blocked"]:
                    return self._blocked_response(filter_result["reason"])
            
            return self._format_success_response(answer, retrieved)
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return self._error_response(str(e))
    
    def _generate_answer(self, query: str, nodes) -> str:
        """Generate answer from retrieved nodes"""
        context = "\n\n".join([node.get_content() for node in nodes])
        
        prompt = self.qa_prompt.format(
            context_str=context,
            query_str=query
        )
        
        response = self.llm.complete(prompt)
        return str(response)
    
    def _format_success_response(self, answer: str, nodes) -> Dict[str, Any]:
        """Format successful response."""
        sources = format_sources(nodes)
        score = nodes[0].score if nodes else 0.0
        confidence = calculate_confidence(score, len(nodes))
        return format_query_response(
            answer, sources, confidence, source_nodes=nodes
        )

    def _blocked_response(self, reason: str) -> Dict[str, Any]:
        """Format blocked response."""
        return format_query_response(
            answer="I cannot process this request.",
            sources="N/A",
            confidence=0.0,
            blocked=True,
            warning=reason,
        )

    def _error_response(self, error: str) -> Dict[str, Any]:
        """Format error response."""
        return format_query_response(
            answer="An error occurred processing your request.",
            sources="N/A",
            confidence=0.0,
            warning=f"Error: {error}",
        )


# Backwards-compat alias
InsuranceQueryEngine = RAGQueryEngine
