"""Hybrid retriever - modularized version"""

from typing import List, Optional, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

from .query_rewriter import rewrite_query
from .score_fusion import normalize_scores
from .metadata_filter import apply_metadata_filter, extract_metadata_from_query
from ..utils.config import RetrievalConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Multi-stage hybrid retriever"""
    
    def __init__(self, index: VectorStoreIndex, config: RetrievalConfig, 
                 llm=None, reranker=None):
        self.index = index
        self.config = config
        self.llm = llm
        self.reranker = reranker
        
        self.semantic_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=config.top_k * 2,
        )
        
        logger.info("Hybrid retriever initialized")
    
    def retrieve(self, query: str, metadata_filters: Optional[Dict] = None):
        """Execute multi-stage retrieval"""
        
        if self.config.use_query_rewriting and self.llm:
            query = rewrite_query(query, self.llm)
        
        nodes = self._semantic_search(query)
        
        if not nodes:
            logger.warning("No results from semantic search")
            return []
        
        nodes = normalize_scores(nodes)
        
        if metadata_filters is None:
            metadata_filters = extract_metadata_from_query(query)
        
        if metadata_filters:
            nodes = apply_metadata_filter(nodes, metadata_filters)
        
        if self.reranker and len(nodes) > self.config.top_k:
            nodes = self._rerank(query, nodes)
        
        return nodes[:self.config.top_k]
    
    def _semantic_search(self, query: str):
        """Semantic search"""
        try:
            return self.semantic_retriever.retrieve(query)
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _rerank(self, query: str, nodes):
        """Rerank nodes"""
        try:
            return self.reranker.postprocess_nodes(nodes, query_str=query)
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return nodes
