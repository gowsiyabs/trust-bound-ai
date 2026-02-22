"""RAG pipeline components for document Q&A (dataset-agnostic)."""

from .document_loader import DocumentLoader, InsuranceDocumentLoader  # alias kept
from .embeddings import EmbeddingModelManager
from .retriever import HybridRetriever
from .query_engine import RAGQueryEngine, InsuranceQueryEngine  # alias kept

# Backwards-compat alias
EmbeddingManager = EmbeddingModelManager

__all__ = [
    "DocumentLoader",
    "EmbeddingModelManager",
    "EmbeddingManager",
    "HybridRetriever",
    "RAGQueryEngine",
    # backwards-compat aliases
    "InsuranceDocumentLoader",
    "InsuranceQueryEngine",
]
