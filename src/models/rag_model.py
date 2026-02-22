"""RAG model - dataset-agnostic orchestrator"""

from pathlib import Path
from typing import Dict, Any, Optional

from ..rag.index_builder import build_index_from_documents, load_index_from_store
from ..rag.retriever import HybridRetriever
from ..rag.query_engine import RAGQueryEngine
from ..rag.embeddings import initialize_embeddings
from .llm_factory import create_llm
from ..utils.config import get_global_config
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_rag_model(
    documents_dir: Path,
    force_reindex: bool = False,
    index_persist_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
) -> "RAGModel":
    """Build or load RAG from documents. Use for indexing or when index may not exist."""
    config = get_global_config()
    persist = index_persist_dir or Path(config.rag.vector_store.persist_directory)
    coll = collection_name or config.rag.vector_store.collection_name
    return RAGModel(
        documents_dir=documents_dir,
        index_persist_dir=persist,
        collection_name=coll,
        force_rebuild=force_reindex,
    )


def load_rag_from_index(
    index_persist_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
) -> "RAGModel":
    """Load RAG from existing index only (no document loading/embedding). Fast for UI."""
    config = get_global_config()
    persist = index_persist_dir or Path(config.rag.vector_store.persist_directory)
    coll = collection_name or config.rag.vector_store.collection_name
    return RAGModel.from_index(index_persist_dir=persist, collection_name=coll)


class RAGModel:
    """Main RAG system orchestrator - dataset-agnostic."""

    def __init__(
        self,
        documents_dir: Path,
        index_persist_dir: Path,
        collection_name: str = "rag_docs",
        force_rebuild: bool = False,
    ):
        self.config = get_global_config()
        self.documents_dir = documents_dir
        self.index_persist_dir = index_persist_dir
        self.collection_name = collection_name

        logger.info("Initializing RAG system (from documents)...")
        # Must initialise embedding model BEFORE building/loading the index so
        # LlamaIndex uses the local HuggingFace model, not the OpenAI default.
        initialize_embeddings(self.config.rag.embeddings)
        self.llm = create_llm(self.config.rag.llm)
        self.index = build_index_from_documents(
            documents_dir, collection_name, index_persist_dir, force_rebuild
        )
        self.retriever = HybridRetriever(
            index=self.index,
            config=self.config.rag.retrieval,
            llm=self.llm,
        )
        self.query_engine = RAGQueryEngine(
            index=self.index,
            retriever=self.retriever,
            llm=self.llm,
        )
        logger.info("RAG system ready")

    @classmethod
    def from_index(
        cls,
        index_persist_dir: Path,
        collection_name: str,
    ) -> "RAGModel":
        """Load RAG from persisted index only. No document loading or embedding."""
        config = get_global_config()
        logger.info("Loading RAG from existing index (fast)...")
        # Initialise local embedding model before loading from vector store.
        initialize_embeddings(config.rag.embeddings)
        llm = create_llm(config.rag.llm)
        index = load_index_from_store(index_persist_dir, collection_name)
        retriever = HybridRetriever(
            index=index,
            config=config.rag.retrieval,
            llm=llm,
        )
        query_engine = RAGQueryEngine(
            index=index,
            retriever=retriever,
            llm=llm,
        )
        instance = cls.__new__(cls)
        instance.config = config
        instance.documents_dir = None
        instance.index_persist_dir = index_persist_dir
        instance.collection_name = collection_name
        instance.llm = llm
        instance.index = index
        instance.retriever = retriever
        instance.query_engine = query_engine
        logger.info("RAG system ready (from index)")
        return instance

    def query(
        self,
        query_text: str,
        input_validator=None,
        output_filter=None,
    ) -> Dict[str, Any]:
        """Execute query."""
        return self.query_engine.query(query_text, input_validator, output_filter)

    def query_with_safety(self, query_text: str, input_validator, output_filter):
        """Query with safety checks."""
        return self.query(query_text, input_validator, output_filter)

    def get_stats(self) -> Dict[str, Any]:
        """Return index stats for UI."""
        try:
            ref_doc_info = self.index.docstore.get_ref_doc_info()
            num_nodes = len(ref_doc_info) if ref_doc_info else 0
            return {"num_nodes": num_nodes, "status": "ready"}
        except Exception:
            return {"num_nodes": 0, "status": "unknown"}
