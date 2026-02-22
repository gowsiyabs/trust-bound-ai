"""Index building utilities - dataset-agnostic."""

from pathlib import Path
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext

from .document_loader import DocumentLoader
from .vector_store import VectorStoreManager
from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_index_from_store(persist_dir: Path, collection_name: str) -> VectorStoreIndex:
    """Load existing vector index from disk. No document loading or embedding."""
    manager = VectorStoreManager(persist_dir=persist_dir)
    if not manager.exists(collection_name):
        raise FileNotFoundError(
            f"No index found for collection '{collection_name}' at {persist_dir}. "
            "Run: python scripts/build_index.py"
        )
    return manager.load_index(collection_name)


def build_index_from_documents(
    documents_dir: Path,
    collection_name: str,
    persist_dir: Path,
    force_rebuild: bool = False,
) -> VectorStoreIndex:
    """Build vector index from documents (or load existing if present)."""

    vector_store_manager = VectorStoreManager(persist_dir=persist_dir)

    if not force_rebuild and vector_store_manager.exists(collection_name):
        logger.info(f"Loading existing index: {collection_name}")
        return vector_store_manager.load_index(collection_name)

    logger.info("Building new index...")
    loader = DocumentLoader()
    documents = loader.load_documents(documents_dir, show_progress=True)

    if not documents:
        raise ValueError("No documents loaded")

    vector_store = vector_store_manager.create(collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    logger.info("Index built successfully")
    return index
