"""Vector store manager - modularized version"""

from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext

try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
except ImportError:
    from llama_index_vector_stores_chroma import ChromaVectorStore

from .chroma_ops import create_chroma_client, create_chroma_collection, collection_exists
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store"""
    
    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.client = create_chroma_client(persist_dir)
        logger.info(f"Vector store: {persist_dir}")
    
    def create(self, collection_name: str):
        """Create new vector store"""
        collection = create_chroma_collection(self.client, collection_name)
        return ChromaVectorStore(chroma_collection=collection)
    
    def load_index(self, collection_name: str) -> VectorStoreIndex:
        """Load existing index"""
        vector_store = self.create(collection_name)
        return VectorStoreIndex.from_vector_store(vector_store)
    
    def exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        return collection_exists(self.client, collection_name)
