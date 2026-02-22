"""ChromaDB operations"""

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_chroma_client(persist_dir: Path):
    """Create ChromaDB client"""
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def create_chroma_collection(client, collection_name: str):
    """Create or get ChromaDB collection"""
    try:
        collection = client.get_collection(collection_name)
        logger.info(f"Loaded collection: {collection_name}")
    except:
        collection = client.create_collection(collection_name)
        logger.info(f"Created collection: {collection_name}")
    
    return collection


def collection_exists(client, collection_name: str) -> bool:
    """Check if collection exists"""
    try:
        client.get_collection(collection_name)
        return True
    except:
        return False
