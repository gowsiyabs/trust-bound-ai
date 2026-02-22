"""Document chunking utilities"""

from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from ..utils.config import ChunkingConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


def chunk_documents(documents: List[Document], config: ChunkingConfig) -> List[Document]:
    """Split documents into chunks"""
    if not documents:
        return []
    
    logger.info(f"Chunking {len(documents)} documents")
    
    splitter = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    
    chunked_docs = []
    for node in nodes:
        doc = Document(
            text=node.get_content(),
            metadata=node.metadata,
        )
        chunked_docs.append(doc)
    
    logger.info(f"Created {len(chunked_docs)} chunks")
    return chunked_docs
