"""Document loader - dataset-agnostic PDF/text/CSV ingestion."""

from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

from llama_index.core import Document
from .pdf_loader import load_pdf_file
from .text_loader import load_text_file
from .csv_loader import load_csv_file
from .chunker import chunk_documents
from .metadata_extractor import is_duplicate

from ..utils.config import ChunkingConfig, get_global_config
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Enhanced document loader for PDFs and text files"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        if config is None:
            config = get_global_config().rag.chunking
        
        self.config = config
        self.loaded_hashes: set = set()
        logger.info(f"Initialized with chunk_size={config.chunk_size}")
    
    def load_documents(self, directory: Path, show_progress: bool = True) -> List[Document]:
        """Load all documents from directory (PDF, TXT, CSV)."""
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        pdf_files = list(directory.glob("*.pdf"))
        txt_files = list(directory.glob("*.txt"))
        csv_files = list(directory.glob("*.csv"))
        all_files = pdf_files + txt_files + csv_files

        if not all_files:
            logger.warning(f"No files found in {directory}")
            return []

        logger.info(
            f"Found {len(pdf_files)} PDFs, {len(txt_files)} text files, "
            f"{len(csv_files)} CSVs in {directory}"
        )
        
        documents = self._load_files(all_files, show_progress)
        
        if not documents:
            logger.warning("No documents loaded")
            return []
        
        logger.info(f"Loaded {len(documents)} documents successfully")
        return chunk_documents(documents, self.config)
    
    def _load_files(self, files: List[Path], show_progress: bool) -> List[Document]:
        """Load all files"""
        documents = []
        iterator = tqdm(files, desc="Loading") if show_progress else files
        
        for file_path in iterator:
            doc_list = self._load_single_file(file_path)
            
            for doc in doc_list:
                doc_hash = doc.metadata.get("doc_hash")
                if doc_hash and not is_duplicate(doc_hash, self.loaded_hashes):
                    documents.append(doc)
                    self.loaded_hashes.add(doc_hash)
        
        return documents
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load single file based on extension."""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return load_pdf_file(file_path)
        elif ext == ".txt":
            return load_text_file(file_path)
        elif ext == ".csv":
            return load_csv_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []


# Backwards-compat alias
InsuranceDocumentLoader = DocumentLoader
