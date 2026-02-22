"""PDF file loader - dataset-agnostic."""

from typing import List
from pathlib import Path
import pdfplumber
from llama_index.core import Document

from .metadata_extractor import extract_doc_metadata, extract_document_hash
from .table_extractor import extract_tables_from_pdf
from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_pdf_file(file_path: Path) -> List[Document]:
    """Load single PDF file"""
    try:
        text_content = _extract_pdf_text(file_path)
        
        if not text_content or len(text_content.strip()) < 100:
            logger.warning(f"PDF has minimal content: {file_path.name}")
            return []
        
        metadata = extract_doc_metadata(text_content, file_path.name)
        doc_hash = extract_document_hash(text_content)
        metadata["doc_hash"] = doc_hash
        
        tables = extract_tables_from_pdf(str(file_path))
        if tables:
            metadata["has_tables"] = True
            text_content += "\n\n" + "\n\n".join(tables)
        
        doc = Document(text=text_content, metadata=metadata)
        return [doc]
        
    except Exception as e:
        logger.error(f"Error loading PDF {file_path.name}: {e}")
        return []


def _extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF"""
    text_parts = []
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    
    return "\n\n".join(text_parts)
