"""Text file loader - dataset-agnostic, with SEC 10-K preprocessing."""

from typing import List
from pathlib import Path
from llama_index.core import Document

from .metadata_extractor import extract_doc_metadata, extract_document_hash
from .sec_preprocessor import is_sec_filing, preprocess_sec_filing, preprocess_generic
from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_text_file(file_path: Path) -> List[Document]:
    """
    Load a single .txt file.

    SEC 10-K filings (detected by filename) are run through the full
    SEC preprocessor which strips EDGAR SGML wrappers, HTML/XBRL tags,
    boilerplate phrases, and duplicate paragraphs before embedding.
    This typically reduces token volume by 40-70%, directly cutting
    OpenAI API cost.

    All other .txt files get lightweight whitespace normalisation only.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        if not raw_text or len(raw_text.strip()) < 100:
            logger.warning(f"Text file has minimal content: {file_path.name}")
            return []

        # Apply appropriate cleaning
        if is_sec_filing(file_path.name):
            text_content = preprocess_sec_filing(raw_text, filename=file_path.name)
        else:
            text_content = preprocess_generic(raw_text)

        if len(text_content.strip()) < 100:
            logger.warning(f"After preprocessing, {file_path.name} has minimal content")
            return []

        metadata = extract_doc_metadata(text_content, file_path.name)
        doc_hash = extract_document_hash(text_content)
        metadata["doc_hash"] = doc_hash

        doc = Document(text=text_content, metadata=metadata)
        return [doc]

    except Exception as e:
        logger.error(f"Error loading text file {file_path.name}: {e}")
        return []
