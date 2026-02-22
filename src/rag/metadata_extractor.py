"""Metadata extraction utilities - dataset-agnostic."""

import re
import hashlib
from pathlib import Path
from typing import Dict, Any


def extract_doc_metadata(text: str, filename: str) -> Dict[str, Any]:
    """
    Extract generic metadata from any document.
    Stores source filename; optionally enriches with SEC fields if the file
    looks like a 10-K (ticker pattern, fiscal year) – those become optional extras.
    """
    metadata: Dict[str, Any] = {"source": filename}

    # Optional: SEC 10-K enrichment (only fires when filename matches ticker pattern)
    ticker_match = re.search(r"([A-Z]{1,5})_\d+", filename)
    if ticker_match:
        metadata["company_ticker"] = ticker_match.group(1)
        year_patterns = [
            r"fiscal\s+year\s+ended?\s+\w+\s+\d+,?\s+(\d{4})",
            r"YEAR ENDED\s+\w+\s+\d+,?\s+(\d{4})",
        ]
        for pattern in year_patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                metadata["fiscal_year"] = match.group(1)
                break

    return metadata


# Backwards-compat alias used by existing callers
def extract_sec_metadata(text: str, filename: str) -> Dict[str, Any]:
    return extract_doc_metadata(text, filename)


def extract_document_hash(text: str) -> str:
    """Generate hash for deduplication."""
    return hashlib.md5(text[:1000].encode()).hexdigest()


def is_duplicate(doc_hash: str, loaded_hashes: set) -> bool:
    """Check if document is already loaded."""
    return doc_hash in loaded_hashes
