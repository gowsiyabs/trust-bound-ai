"""CSV file loader - converts CSV rows into LlamaIndex Documents.

Strategy:
- Each row becomes one Document with text formatted as "field: value" pairs.
- Groups of `rows_per_chunk` rows are batched into a single Document to avoid
  creating thousands of tiny chunks that overwhelm the vector store.
- Metadata stores the source filename and column names.
"""

import csv
from pathlib import Path
from typing import List

from llama_index.core import Document

from .metadata_extractor import extract_document_hash
from ..utils.logging import get_logger

logger = get_logger(__name__)

# How many CSV rows to combine into one Document. Tune via configs if needed.
ROWS_PER_CHUNK = 20


def load_csv_file(file_path: Path, rows_per_chunk: int = ROWS_PER_CHUNK) -> List[Document]:
    """Load a CSV file and return a list of Documents (one per row batch)."""
    try:
        rows = _read_csv(file_path)
        if not rows:
            logger.warning(f"CSV is empty: {file_path.name}")
            return []

        columns = list(rows[0].keys())
        logger.info(f"CSV {file_path.name}: {len(rows)} rows, columns: {columns}")

        documents = []
        for batch_start in range(0, len(rows), rows_per_chunk):
            batch = rows[batch_start : batch_start + rows_per_chunk]
            text = _rows_to_text(batch, columns)

            if len(text.strip()) < 20:
                continue

            doc_hash = extract_document_hash(text)
            doc = Document(
                text=text,
                metadata={
                    "source": file_path.name,
                    "file_type": "csv",
                    "columns": ", ".join(columns),
                    "row_range": f"{batch_start + 1}-{batch_start + len(batch)}",
                    "doc_hash": doc_hash,
                },
            )
            documents.append(doc)

        logger.info(f"Created {len(documents)} documents from {file_path.name}")
        return documents

    except Exception as e:
        logger.error(f"Error loading CSV {file_path.name}: {e}")
        return []


def _read_csv(file_path: Path) -> List[dict]:
    """Read CSV into list of dicts."""
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from values
            rows.append({k: v.strip() for k, v in row.items() if k})
    return rows


def _rows_to_text(rows: List[dict], columns: List[str]) -> str:
    """Convert a batch of rows into readable text for embedding."""
    parts = []
    for i, row in enumerate(rows, 1):
        fields = []
        for col in columns:
            val = row.get(col, "")
            if val:
                fields.append(f"{col}: {val}")
        if fields:
            parts.append(f"Entry {i}: " + " | ".join(fields))
    return "\n".join(parts)
